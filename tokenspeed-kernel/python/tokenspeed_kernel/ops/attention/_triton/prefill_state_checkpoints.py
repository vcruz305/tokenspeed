# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Shared Triton checkpoint preparation for GDN and KDA recurrent prefill."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from tokenspeed_kernel._triton import tl, triton


@dataclass(frozen=True)
class PackedPrefillCheckpointInputs:
    """Token-packed inputs and row-packed initial state for one prefix scan."""

    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    recurrent_state: torch.Tensor
    a: torch.Tensor | None
    b: torch.Tensor | None
    g_raw: torch.Tensor | None
    f_a_out: torch.Tensor | None
    beta_raw: torch.Tensor | None


@triton.jit(
    do_not_specialize=["num_rows", "num_channels", "state_len"],
    do_not_specialize_on_alignment=["sequence_starts"],
)
def _write_prefill_conv_checkpoints_kernel(
    raw_inputs,
    conv_states,
    state_in_blocks,
    state_out_blocks,
    checkpoint_blocks,
    rows,
    sequence_starts,
    checkpoint_seq_lens,
    num_rows,
    num_channels,
    state_len,
    raw_token_stride,
    raw_channel_stride,
    state_page_stride,
    state_channel_stride,
    state_position_stride,
    STATE_BLOCK: tl.constexpr,
):
    checkpoint_row = tl.program_id(0)
    channel = tl.program_id(1)
    positions = tl.arange(0, STATE_BLOCK)
    live = (checkpoint_row < num_rows) & (channel < num_channels)
    position_live = live & (positions < state_len)

    request_row = tl.load(rows + checkpoint_row, mask=live, other=0)
    sequence_start = tl.load(sequence_starts + checkpoint_row, mask=live, other=0)
    checkpoint_len = tl.load(checkpoint_seq_lens + checkpoint_row, mask=live, other=0)
    # Page IDs fit int32, but their element offsets in large pools may not.
    state_in = tl.load(state_in_blocks + request_row, mask=live, other=0).to(tl.int64)
    state_out = tl.load(state_out_blocks + request_row, mask=live, other=0).to(tl.int64)
    source = tl.where(state_in > 0, state_in, state_out)
    destination = tl.load(checkpoint_blocks + request_row, mask=live, other=0).to(
        tl.int64
    )
    position_live = position_live & (destination >= 0)

    relative_token = checkpoint_len - state_len + positions
    from_raw = relative_token >= 0
    raw_offsets = (
        sequence_start + relative_token
    ) * raw_token_stride + channel * raw_channel_stride
    source_offsets = (
        source * state_page_stride
        + channel * state_channel_stride
        + (state_len + relative_token) * state_position_stride
    )
    # Load the complete old-state window before any store. This makes the
    # operation safe even if a caller aliases source and destination pages.
    raw_value = tl.load(raw_inputs + raw_offsets, mask=position_live & from_raw)
    source_value = tl.load(conv_states + source_offsets, mask=position_live & ~from_raw)
    value = tl.where(from_raw, raw_value, source_value)
    destination_offsets = (
        destination * state_page_stride
        + channel * state_channel_stride
        + positions * state_position_stride
    )
    tl.store(conv_states + destination_offsets, value, mask=position_live)


def _torch_write_prefill_conv_checkpoints(
    raw_inputs: torch.Tensor,
    conv_states: torch.Tensor,
    state_in_blocks: torch.Tensor,
    state_out_blocks: torch.Tensor,
    checkpoint_blocks: torch.Tensor,
    rows: torch.Tensor,
    sequence_starts: torch.Tensor,
    checkpoint_seq_lens: torch.Tensor,
) -> None:
    state_len = conv_states.shape[-1]
    active = checkpoint_blocks.index_select(0, rows) >= 0
    rows = rows[active]
    sequence_starts = sequence_starts[active]
    checkpoint_seq_lens = checkpoint_seq_lens[active]
    if rows.numel() == 0:
        return
    destinations = checkpoint_blocks.index_select(0, rows).to(torch.int64)
    state_in = state_in_blocks.index_select(0, rows)
    state_out = state_out_blocks.index_select(0, rows)
    sources = torch.where(state_in > 0, state_in, state_out).to(torch.int64)
    source_state = conv_states.index_select(0, sources)
    positions = torch.arange(state_len, dtype=torch.int64, device=raw_inputs.device)
    relative = checkpoint_seq_lens.unsqueeze(1) - state_len + positions.unsqueeze(0)
    raw_indices = sequence_starts.unsqueeze(1) + relative
    raw_indices.clamp_(min=0)
    raw_window = raw_inputs.index_select(0, raw_indices.flatten()).view(
        rows.numel(), state_len, raw_inputs.shape[1]
    )
    raw_window = raw_window.transpose(1, 2)
    source_positions = (state_len + relative).clamp_(min=0, max=state_len - 1)
    source_window = source_state.gather(
        2, source_positions.unsqueeze(1).expand(-1, source_state.shape[1], -1)
    )
    checkpoint_state = torch.where(
        (relative >= 0).unsqueeze(1), raw_window, source_window
    )
    conv_states.index_copy_(0, destinations, checkpoint_state)


def write_prefill_conv_checkpoints(
    raw_inputs: torch.Tensor,
    conv_states: torch.Tensor,
    state_in_blocks: torch.Tensor,
    state_out_blocks: torch.Tensor,
    checkpoint_blocks: torch.Tensor,
    rows: torch.Tensor,
    sequence_starts: torch.Tensor,
    checkpoint_seq_lens: torch.Tensor,
) -> None:
    """Write sparse convolution checkpoints with one launch.

    Args:
        raw_inputs: Token-major raw convolution inputs ``[tokens, channels]``.
        conv_states: State pool ``[pages, channels, state_len]``, updated in place.
        state_in_blocks: Per-request input state page ids.
        state_out_blocks: Per-request final output state page ids.
        checkpoint_blocks: Per-request internal-checkpoint destination page ids;
            negative destinations skip both state reads and writes.
        rows: Request rows that own an internal checkpoint.
        sequence_starts: Token offset of each selected request in ``raw_inputs``.
        checkpoint_seq_lens: Request-local length of each selected checkpoint.

    Returns:
        None. ``conv_states`` is updated in place.
    """
    if rows.numel() == 0 or conv_states.shape[-1] == 0:
        return
    if raw_inputs.ndim != 2 or conv_states.ndim != 3:
        raise ValueError("raw_inputs must be 2D and conv_states must be 3D")
    if not (rows.numel() == sequence_starts.numel() == checkpoint_seq_lens.numel()):
        raise ValueError("checkpoint row, start, and length counts must agree")
    if not conv_states.is_cuda:
        _torch_write_prefill_conv_checkpoints(
            raw_inputs,
            conv_states,
            state_in_blocks,
            state_out_blocks,
            checkpoint_blocks,
            rows,
            sequence_starts,
            checkpoint_seq_lens,
        )
        return
    state_len = conv_states.shape[-1]
    state_block = triton.next_power_of_2(state_len)
    _write_prefill_conv_checkpoints_kernel[(rows.numel(), conv_states.shape[1])](
        raw_inputs,
        conv_states,
        state_in_blocks,
        state_out_blocks,
        checkpoint_blocks,
        rows,
        sequence_starts,
        checkpoint_seq_lens,
        rows.numel(),
        conv_states.shape[1],
        state_len,
        raw_inputs.stride(0),
        raw_inputs.stride(1),
        conv_states.stride(0),
        conv_states.stride(1),
        conv_states.stride(2),
        STATE_BLOCK=state_block,
    )


@triton.jit
def _pack_token_rows(
    source,
    packed,
    source_token,
    token,
    live,
    SOURCE_STRIDE: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK_F: tl.constexpr,
):
    """Gather ``[BLOCK_T, WIDTH]`` token rows in ``BLOCK_F``-wide column tiles."""
    present = live & (source_token >= 0)
    source_rows = source + source_token.to(tl.int64)[:, None] * SOURCE_STRIDE
    packed_rows = packed + token.to(tl.int64)[:, None] * WIDTH
    for start in tl.static_range(0, WIDTH, BLOCK_F):
        feature = start + tl.arange(0, BLOCK_F)
        in_row = (feature < WIDTH)[None, :]
        values = tl.load(
            source_rows + feature[None, :], mask=present[:, None] & in_row, other=0
        )
        tl.store(packed_rows + feature[None, :], values, mask=live[:, None] & in_row)


@triton.jit(
    do_not_specialize=["num_tokens", "token_programs"],
    do_not_specialize_on_alignment=["token_indices"],
)
def _pack_prefill_recurrent_inputs_kernel(
    token_indices,
    rows,
    query,
    query_out,
    key,
    key_out,
    value,
    value_out,
    recurrent_state,
    recurrent_state_out,
    a,
    a_out,
    b,
    b_out,
    g_raw,
    g_raw_out,
    f_a_out,
    f_a_packed,
    beta_raw,
    beta_raw_out,
    num_tokens,
    token_programs,
    query_stride: tl.constexpr,
    key_stride: tl.constexpr,
    value_stride: tl.constexpr,
    state_stride: tl.constexpr,
    state_dim_1: tl.constexpr,
    state_dim_2: tl.constexpr,
    state_dim_3: tl.constexpr,
    state_stride_1: tl.constexpr,
    state_stride_2: tl.constexpr,
    state_stride_3: tl.constexpr,
    a_stride: tl.constexpr,
    b_stride: tl.constexpr,
    g_stride: tl.constexpr,
    f_a_stride: tl.constexpr,
    beta_stride: tl.constexpr,
    query_width: tl.constexpr,
    key_width: tl.constexpr,
    value_width: tl.constexpr,
    state_width: tl.constexpr,
    a_width: tl.constexpr,
    b_width: tl.constexpr,
    g_width: tl.constexpr,
    f_a_width: tl.constexpr,
    beta_width: tl.constexpr,
    HAS_A: tl.constexpr,
    HAS_B: tl.constexpr,
    HAS_G: tl.constexpr,
    HAS_F_A: tl.constexpr,
    HAS_BETA: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_F: tl.constexpr,
    STATE_BLOCK: tl.constexpr,
):
    """The first ``token_programs`` programs gather ``BLOCK_T`` token rows of
    every token input each; the rest copy one ``STATE_BLOCK`` chunk of one
    selected state row each. Token rows move in column tiles, so no
    per-element division recovers the token and its loads stay vectorized."""
    program = tl.program_id(0)
    if program < token_programs:
        token = program * BLOCK_T + tl.arange(0, BLOCK_T)
        live = token < num_tokens
        source_token = tl.load(token_indices + token, mask=live, other=-1)
        _pack_token_rows(
            query, query_out, source_token, token, live,
            query_stride, query_width, BLOCK_F,
        )  # fmt: skip
        _pack_token_rows(
            key, key_out, source_token, token, live, key_stride, key_width, BLOCK_F
        )
        _pack_token_rows(
            value, value_out, source_token, token, live,
            value_stride, value_width, BLOCK_F,
        )  # fmt: skip
        if HAS_A:
            _pack_token_rows(
                a, a_out, source_token, token, live, a_stride, a_width, BLOCK_F
            )
        if HAS_B:
            _pack_token_rows(
                b, b_out, source_token, token, live, b_stride, b_width, BLOCK_F
            )
        if HAS_G:
            _pack_token_rows(
                g_raw, g_raw_out, source_token, token, live,
                g_stride, g_width, BLOCK_F,
            )  # fmt: skip
        if HAS_F_A:
            _pack_token_rows(
                f_a_out, f_a_packed, source_token, token, live,
                f_a_stride, f_a_width, BLOCK_F,
            )  # fmt: skip
        if HAS_BETA:
            _pack_token_rows(
                beta_raw, beta_raw_out, source_token, token, live,
                beta_stride, beta_width, BLOCK_F,
            )  # fmt: skip
    else:
        STATE_CHUNKS: tl.constexpr = (state_width + STATE_BLOCK - 1) // STATE_BLOCK
        chunk = program - token_programs
        row = chunk // STATE_CHUNKS
        feature = (chunk % STATE_CHUNKS) * STATE_BLOCK + tl.arange(0, STATE_BLOCK)
        mask = feature < state_width
        source_row = tl.load(rows + row).to(tl.int64)
        dim_1_index = feature // (state_dim_2 * state_dim_3)
        remainder = feature % (state_dim_2 * state_dim_3)
        dim_2_index = remainder // state_dim_3
        dim_3_index = remainder % state_dim_3
        state_feature_offset = (
            dim_1_index * state_stride_1
            + dim_2_index * state_stride_2
            + dim_3_index * state_stride_3
        )
        tl.store(
            recurrent_state_out + row.to(tl.int64) * state_width + feature,
            tl.load(
                recurrent_state + source_row * state_stride + state_feature_offset,
                mask=mask,
            ),
            mask=mask,
        )


def _token_width(tensor: torch.Tensor, token_dim: int) -> int:
    return tensor.numel() // tensor.shape[token_dim]


def _allocate_token_pack(
    tensor: torch.Tensor | None, token_dim: int, num_tokens: int
) -> torch.Tensor | None:
    if tensor is None:
        return None
    shape = list(tensor.shape)
    shape[token_dim] = num_tokens
    return torch.empty(shape, dtype=tensor.dtype, device=tensor.device)


def pack_prefill_recurrent_checkpoint_inputs(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    recurrent_state: torch.Tensor,
    rows: torch.Tensor,
    token_indices: torch.Tensor,
    a: torch.Tensor | None,
    b: torch.Tensor | None,
    g_raw: torch.Tensor | None,
    f_a_out: torch.Tensor | None,
    beta_raw: torch.Tensor | None,
) -> PackedPrefillCheckpointInputs:
    """Pack every checkpoint prefix and its initial state with one GPU launch.

    Token dimension is 1 for query/key/value (with batch dimension 1) and 0
    for gate tensors. Recurrent state uses the four-dimensional ``[N, H, V,
    K]`` cache layout. Features within each token must be dense, but token and
    state dimensions may have noncontiguous strides, as with fused projection
    slices and transposed scan results. The returned tensors are contiguous and
    preserve the corresponding input shapes with packed token/request counts.

    Args:
        query: Query tensor with token dimension 1.
        key: Key tensor with token dimension 1.
        value: Value tensor with token dimension 1.
        recurrent_state: Per-request initial recurrent states. Row and feature
            strides may be noncontiguous, including transposed scan results.
        rows: Request rows selected for checkpoint scans.
        token_indices: Source-token indices for all selected checkpoint prefixes;
            negative entries produce zero rows for capacity padding.
        a: Optional token-major GDN scan input.
        b: Optional token-major GDN scan input.
        g_raw: Optional token-major KDA gate input.
        f_a_out: Optional token-major KDA gate input.
        beta_raw: Optional token-major KDA beta input.
    Returns:
        Packed scan inputs backed by newly allocated storage.
    """
    num_tokens = token_indices.numel()
    num_rows = rows.numel()
    optional_inputs = (a, b, g_raw, f_a_out, beta_raw)
    if recurrent_state.ndim != 4:
        raise ValueError("checkpoint recurrent state must be four-dimensional")
    if not query.is_cuda:

        def gather(tensor, dim):
            selected = tensor.index_select(dim, token_indices.clamp_min(0))
            shape = [1] * selected.ndim
            shape[dim] = num_tokens
            return selected.masked_fill((token_indices < 0).view(shape), 0)

        return PackedPrefillCheckpointInputs(
            query=gather(query, 1),
            key=gather(key, 1),
            value=gather(value, 1),
            recurrent_state=recurrent_state.index_select(0, rows),
            a=None if a is None else gather(a, 0),
            b=None if b is None else gather(b, 0),
            g_raw=None if g_raw is None else gather(g_raw, 0),
            f_a_out=(None if f_a_out is None else gather(f_a_out, 0)),
            beta_raw=(None if beta_raw is None else gather(beta_raw, 0)),
        )

    query_out = _allocate_token_pack(query, 1, num_tokens)
    key_out = _allocate_token_pack(key, 1, num_tokens)
    value_out = _allocate_token_pack(value, 1, num_tokens)
    assert query_out is not None and key_out is not None and value_out is not None
    state_out = torch.empty(
        (num_rows, *recurrent_state.shape[1:]),
        dtype=recurrent_state.dtype,
        device=recurrent_state.device,
    )
    optional_outputs = tuple(
        _allocate_token_pack(tensor, 0, num_tokens) for tensor in optional_inputs
    )

    tensors = (query, key, value, recurrent_state, *optional_inputs)
    token_dims = (1, 1, 1, 0, 0, 0, 0, 0, 0)
    for index, (tensor, token_dim) in enumerate(zip(tensors, token_dims)):
        if tensor is None:
            continue
        if token_dim == 1 and tensor.shape[0] != 1:
            raise ValueError("checkpoint pack query/key/value batch size must be 1")
        if index != 3 and not tensor.select(token_dim, 0).is_contiguous():
            raise ValueError("checkpoint pack inputs must have dense row features")
    strides = tuple(
        1 if tensor is None else tensor.stride(token_dim)
        for tensor, token_dim in zip(tensors, token_dims)
    )
    widths = (
        _token_width(query, 1),
        _token_width(key, 1),
        _token_width(value, 1),
        recurrent_state[0].numel(),
        *(
            1 if tensor is None else _token_width(tensor, 0)
            for tensor in optional_inputs
        ),
    )
    # Token rows move as [BLOCK_T, BLOCK_F] tiles; each selected state row
    # splits into STATE_BLOCK chunks. Both program counts are runtime values.
    block_t = 4
    block_f = min(1024, triton.next_power_of_2(max(*widths[:3], *widths[4:])))
    state_block = min(4096, triton.next_power_of_2(widths[3]))
    token_programs = triton.cdiv(num_tokens, block_t)
    state_programs = num_rows * triton.cdiv(widths[3], state_block)
    if token_programs + state_programs == 0:
        return PackedPrefillCheckpointInputs(
            query=query_out,
            key=key_out,
            value=value_out,
            recurrent_state=state_out,
            a=optional_outputs[0],
            b=optional_outputs[1],
            g_raw=optional_outputs[2],
            f_a_out=optional_outputs[3],
            beta_raw=optional_outputs[4],
        )
    dummy = query
    kernel_outputs = tuple(
        dummy if tensor is None else tensor for tensor in optional_outputs
    )
    kernel_inputs = tuple(
        dummy if tensor is None else tensor for tensor in optional_inputs
    )
    _pack_prefill_recurrent_inputs_kernel[(token_programs + state_programs,)](
        token_indices,
        rows,
        query,
        query_out,
        key,
        key_out,
        value,
        value_out,
        recurrent_state,
        state_out,
        kernel_inputs[0],
        kernel_outputs[0],
        kernel_inputs[1],
        kernel_outputs[1],
        kernel_inputs[2],
        kernel_outputs[2],
        kernel_inputs[3],
        kernel_outputs[3],
        kernel_inputs[4],
        kernel_outputs[4],
        num_tokens,
        token_programs,
        query_stride=strides[0],
        key_stride=strides[1],
        value_stride=strides[2],
        state_stride=strides[3],
        state_dim_1=recurrent_state.shape[1],
        state_dim_2=recurrent_state.shape[2],
        state_dim_3=recurrent_state.shape[3],
        state_stride_1=recurrent_state.stride(1),
        state_stride_2=recurrent_state.stride(2),
        state_stride_3=recurrent_state.stride(3),
        a_stride=strides[4],
        b_stride=strides[5],
        g_stride=strides[6],
        f_a_stride=strides[7],
        beta_stride=strides[8],
        query_width=widths[0],
        key_width=widths[1],
        value_width=widths[2],
        state_width=widths[3],
        a_width=widths[4],
        b_width=widths[5],
        g_width=widths[6],
        f_a_width=widths[7],
        beta_width=widths[8],
        HAS_A=a is not None,
        HAS_B=b is not None,
        HAS_G=g_raw is not None,
        HAS_F_A=f_a_out is not None,
        HAS_BETA=beta_raw is not None,
        BLOCK_T=block_t,
        BLOCK_F=block_f,
        STATE_BLOCK=state_block,
    )
    return PackedPrefillCheckpointInputs(
        query=query_out,
        key=key_out,
        value=value_out,
        recurrent_state=state_out,
        a=optional_outputs[0],
        b=optional_outputs[1],
        g_raw=optional_outputs[2],
        f_a_out=optional_outputs[3],
        beta_raw=optional_outputs[4],
    )


@triton.jit(do_not_specialize=["TOKENS"], do_not_specialize_on_alignment=["indices"])
def _scatter_checkpoint_output_kernel(
    source,
    indices,
    output,
    STRIDES: tl.constexpr,
    FEATURES: tl.constexpr,
    # Token counts follow the batch; runtime so every batch shape shares one
    # binary. The feature geometry is fixed per layer.
    TOKENS,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    token, feature = offset // WIDTH, offset % WIDTH
    destination = tl.load(indices + token, token < TOKENS, other=-1)
    source_offset = token * STRIDES[0]
    for dim in tl.static_range(len(FEATURES) - 1, -1, -1):
        source_offset += (feature % FEATURES[dim]) * STRIDES[dim + 1]
        feature = feature // FEATURES[dim]
    live = (token < TOKENS) & (destination >= 0)
    value = tl.load(source + source_offset, live, other=0)
    tl.store(output + destination * WIDTH + offset % WIDTH, value, live)


@triton.jit(
    do_not_specialize=["BODY_TOKENS", "TAIL_TOKENS", "TOKENS"],
    do_not_specialize_on_alignment=["sources"],
)
def _gather_checkpoint_output_kernel(
    body,
    tail,
    sources,
    output,
    BODY_STRIDES: tl.constexpr,
    TAIL_STRIDES: tl.constexpr,
    FEATURES: tl.constexpr,
    # Token counts follow the batch; runtime so every batch shape shares one
    # binary. The feature geometry is fixed per layer.
    BODY_TOKENS,
    TAIL_TOKENS,
    TOKENS,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    token, feature = offset // WIDTH, offset % WIDTH
    source = tl.load(sources + token, token < TOKENS, -1)
    is_body = (token < TOKENS) & (source >= 0) & (source < BODY_TOKENS)
    is_tail = (
        (token < TOKENS)
        & (source >= BODY_TOKENS)
        & (source < BODY_TOKENS + TAIL_TOKENS)
    )
    body_offset = source * BODY_STRIDES[0]
    tail_offset = (source - BODY_TOKENS) * TAIL_STRIDES[0]
    for dim in tl.static_range(len(FEATURES) - 1, -1, -1):
        component = feature % FEATURES[dim]
        feature //= FEATURES[dim]
        body_offset += component * BODY_STRIDES[dim + 1]
        tail_offset += component * TAIL_STRIDES[dim + 1]
    body_value = tl.load(body + body_offset, is_body, 0)
    tail_value = tl.load(tail + tail_offset, is_tail, 0)
    tl.store(output + offset, tl.where(is_body, body_value, tail_value), token < TOKENS)


def merge_prefill_checkpoint_outputs(
    body: torch.Tensor,
    tail: torch.Tensor,
    body_indices: torch.Tensor,
    tail_indices: torch.Tensor,
    token_dim: int,
    token_extent: int,
    output_sources: torch.Tensor | None,
) -> torch.Tensor:
    """Restore packed body/tail outputs to original token order.

    Without a CUDA inverse map, zero-initialize and scatter each scan's output.
    With one, a single gather writes the full output, including zero padding.
    The caller guarantees valid indices are in range. This copies token outputs,
    not checkpoint state.

    Args:
        body: Body scan output, with dense or strided feature dimensions.
        tail: Tail scan output with the same feature geometry as body.
        body_indices: Original token indices, with negative padding entries.
        tail_indices: Original token indices, disjoint from body indices.
        token_dim: Token axis; any preceding dimensions must be singleton.
        token_extent: Output token capacity; unwritten padding stays zero.
        output_sources: Optional per-forward inverse token map. Nonnegative
            entries index concatenated body/tail tokens; negative entries
            produce zero. Shared across layers, including graph replays.

    Returns:
        Contiguous output with the original token order and zero bucket padding.
        Nonnegative indices must cover each live output token exactly once.
    """
    if any(size != 1 for size in body.shape[:token_dim]):
        raise ValueError("checkpoint outputs require singleton leading dimensions")
    for source, indices in ((body, body_indices), (tail, tail_indices)):
        if source.shape[token_dim] != indices.numel():
            raise ValueError("checkpoint output extent differs from token indices")
    shape = list(body.shape)
    shape[token_dim] = token_extent
    if output_sources is not None and body.is_cuda:
        if output_sources.numel() != token_extent:
            raise ValueError("checkpoint inverse map differs from output extent")
        if (
            body.shape[token_dim + 1 :] != tail.shape[token_dim + 1 :]
            or body.dtype != tail.dtype
        ):
            raise ValueError("checkpoint outputs have different feature geometry")
        output = torch.empty(shape, dtype=body.dtype, device=body.device)
        width = body.numel() // body.shape[token_dim]
        _gather_checkpoint_output_kernel[(triton.cdiv(output.numel(), 1024),)](
            body,
            tail,
            output_sources,
            output,
            BODY_STRIDES=body.stride()[token_dim:],
            TAIL_STRIDES=tail.stride()[token_dim:],
            FEATURES=body.shape[token_dim + 1 :],
            BODY_TOKENS=body.shape[token_dim],
            TAIL_TOKENS=tail.shape[token_dim],
            TOKENS=token_extent,
            WIDTH=width,
            BLOCK=1024,
        )
        return output
    output = torch.zeros(shape, dtype=body.dtype, device=body.device)
    for source, indices in ((body, body_indices), (tail, tail_indices)):
        if not source.is_cuda:
            live = indices >= 0
            output.index_copy_(
                token_dim,
                indices[live],
                source.index_select(token_dim, torch.nonzero(live).flatten()),
            )
            continue
        width = source.numel() // source.shape[token_dim]
        _scatter_checkpoint_output_kernel[(triton.cdiv(source.numel(), 256),)](
            source,
            indices,
            output,
            STRIDES=source.stride()[token_dim:],
            FEATURES=source.shape[token_dim + 1 :],
            TOKENS=indices.numel(),
            WIDTH=width,
            BLOCK=256,
        )
    return output


@triton.jit(do_not_specialize=["num_rows", "state_width"])
def _write_prefill_recurrent_checkpoints_kernel(
    checkpoint_state,
    ssm_states,
    checkpoint_blocks,
    rows,
    num_rows,
    state_width,
    state_dim_2,
    state_dim_3,
    checkpoint_stride_0,
    checkpoint_stride_1,
    checkpoint_stride_2,
    checkpoint_stride_3,
    pool_stride_0,
    pool_stride_1,
    pool_stride_2,
    pool_stride_3,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row = offsets // state_width
    feature = offsets % state_width
    live = (row < num_rows) & (feature < state_width)
    request_row = tl.load(rows + row, mask=live, other=0)
    live = live & (request_row >= 0)
    destination = tl.load(checkpoint_blocks + request_row, mask=live, other=-1).to(
        tl.int64
    )
    live = live & (destination >= 0)
    dim_1_index = feature // (state_dim_2 * state_dim_3)
    remainder = feature % (state_dim_2 * state_dim_3)
    dim_2_index = remainder // state_dim_3
    dim_3_index = remainder % state_dim_3
    checkpoint_offset = (
        row * checkpoint_stride_0
        + dim_1_index * checkpoint_stride_1
        + dim_2_index * checkpoint_stride_2
        + dim_3_index * checkpoint_stride_3
    )
    pool_offset = (
        destination * pool_stride_0
        + dim_1_index * pool_stride_1
        + dim_2_index * pool_stride_2
        + dim_3_index * pool_stride_3
    )
    value = tl.load(checkpoint_state + checkpoint_offset, mask=live)
    tl.store(ssm_states + pool_offset, value, mask=live)


def write_prefill_recurrent_checkpoints(
    checkpoint_state: torch.Tensor,
    ssm_states: torch.Tensor,
    checkpoint_blocks: torch.Tensor,
    rows: torch.Tensor,
) -> None:
    """Scatter packed recurrent states into a pool or temporary state tensor.

    Args:
        checkpoint_state: Scan results, one state per selected row; strides
            determine the state layout.
        ssm_states: Recurrent-state destination, updated in place. May also be
            temporary body final states receiving the active tail states.
        checkpoint_blocks: Per-request destination indices into ``ssm_states``;
            these are pool block IDs or temporary state row numbers. Negative
            destinations skip the write. Destination zero is valid.
        rows: Request rows corresponding to ``checkpoint_state``; negative
            rows skip the write without reading a destination.

    Returns:
        None. ``ssm_states`` is updated in place.
    """
    if rows.numel() == 0:
        return
    if checkpoint_state.ndim != 4 or ssm_states.ndim != 4:
        raise ValueError("recurrent checkpoint states must be four-dimensional")
    if checkpoint_state.shape[1:] != ssm_states.shape[1:]:
        raise ValueError("checkpoint and pool recurrent-state shapes must agree")
    if not checkpoint_state.is_cuda:
        active = rows >= 0
        destinations = checkpoint_blocks.index_select(0, rows[active]).to(torch.int64)
        values = checkpoint_state[active]
        live = destinations >= 0
        ssm_states.index_copy_(0, destinations[live], values[live].to(ssm_states.dtype))
        return
    state_width = checkpoint_state[0].numel()
    block = 256
    _write_prefill_recurrent_checkpoints_kernel[
        (triton.cdiv(rows.numel() * state_width, block),)
    ](
        checkpoint_state,
        ssm_states,
        checkpoint_blocks,
        rows,
        rows.numel(),
        state_width,
        checkpoint_state.shape[2],
        checkpoint_state.shape[3],
        checkpoint_state.stride(0),
        checkpoint_state.stride(1),
        checkpoint_state.stride(2),
        checkpoint_state.stride(3),
        ssm_states.stride(0),
        ssm_states.stride(1),
        ssm_states.stride(2),
        ssm_states.stride(3),
        BLOCK=block,
    )
