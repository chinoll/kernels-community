import torch
from torch import Tensor

from ._ops import ops


HEAD_SIZE = 64
CHUNK_LEN = 16


class _Sequence(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        r: Tensor,
        w: Tensor,
        k: Tensor,
        v: Tensor,
        a: Tensor,
        b: Tensor,
    ) -> Tensor:
        r, w, k, v, a, b = _prepare_sequence_inputs(r, w, k, v, a, b)
        batch_size, seq_len, num_heads, head_size = r.shape
        chunks = seq_len // CHUNK_LEN

        y = torch.empty_like(r)
        saved_states = torch.empty(
            batch_size,
            num_heads,
            chunks,
            head_size,
            head_size,
            device=r.device,
            dtype=torch.float32,
        )
        saved_sa = torch.empty(batch_size, seq_len, num_heads, head_size, device=r.device, dtype=torch.float32)

        ops.sequence_forward(r, w, k, v, a, b, y, saved_states, saved_sa)

        ctx.save_for_backward(r, w, k, v, a, b, saved_states, saved_sa)
        return y

    @staticmethod
    def backward(ctx, grad_y: Tensor):
        r, w, k, v, a, b, saved_states, saved_sa = ctx.saved_tensors

        if grad_y is None:
            grad_y = torch.zeros_like(r)
        elif grad_y.dtype != torch.bfloat16:
            grad_y = grad_y.to(dtype=torch.bfloat16)
        grad_y = grad_y.contiguous()

        grad_r = torch.empty_like(r)
        grad_w = torch.empty_like(w)
        grad_k = torch.empty_like(k)
        grad_v = torch.empty_like(v)
        grad_a = torch.empty_like(a)
        grad_b = torch.empty_like(b)

        ops.sequence_backward(
            r,
            w,
            k,
            v,
            a,
            b,
            grad_y,
            saved_states,
            saved_sa,
            grad_r,
            grad_w,
            grad_k,
            grad_v,
            grad_a,
            grad_b,
        )

        return grad_r, grad_w, grad_k, grad_v, grad_a, grad_b


class _StatePassing(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        state: Tensor,
        r: Tensor,
        w: Tensor,
        k: Tensor,
        v: Tensor,
        a: Tensor,
        b: Tensor,
    ) -> tuple[Tensor, Tensor]:
        state, r, w, k, v, a, b = _prepare_inputs(state, r, w, k, v, a, b)
        batch_size, seq_len, num_heads, head_size = r.shape
        chunks = seq_len // CHUNK_LEN

        y = torch.empty_like(r)
        state_out = torch.empty_like(state)
        saved_states = torch.empty(
            batch_size,
            num_heads,
            chunks,
            head_size,
            head_size,
            device=r.device,
            dtype=torch.float32,
        )
        saved_sa = torch.empty(batch_size, seq_len, num_heads, head_size, device=r.device, dtype=torch.float32)

        ops.state_passing_forward(state, r, w, k, v, a, b, y, state_out, saved_states, saved_sa)

        ctx.save_for_backward(r, w, k, v, a, b, saved_states, saved_sa)
        ctx.shape = (batch_size, seq_len, num_heads, head_size)
        return y, state_out

    @staticmethod
    def backward(ctx, grad_y: Tensor, grad_state_out: Tensor):
        r, w, k, v, a, b, saved_states, saved_sa = ctx.saved_tensors
        batch_size, seq_len, num_heads, head_size = ctx.shape

        if grad_y is None:
            grad_y = torch.zeros_like(r)
        elif grad_y.dtype != torch.bfloat16:
            grad_y = grad_y.to(dtype=torch.bfloat16)
        grad_y = grad_y.contiguous()

        if grad_state_out is None:
            grad_state_out = torch.zeros(
                batch_size,
                num_heads,
                head_size,
                head_size,
                device=r.device,
                dtype=torch.float32,
            )
        elif grad_state_out.dtype != torch.float32:
            grad_state_out = grad_state_out.to(dtype=torch.float32)
        grad_state_out = grad_state_out.contiguous()

        grad_state = torch.empty(batch_size, num_heads, head_size, head_size, device=r.device, dtype=torch.float32)
        grad_r = torch.empty_like(r)
        grad_w = torch.empty_like(w)
        grad_k = torch.empty_like(k)
        grad_v = torch.empty_like(v)
        grad_a = torch.empty_like(a)
        grad_b = torch.empty_like(b)

        ops.state_passing_backward(
            r,
            w,
            k,
            v,
            a,
            b,
            grad_y,
            grad_state_out,
            saved_states,
            saved_sa,
            grad_state,
            grad_r,
            grad_w,
            grad_k,
            grad_v,
            grad_a,
            grad_b,
        )

        return grad_state, grad_r, grad_w, grad_k, grad_v, grad_a, grad_b


def sequence(r: Tensor, w: Tensor, k: Tensor, v: Tensor, a: Tensor, b: Tensor) -> Tensor:
    """Run the RWKV7 BF16 sequence time-mix kernel from zero initial state.

    This mirrors RWKV-CUDA ``rwkv7_clampw`` and is intended for full-sequence
    calls that do not need the final state.

    Args:
        r: Receptance tensor, shape ``[B, T, H, 64]``, dtype ``torch.bfloat16``.
        w: Raw RWKV7 decay logits, shape ``[B, T, H, 64]``, dtype ``torch.bfloat16``.
        k: Key tensor, shape ``[B, T, H, 64]``, dtype ``torch.bfloat16``.
        v: Value tensor, shape ``[B, T, H, 64]``, dtype ``torch.bfloat16``.
        a: In-context state tensor, shape ``[B, T, H, 64]``, dtype ``torch.bfloat16``.
        b: In-context rate tensor, shape ``[B, T, H, 64]``, dtype ``torch.bfloat16``.

    Returns:
        The BF16 output tensor, shape ``[B, T, H, 64]``.
    """
    return _Sequence.apply(r, w, k, v, a, b)


def state_passing(state: Tensor, r: Tensor, w: Tensor, k: Tensor, v: Tensor, a: Tensor, b: Tensor) -> tuple[Tensor, Tensor]:
    """Run the RWKV7 BF16 state-passing time-mix kernel.

    Args:
        state: Initial matrix state, shape ``[B, H, 64, 64]``, dtype ``torch.float32``.
        r: Receptance tensor, shape ``[B, T, H, 64]``, dtype ``torch.bfloat16``.
        w: Raw RWKV7 decay logits, shape ``[B, T, H, 64]``, dtype ``torch.bfloat16``.
        k: Key tensor, shape ``[B, T, H, 64]``, dtype ``torch.bfloat16``.
        v: Value tensor, shape ``[B, T, H, 64]``, dtype ``torch.bfloat16``.
        a: In-context state tensor, shape ``[B, T, H, 64]``, dtype ``torch.bfloat16``.
        b: In-context rate tensor, shape ``[B, T, H, 64]``, dtype ``torch.bfloat16``.

    Returns:
        A tuple ``(y, state_out)`` containing the BF16 output and float32 final state.
    """
    return _StatePassing.apply(state, r, w, k, v, a, b)


def _prepare_inputs(state: Tensor, r: Tensor, w: Tensor, k: Tensor, v: Tensor, a: Tensor, b: Tensor):
    _validate_inputs(state, r, w, k, v, a, b)
    return (
        state.contiguous(),
        r.contiguous(),
        w.contiguous(),
        k.contiguous(),
        v.contiguous(),
        a.contiguous(),
        b.contiguous(),
    )


def _prepare_sequence_inputs(r: Tensor, w: Tensor, k: Tensor, v: Tensor, a: Tensor, b: Tensor):
    _validate_sequence_inputs(r, w, k, v, a, b)
    return (
        r.contiguous(),
        w.contiguous(),
        k.contiguous(),
        v.contiguous(),
        a.contiguous(),
        b.contiguous(),
    )


def _validate_inputs(state: Tensor, r: Tensor, w: Tensor, k: Tensor, v: Tensor, a: Tensor, b: Tensor) -> None:
    _validate_sequence_inputs(r, w, k, v, a, b)

    batch_size, seq_len, num_heads, head_size = r.shape
    expected_state_shape = (batch_size, num_heads, head_size, head_size)

    if state.shape != expected_state_shape:
        raise RuntimeError(f"RWKV7 state must have shape {expected_state_shape}, got {tuple(state.shape)}")
    if state.dtype != torch.float32:
        raise RuntimeError(f"RWKV7 state must use torch.float32, got {state.dtype}")
    _validate_device_match((state, r))


def _validate_sequence_inputs(r: Tensor, w: Tensor, k: Tensor, v: Tensor, a: Tensor, b: Tensor) -> None:
    tensors = (r, w, k, v, a, b)
    _validate_device_match(tensors)

    if r.ndim != 4:
        raise RuntimeError(f"RWKV7 inputs must have shape [B, T, H, {HEAD_SIZE}], got {tuple(r.shape)}")
    batch_size, seq_len, num_heads, head_size = r.shape
    expected_input_shape = (batch_size, seq_len, num_heads, head_size)

    for name, tensor in (("w", w), ("k", k), ("v", v), ("a", a), ("b", b)):
        if tensor.shape != expected_input_shape:
            raise RuntimeError(f"RWKV7 {name} must have shape {expected_input_shape}, got {tuple(tensor.shape)}")

    if head_size != HEAD_SIZE:
        raise RuntimeError(f"RWKV7 kernel was built for head size {HEAD_SIZE}, got {head_size}")
    if seq_len % CHUNK_LEN != 0:
        raise RuntimeError(f"RWKV7 sequence length must be divisible by {CHUNK_LEN}, got {seq_len}")

    for name, tensor in (("r", r), ("w", w), ("k", k), ("v", v), ("a", a), ("b", b)):
        if tensor.dtype != torch.bfloat16:
            raise RuntimeError(f"RWKV7 {name} must use torch.bfloat16, got {tensor.dtype}")


def _validate_device_match(tensors: tuple[Tensor, ...]) -> None:
    device = tensors[0].device
    if device.type != "cuda":
        raise RuntimeError("RWKV7 CUDA ops require CUDA tensors")
    for tensor in tensors[1:]:
        if tensor.device != device:
            raise RuntimeError("All RWKV7 tensors must be on the same CUDA device")


__all__ = ["CHUNK_LEN", "HEAD_SIZE", "sequence", "state_passing"]
