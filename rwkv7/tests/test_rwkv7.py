import pytest
import rwkv7
import torch
import torch.nn.functional as F


def _reference_decay(w):
    log_decay = (-F.softplus(-w.float()) - 0.5).to(w.dtype)
    return torch.exp(-torch.exp(log_decay.float()))


def _reference_state_passing(state, r, w, k, v, a, b):
    state = state.float().clone()
    output = torch.empty_like(r)
    decay = _reference_decay(w)

    for token_index in range(r.size(1)):
        rt = r[:, token_index].float()
        kt = k[:, token_index].float()
        vt = v[:, token_index].float()
        at = a[:, token_index].float()
        bt = b[:, token_index].float()
        wt = decay[:, token_index]

        sa = torch.einsum("bhij,bhj->bhi", state, at)
        state = (
            state * wt.unsqueeze(-2)
            + torch.einsum("bhi,bhj->bhij", sa, bt)
            + torch.einsum("bhi,bhj->bhij", vt, kt)
        )
        output[:, token_index] = torch.einsum("bhij,bhj->bhi", state, rt).to(r.dtype)

    return output, state


def _reference_sequence(r, w, k, v, a, b):
    state = torch.zeros(
        r.size(0),
        r.size(2),
        r.size(3),
        r.size(3),
        device=r.device,
        dtype=torch.float32,
    )
    output, _ = _reference_state_passing(state, r, w, k, v, a, b)
    return output


def _make_inputs(batch_size=2, seq_len=32, num_heads=2):
    torch.manual_seed(0)
    shape = (batch_size, seq_len, num_heads, rwkv7.HEAD_SIZE)
    device = torch.device("cuda")
    state = torch.randn(
        batch_size,
        num_heads,
        rwkv7.HEAD_SIZE,
        rwkv7.HEAD_SIZE,
        device=device,
        dtype=torch.float32,
    ) * 0.01
    tensors = [torch.randn(*shape, device=device, dtype=torch.bfloat16) * 0.1 for _ in range(6)]
    return (state, *tensors)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="RWKV7 kernel requires CUDA")
def test_sequence_matches_rwkv_lm_fast_reference():
    _, r, w, k, v, a, b = _make_inputs()

    expected_y = _reference_sequence(r, w, k, v, a, b)
    actual_y = rwkv7.sequence(r, w, k, v, a, b)

    torch.testing.assert_close(actual_y.float(), expected_y.float(), rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="RWKV7 kernel requires CUDA")
def test_sequence_backward_matches_reference():
    _, *source_inputs = _make_inputs(batch_size=1, seq_len=16, num_heads=1)
    actual_inputs = tuple(t.detach().requires_grad_(True) for t in source_inputs)
    expected_inputs = tuple(t.detach().requires_grad_(True) for t in source_inputs)

    y = rwkv7.sequence(*actual_inputs)
    expected_y = _reference_sequence(*expected_inputs)

    torch.manual_seed(1)
    grad_y = torch.randn_like(y)

    actual_loss = (y.float() * grad_y.float()).sum()
    expected_loss = (expected_y.float() * grad_y.float()).sum()

    actual_loss.backward()
    expected_loss.backward()

    for actual, expected in zip(actual_inputs, expected_inputs):
        torch.testing.assert_close(actual.grad.float(), expected.grad.float(), rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="RWKV7 kernel requires CUDA")
def test_state_passing_matches_rwkv_lm_fast_reference():
    state, r, w, k, v, a, b = _make_inputs()

    expected_y, expected_state = _reference_state_passing(state, r, w, k, v, a, b)
    actual_y, actual_state = rwkv7.state_passing(state, r, w, k, v, a, b)

    torch.testing.assert_close(actual_y.float(), expected_y.float(), rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(actual_state, expected_state, rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="RWKV7 kernel requires CUDA")
def test_state_passing_backward_matches_reference():
    source_inputs = _make_inputs(batch_size=1, seq_len=16, num_heads=1)
    actual_inputs = tuple(t.detach().requires_grad_(True) for t in source_inputs)
    expected_inputs = tuple(t.detach().requires_grad_(True) for t in source_inputs)

    y, state_out = rwkv7.state_passing(*actual_inputs)
    expected_y, expected_state_out = _reference_state_passing(*expected_inputs)

    torch.manual_seed(1)
    grad_y = torch.randn_like(y)
    grad_state_out = torch.randn_like(state_out) * 0.01

    actual_loss = (y.float() * grad_y.float()).sum() + (state_out * grad_state_out).sum()
    expected_loss = (expected_y.float() * grad_y.float()).sum() + (expected_state_out * grad_state_out).sum()

    actual_loss.backward()
    expected_loss.backward()

    for actual, expected in zip(actual_inputs, expected_inputs):
        torch.testing.assert_close(actual.grad.float(), expected.grad.float(), rtol=1e-2, atol=1e-2)
