---
license: apache-2.0
tags:
  - kernel
---

![Status](https://hubwebhook.dholtz.com/shield?repo=kernels-community/rwkv7)

## RWKV7

CUDA BF16 state-passing kernel for RWKV7 time mixing.

This kernel was ported from the RWKV7 fast fused CUDA implementation in
[BlinkDL/RWKV-CUDA](https://github.com/BlinkDL/RWKV-CUDA/tree/main/rwkv7_fast_fused).
Original source copyright belongs to Bo Peng (BlinkDL). The port adapts the
kernel to the `kernels` packaging and Torch custom op registration conventions.

The public Python API is:

```python
state_passing(state, r, w, k, v, a, b) -> tuple[torch.Tensor, torch.Tensor]
```

where `r`, `w`, `k`, `v`, `a`, and `b` are BF16 tensors with shape
`[batch, sequence, heads, 64]`. `state` is a float32 tensor with shape
`[batch, heads, 64, 64]`. The kernel returns the BF16 output `y` and the
float32 final state.

`w` must be the raw RWKV7 decay logits. The kernel applies the RWKV7 decay
transform internally:

```text
decay = exp(-exp(-0.5) * sigmoid(w))
```

The current build is specialized for `head_size=64` and `chunk_len=16`, matching
the published RWKV7 G1 checkpoints. Sequence length must be divisible by 16.

## Usage

```python
# /// script
# dependencies = [
#   "torch",
#   "kernels",
# ]
# ///
import torch
from kernels import get_kernel

rwkv7 = get_kernel("kernels-community/rwkv7")

B, T, H, N = 2, 128, 4, 64
device = torch.device("cuda")
state = torch.zeros(B, H, N, N, device=device, dtype=torch.float32)
r = torch.randn(B, T, H, N, device=device, dtype=torch.bfloat16)
w = torch.randn(B, T, H, N, device=device, dtype=torch.bfloat16)
k = torch.randn(B, T, H, N, device=device, dtype=torch.bfloat16)
v = torch.randn(B, T, H, N, device=device, dtype=torch.bfloat16)
a = torch.randn(B, T, H, N, device=device, dtype=torch.bfloat16)
b = torch.randn(B, T, H, N, device=device, dtype=torch.bfloat16)

y, state_out = rwkv7.state_passing(state, r, w, k, v, a, b)

print(y.shape, state_out.shape)
```
