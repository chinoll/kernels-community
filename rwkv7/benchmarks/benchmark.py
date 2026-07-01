import argparse
import time

import rwkv7
import torch


def benchmark(batch_size, seq_len, num_heads, warmup, steps):
    device = torch.device("cuda")
    shape = (batch_size, seq_len, num_heads, rwkv7.HEAD_SIZE)
    state = torch.zeros(batch_size, num_heads, rwkv7.HEAD_SIZE, rwkv7.HEAD_SIZE, device=device, dtype=torch.float32)
    r, w, k, v, a, b = [torch.randn(*shape, device=device, dtype=torch.bfloat16) for _ in range(6)]

    for _ in range(warmup):
        rwkv7.state_passing(state, r, w, k, v, a, b)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(steps):
        rwkv7.state_passing(state, r, w, k, v, a, b)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / steps

    tokens = batch_size * seq_len
    print(f"B={batch_size} T={seq_len} H={num_heads} latency={elapsed * 1000:.3f} ms tokens/s={tokens / elapsed:.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--num-heads", type=int, default=12)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--steps", type=int, default=50)
    args = parser.parse_args()
    benchmark(args.batch_size, args.seq_len, args.num_heads, args.warmup, args.steps)
