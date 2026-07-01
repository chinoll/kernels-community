// SPDX-FileCopyrightText: Copyright (c) Bo Peng (BlinkDL)
// SPDX-FileCopyrightText: Copyright (c) 2026 Hugging Face Inc. and contributors
// SPDX-License-Identifier: Apache-2.0
//
// Ported from BlinkDL/RWKV-CUDA rwkv7_fast_fused/cuda/rwkv7_clampw.cu:
// https://github.com/BlinkDL/RWKV-CUDA

#include <assert.h>
#include <cuda_bf16.h>

#ifndef _N_
#define _N_ 64
#endif

#ifndef _CHUNK_LEN_
#define _CHUNK_LEN_ 16
#endif

using bf = __nv_bfloat16;
using i64 = long long int;

using BFPtr = bf *__restrict__;
using CBFPtr = const bf *__restrict__;

constexpr float W_SCALE = -0.6065306597f;  // -exp(-0.5)

#define to_float(u) (__bfloat162float(u))
#define to_bf(u) (__float2bfloat16_rn(u))

template <int N>
__launch_bounds__(N, 2) __global__
void sequence_forward_kernel(
    int T,
    int H,
    CBFPtr r_,
    CBFPtr w_,
    CBFPtr k_,
    CBFPtr v_,
    CBFPtr a_,
    CBFPtr b_,
    BFPtr y_,
    float *__restrict__ s__,
    float *__restrict__ sa_) {
    const int bb = blockIdx.y;
    const int hh = blockIdx.x;
    const int i = threadIdx.x;

    float *__restrict__ s_ = s__ + i64(bb * H + hh) * i64((T / _CHUNK_LEN_) * N * N);

    float state[N];
#pragma unroll
    for (int j = 0; j < N; ++j) {
        state[j] = 0.0f;
    }

    __shared__ float r[N];
    __shared__ float w[N];
    __shared__ float k[N];
    __shared__ float a[N];
    __shared__ float b[N];

    for (int t = 0; t < T; ++t) {
        const int idx = ((bb * T + t) * H + hh) * N + i;

        __syncthreads();
        r[i] = to_float(r_[idx]);
        w[i] = __expf(W_SCALE / (1.0f + __expf(-to_float(w_[idx]))));
        k[i] = to_float(k_[idx]);
        a[i] = to_float(a_[idx]);
        b[i] = to_float(b_[idx]);
        __syncthreads();

        float sa = 0.0f;
#pragma unroll
        for (int j = 0; j < N; ++j) {
            sa += state[j] * a[j];
        }
        sa_[idx] = sa;

        const float vi = to_float(v_[idx]);
        float y = 0.0f;
#pragma unroll
        for (int j = 0; j < N; ++j) {
            float s = state[j];
            s = s * w[j] + (sa * b[j] + k[j] * vi);
            y += s * r[j];
            state[j] = s;
        }

        y_[idx] = to_bf(y);

        if ((t + 1) % _CHUNK_LEN_ == 0) {
            const int base = (t / _CHUNK_LEN_) * N * N + i;
#pragma unroll
            for (int j = 0; j < N; ++j) {
                s_[base + j * N] = state[j];
            }
        }
    }
}

void rwkv7_sequence_forward_cuda(
    int B,
    int T,
    int H,
    const void *r,
    const void *w,
    const void *k,
    const void *v,
    const void *a,
    const void *b,
    void *y,
    float *s,
    float *sa) {
    assert(T % _CHUNK_LEN_ == 0);
    sequence_forward_kernel<_N_><<<dim3(H, B), dim3(_N_)>>>(
        T,
        H,
        static_cast<CBFPtr>(r),
        static_cast<CBFPtr>(w),
        static_cast<CBFPtr>(k),
        static_cast<CBFPtr>(v),
        static_cast<CBFPtr>(a),
        static_cast<CBFPtr>(b),
        static_cast<BFPtr>(y),
        s,
        sa);
}

template <int N>
__global__ void sequence_backward_kernel(
    int T,
    int H,
    CBFPtr r_,
    CBFPtr w_,
    CBFPtr k_,
    CBFPtr v_,
    CBFPtr a_,
    CBFPtr b_,
    CBFPtr dy_,
    const float *__restrict__ s__,
    const float *__restrict__ sa_,
    BFPtr dr_,
    BFPtr dw_,
    BFPtr dk_,
    BFPtr dv_,
    BFPtr da_,
    BFPtr db_) {
    const int bb = blockIdx.y;
    const int hh = blockIdx.x;
    const int i = threadIdx.x;

    const float *__restrict__ s_ = s__ + i64(bb * H + hh) * i64((T / _CHUNK_LEN_) * N * N);

    float stateT[N] = {0.0f};
    float dstate[N] = {0.0f};
    float dstateT[N] = {0.0f};
    __shared__ float r[N];
    __shared__ float w[N];
    __shared__ float k[N];
    __shared__ float v[N];
    __shared__ float a[N];
    __shared__ float b[N];
    __shared__ float dy[N];
    __shared__ float sa[N];
    __shared__ float dSb_shared[N];

    float ri;
    float wi;
    float ki;
    float ai;
    float bi;
    float dyi;

    for (int t = T - 1; t >= 0; --t) {
        const int idx = bb * T * H * N + t * H * N + hh * N + i;

        __syncthreads();
        r[i] = ri = to_float(r_[idx]);
        const float w_sig = 1.0f / (1.0f + __expf(-to_float(w_[idx])));
        w[i] = wi = __expf(W_SCALE * w_sig);
        k[i] = ki = to_float(k_[idx]);
        v[i] = to_float(v_[idx]);
        a[i] = ai = to_float(a_[idx]);
        b[i] = bi = to_float(b_[idx]);
        dy[i] = dyi = to_float(dy_[idx]);
        sa[i] = sa_[idx];
        __syncthreads();

        if ((t + 1) % _CHUNK_LEN_ == 0) {
            const int base = (t / _CHUNK_LEN_) * N * N + i * N;
            const float4 *s4 = reinterpret_cast<const float4 *>(s_ + base);
#pragma unroll
            for (int j4 = 0; j4 < N / 4; ++j4) {
                const float4 q = s4[j4];
                const int j = j4 << 2;
                stateT[j + 0] = q.x;
                stateT[j + 1] = q.y;
                stateT[j + 2] = q.z;
                stateT[j + 3] = q.w;
            }
        }

        float dr = 0.0f;
#pragma unroll
        for (int j = 0; j < N; ++j) {
            dr += stateT[j] * dy[j];
        }
        dr_[idx] = to_bf(dr);

        const float iwi = 1.0f / wi;
#pragma unroll
        for (int j = 0; j < N; ++j) {
            stateT[j] = (stateT[j] - ki * v[j] - bi * sa[j]) * iwi;
            dstate[j] += dyi * r[j];
            dstateT[j] += ri * dy[j];
        }

        float dw = 0.0f;
        float dk = 0.0f;
        float dv = 0.0f;
        float db = 0.0f;
        float dSb = 0.0f;
#pragma unroll
        for (int j = 0; j < N; ++j) {
            dw += dstateT[j] * stateT[j];
            dk += dstateT[j] * v[j];
            dv += dstate[j] * k[j];
            dSb += dstate[j] * b[j];
            db += dstateT[j] * sa[j];
        }
        dw_[idx] = to_bf(W_SCALE * dw * wi * w_sig * (1.0f - w_sig));

        dk_[idx] = to_bf(dk);
        dv_[idx] = to_bf(dv);
        db_[idx] = to_bf(db);

        __syncthreads();
        dSb_shared[i] = dSb;
        __syncthreads();

        float da = 0.0f;
#pragma unroll
        for (int j = 0; j < N; ++j) {
            da += stateT[j] * dSb_shared[j];
        }
        da_[idx] = to_bf(da);

#pragma unroll
        for (int j = 0; j < N; ++j) {
            dstate[j] = dstate[j] * w[j] + dSb * a[j];
            dstateT[j] = dstateT[j] * wi + ai * dSb_shared[j];
        }
    }
}

void rwkv7_sequence_backward_cuda(
    int B,
    int T,
    int H,
    const void *r,
    const void *w,
    const void *k,
    const void *v,
    const void *a,
    const void *b,
    const void *dy,
    const float *s,
    const float *sa,
    void *dr,
    void *dw,
    void *dk,
    void *dv,
    void *da,
    void *db) {
    assert(T % _CHUNK_LEN_ == 0);
    sequence_backward_kernel<_N_><<<dim3(H, B), dim3(_N_)>>>(
        T,
        H,
        static_cast<CBFPtr>(r),
        static_cast<CBFPtr>(w),
        static_cast<CBFPtr>(k),
        static_cast<CBFPtr>(v),
        static_cast<CBFPtr>(a),
        static_cast<CBFPtr>(b),
        static_cast<CBFPtr>(dy),
        s,
        sa,
        static_cast<BFPtr>(dr),
        static_cast<BFPtr>(dw),
        static_cast<BFPtr>(dk),
        static_cast<BFPtr>(dv),
        static_cast<BFPtr>(da),
        static_cast<BFPtr>(db));
}
