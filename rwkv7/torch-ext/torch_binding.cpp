// SPDX-FileCopyrightText: Copyright (c) Bo Peng (BlinkDL)
// SPDX-FileCopyrightText: Copyright (c) 2026 Hugging Face Inc. and chinoll
// SPDX-License-Identifier: Apache-2.0
//
// Ported from BlinkDL/RWKV-CUDA rwkv7_fast_fused:
// https://github.com/BlinkDL/RWKV-CUDA

#include <torch/all.h>
#include "ATen/ATen.h"
#include <torch/library.h>

#include "registration.h"

void rwkv7_state_passing_forward_cuda(
    int B,
    int T,
    int H,
    const float *s0,
    const void *r,
    const void *w,
    const void *k,
    const void *v,
    const void *a,
    const void *b,
    void *y,
    float *sT,
    float *s,
    float *sa);

void rwkv7_state_passing_backward_cuda(
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
    const float *dsT,
    const float *s,
    const float *sa,
    float *ds0,
    void *dr,
    void *dw,
    void *dk,
    void *dv,
    void *da,
    void *db);

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
    float *sa);

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
    void *db);

void sequence_forward(
    const torch::Tensor &r,
    const torch::Tensor &w,
    const torch::Tensor &k,
    const torch::Tensor &v,
    const torch::Tensor &a,
    const torch::Tensor &b,
    torch::Tensor &y,
    torch::Tensor &s,
    torch::Tensor &sa) {
    const int B = r.size(0);
    const int T = r.size(1);
    const int H = r.size(2);
    rwkv7_sequence_forward_cuda(
        B,
        T,
        H,
        r.data_ptr(),
        w.data_ptr(),
        k.data_ptr(),
        v.data_ptr(),
        a.data_ptr(),
        b.data_ptr(),
        y.data_ptr(),
        s.data_ptr<float>(),
        sa.data_ptr<float>());
}

void sequence_backward(
    const torch::Tensor &r,
    const torch::Tensor &w,
    const torch::Tensor &k,
    const torch::Tensor &v,
    const torch::Tensor &a,
    const torch::Tensor &b,
    const torch::Tensor &dy,
    const torch::Tensor &s,
    const torch::Tensor &sa,
    torch::Tensor &dr,
    torch::Tensor &dw,
    torch::Tensor &dk,
    torch::Tensor &dv,
    torch::Tensor &da,
    torch::Tensor &db) {
    const int B = r.size(0);
    const int T = r.size(1);
    const int H = r.size(2);
    rwkv7_sequence_backward_cuda(
        B,
        T,
        H,
        r.data_ptr(),
        w.data_ptr(),
        k.data_ptr(),
        v.data_ptr(),
        a.data_ptr(),
        b.data_ptr(),
        dy.data_ptr(),
        s.data_ptr<float>(),
        sa.data_ptr<float>(),
        dr.data_ptr(),
        dw.data_ptr(),
        dk.data_ptr(),
        dv.data_ptr(),
        da.data_ptr(),
        db.data_ptr());
}

void state_passing_forward(
    const torch::Tensor &s0,
    const torch::Tensor &r,
    const torch::Tensor &w,
    const torch::Tensor &k,
    const torch::Tensor &v,
    const torch::Tensor &a,
    const torch::Tensor &b,
    torch::Tensor &y,
    torch::Tensor &sT,
    torch::Tensor &s,
    torch::Tensor &sa) {
    const int B = r.size(0);
    const int T = r.size(1);
    const int H = r.size(2);
    rwkv7_state_passing_forward_cuda(
        B,
        T,
        H,
        s0.data_ptr<float>(),
        r.data_ptr(),
        w.data_ptr(),
        k.data_ptr(),
        v.data_ptr(),
        a.data_ptr(),
        b.data_ptr(),
        y.data_ptr(),
        sT.data_ptr<float>(),
        s.data_ptr<float>(),
        sa.data_ptr<float>());
}

void state_passing_backward(
    const torch::Tensor &r,
    const torch::Tensor &w,
    const torch::Tensor &k,
    const torch::Tensor &v,
    const torch::Tensor &a,
    const torch::Tensor &b,
    const torch::Tensor &dy,
    const torch::Tensor &dsT,
    const torch::Tensor &s,
    const torch::Tensor &sa,
    torch::Tensor &ds0,
    torch::Tensor &dr,
    torch::Tensor &dw,
    torch::Tensor &dk,
    torch::Tensor &dv,
    torch::Tensor &da,
    torch::Tensor &db) {
    const int B = r.size(0);
    const int T = r.size(1);
    const int H = r.size(2);
    rwkv7_state_passing_backward_cuda(
        B,
        T,
        H,
        r.data_ptr(),
        w.data_ptr(),
        k.data_ptr(),
        v.data_ptr(),
        a.data_ptr(),
        b.data_ptr(),
        dy.data_ptr(),
        dsT.data_ptr<float>(),
        s.data_ptr<float>(),
        sa.data_ptr<float>(),
        ds0.data_ptr<float>(),
        dr.data_ptr(),
        dw.data_ptr(),
        dk.data_ptr(),
        dv.data_ptr(),
        da.data_ptr(),
        db.data_ptr());
}

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
    ops.def(
        "sequence_forward(Tensor r, Tensor w, Tensor k, Tensor v, Tensor a, Tensor b, Tensor! y, Tensor! s, "
        "Tensor! sa) -> ()");
    ops.impl("sequence_forward", torch::kCUDA, &sequence_forward);

    ops.def(
        "sequence_backward(Tensor r, Tensor w, Tensor k, Tensor v, Tensor a, Tensor b, Tensor dy, Tensor s, "
        "Tensor sa, Tensor! dr, Tensor! dw, Tensor! dk, Tensor! dv, Tensor! da, Tensor! db) -> ()");
    ops.impl("sequence_backward", torch::kCUDA, &sequence_backward);

    ops.def(
        "state_passing_forward(Tensor s0, Tensor r, Tensor w, Tensor k, Tensor v, Tensor a, Tensor b, "
        "Tensor! y, Tensor! sT, Tensor! s, Tensor! sa) -> ()");
    ops.impl("state_passing_forward", torch::kCUDA, &state_passing_forward);

    ops.def(
        "state_passing_backward(Tensor r, Tensor w, Tensor k, Tensor v, Tensor a, Tensor b, Tensor dy, "
        "Tensor dsT, Tensor s, Tensor sa, Tensor! ds0, Tensor! dr, Tensor! dw, Tensor! dk, Tensor! dv, "
        "Tensor! da, Tensor! db) -> ()");
    ops.impl("state_passing_backward", torch::kCUDA, &state_passing_backward);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
