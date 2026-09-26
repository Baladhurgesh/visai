import mlx.core as mx

_SOURCE = """
    uint row = threadgroup_position_in_grid.x;
    uint tid = thread_position_in_threadgroup.x;
    uint tg = threads_per_threadgroup.x;
    uint D = x_shape[1];
    uint base = row * D;

    float acc = 0.0f;
    for (uint i = tid; i < D; i += tg) {
        T hv = static_cast<T>(static_cast<float>(x[base + i]) + static_cast<float>(residual[base + i]));
        h[base + i] = hv;
        float hf = static_cast<float>(hv);
        acc += hf * hf;
    }

    threadgroup float partial[32];
    float s = simd_sum(acc);
    if (thread_index_in_simdgroup == 0) {
        partial[simdgroup_index_in_threadgroup] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simdgroup_index_in_threadgroup == 0) {
        uint nsg = (tg + 31) / 32;
        float v = thread_index_in_simdgroup < nsg ? partial[thread_index_in_simdgroup] : 0.0f;
        v = simd_sum(v);
        if (thread_index_in_simdgroup == 0) {
            partial[0] = v;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv = metal::rsqrt(partial[0] / float(D) + eps[0]);

    for (uint i = tid; i < D; i += tg) {
        float hf = static_cast<float>(
            static_cast<T>(static_cast<float>(x[base + i]) + static_cast<float>(residual[base + i])));
        out[base + i] = static_cast<T>(hf * inv * static_cast<float>(w[i]));
    }
"""

_KERNEL = mx.fast.metal_kernel(
    name="visai_add_rmsnorm",
    input_names=["x", "residual", "w", "eps"],
    output_names=["out", "h"],
    source=_SOURCE,
)


def kernel(x, residual, w, eps: float = 1e-6):
    rows, d = x.shape
    tg = 256 if d >= 256 else 32
    out, h = _KERNEL(
        inputs=[x, residual, w, mx.array([eps], dtype=mx.float32)],
        template=[("T", x.dtype)],
        grid=(rows * tg, 1, 1),
        threadgroup=(tg, 1, 1),
        output_shapes=[x.shape, x.shape],
        output_dtypes=[x.dtype, x.dtype],
    )
    return out, h
