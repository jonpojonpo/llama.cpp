#include "inkling.cuh"

// ---------------------------------------------------------------------------------------------------------------------
// CLIP: one warp per row, x = a*scale, out = x*clamp(t/(|x| + 1e-6), 0, 1)

static __global__ void inkling_clip_kernel(
        const float * __restrict__ a, const float * __restrict__ scale, float * __restrict__ dst,
        const int ne0, const int ne1, const int ne2, const int64_t nrows,
        const int sne0, const int sne1, const int sne2, const int sne3,
        const int64_t snb1, const int64_t snb2, const int64_t snb3, const float thr) {
    const int64_t row = int64_t(blockIdx.x)*blockDim.y + threadIdx.y;
    if (row >= nrows) {
        return;
    }
    const float * x = a   + row*ne0;
    float       * y = dst + row*ne0;

    const float * sc = nullptr;
    if (scale) {
        const int64_t i1 = row % ne1;
        const int64_t i2 = (row / ne1) % ne2;
        const int64_t i3 = row / (int64_t(ne1)*ne2);
        sc = scale + (i1 % sne1)*snb1 + (i2 % sne2)*snb2 + (i3 % sne3)*snb3;
    }

    float sum = 0.0f;
    for (int i = threadIdx.x; i < ne0; i += WARP_SIZE) {
        const float v = sc ? x[i]*sc[i % sne0] : x[i];
        sum += v*v;
    }
    sum = warp_reduce_sum(sum);

    const float factor = fminf(1.0f, fmaxf(0.0f, thr/(sqrtf(sum) + 1e-6f)));
    for (int i = threadIdx.x; i < ne0; i += WARP_SIZE) {
        const float v = sc ? x[i]*sc[i % sne0] : x[i];
        y[i] = v*factor;
    }
}

static void inkling_clip(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * a     = dst->src[0];
    const ggml_tensor * scale = dst->src[1];
    const float thr = ggml_get_op_params_f32(dst, 1);

    const int64_t nrows = ggml_nrows(a);
    const dim3 block(WARP_SIZE, 4, 1);
    const dim3 grid((nrows + 3)/4, 1, 1);

    inkling_clip_kernel<<<grid, block, 0, ctx.stream()>>>(
        (const float *) a->data, scale ? (const float *) scale->data : nullptr, (float *) dst->data,
        a->ne[0], a->ne[1], a->ne[2], nrows,
        scale ? scale->ne[0] : 1, scale ? scale->ne[1] : 1, scale ? scale->ne[2] : 1, scale ? scale->ne[3] : 1,
        scale ? scale->nb[1]/sizeof(float) : 0, scale ? scale->nb[2]/sizeof(float) : 0, scale ? scale->nb[3]/sizeof(float) : 0,
        thr);
}

// ---------------------------------------------------------------------------------------------------------------------
// MIX: one block per token; all (query, state) dot products and the state norms are reduced in one pass

#define INKLING_MIX_MAX_STATES  (GGML_MAX_SRC - 1)
#define INKLING_MIX_MAX_QUERIES 3
#define INKLING_MIX_THREADS     256

struct inkling_mix_states {
    const float * ptr[INKLING_MIX_MAX_STATES];
    int64_t       s1[INKLING_MIX_MAX_STATES]; // row stride in floats
};

template <int NS, int NQ>
static __global__ void inkling_mix_kernel(
        const float * __restrict__ query, const int64_t qs1, const inkling_mix_states states,
        float * __restrict__ dst, const int n_embd, const int64_t ds1, const int64_t ds2) {
    constexpr int NR = NS + NQ*NS; // the state sums of squares, then the dots
    const int t = blockIdx.x;

    float part[NR];
#pragma unroll
    for (int r = 0; r < NR; ++r) {
        part[r] = 0.0f;
    }

    for (int k = threadIdx.x; k < n_embd; k += INKLING_MIX_THREADS) {
        float q[NQ];
#pragma unroll
        for (int j = 0; j < NQ; ++j) {
            q[j] = query[j*qs1 + k];
        }
#pragma unroll
        for (int i = 0; i < NS; ++i) {
            const float s = states.ptr[i][t*states.s1[i] + k];
            part[i] += s*s;
#pragma unroll
            for (int j = 0; j < NQ; ++j) {
                part[NS + j*NS + i] += q[j]*s;
            }
        }
    }

    __shared__ float red[INKLING_MIX_THREADS/WARP_SIZE][NR];
    __shared__ float fin[NR];
    const int warp = threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
#pragma unroll
    for (int r = 0; r < NR; ++r) {
        const float v = warp_reduce_sum(part[r]);
        if (lane == 0) {
            red[warp][r] = v;
        }
    }
    __syncthreads();
    if (threadIdx.x < NR) {
        float v = 0.0f;
#pragma unroll
        for (int w = 0; w < INKLING_MIX_THREADS/WARP_SIZE; ++w) {
            v += red[w][threadIdx.x];
        }
        fin[threadIdx.x] = v;
    }
    __syncthreads();

    // weights p[j][i] = softmax_i(dot/(rms_i*sqrt(n_embd))) * inv_rms_i, applied to the raw states
    float inv[NS];
#pragma unroll
    for (int i = 0; i < NS; ++i) {
        inv[i] = rsqrtf(fin[i]/n_embd + 1e-6f);
    }
    const float qscale = rsqrtf((float) n_embd);
    float w[NQ][NS];
#pragma unroll
    for (int j = 0; j < NQ; ++j) {
        float mx = -INFINITY;
#pragma unroll
        for (int i = 0; i < NS; ++i) {
            w[j][i] = fin[NS + j*NS + i]*inv[i]*qscale;
            mx = fmaxf(mx, w[j][i]);
        }
        float sum = 0.0f;
#pragma unroll
        for (int i = 0; i < NS; ++i) {
            w[j][i] = expf(w[j][i] - mx);
            sum += w[j][i];
        }
#pragma unroll
        for (int i = 0; i < NS; ++i) {
            w[j][i] *= inv[i]/sum;
        }
    }

    for (int k = threadIdx.x; k < n_embd; k += INKLING_MIX_THREADS) {
        float acc[NQ];
#pragma unroll
        for (int j = 0; j < NQ; ++j) {
            acc[j] = 0.0f;
        }
#pragma unroll
        for (int i = 0; i < NS; ++i) {
            const float s = states.ptr[i][t*states.s1[i] + k];
#pragma unroll
            for (int j = 0; j < NQ; ++j) {
                acc[j] += w[j][i]*s;
            }
        }
#pragma unroll
        for (int j = 0; j < NQ; ++j) {
            dst[j*ds2 + t*ds1 + k] = acc[j];
        }
    }
}

template <int NS>
static void inkling_mix_launch_nq(int nq, const float * query, int64_t qs1, const inkling_mix_states & st,
        float * dst, int n_embd, int64_t ds1, int64_t ds2, int n_tokens, cudaStream_t stream) {
    const dim3 grid(n_tokens, 1, 1);
    const dim3 block(INKLING_MIX_THREADS, 1, 1);
    switch (nq) {
        case 1: inkling_mix_kernel<NS, 1><<<grid, block, 0, stream>>>(query, qs1, st, dst, n_embd, ds1, ds2); break;
        case 2: inkling_mix_kernel<NS, 2><<<grid, block, 0, stream>>>(query, qs1, st, dst, n_embd, ds1, ds2); break;
        case 3: inkling_mix_kernel<NS, 3><<<grid, block, 0, stream>>>(query, qs1, st, dst, n_embd, ds1, ds2); break;
        default: GGML_ABORT("inkling mix: unsupported number of queries");
    }
}

static void inkling_mix(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * query = dst->src[0];
    const int ns = ggml_get_op_params_i32(dst, 1);

    inkling_mix_states st = {};
    for (int i = 0; i < ns; ++i) {
        st.ptr[i] = (const float *) dst->src[1 + i]->data;
        st.s1[i]  = dst->src[1 + i]->nb[1]/sizeof(float);
    }

    const int     nq     = query->ne[1];
    const int     n_embd = query->ne[0];
    const int64_t qs1    = query->nb[1]/sizeof(float);
    const int64_t ds1    = dst->nb[1]/sizeof(float);
    const int64_t ds2    = dst->nb[2]/sizeof(float);
    const int     nt     = dst->ne[1];
    float * d = (float *) dst->data;
    const float * q = (const float *) query->data;
    cudaStream_t stream = ctx.stream();

    switch (ns) {
        case 1: inkling_mix_launch_nq<1>(nq, q, qs1, st, d, n_embd, ds1, ds2, nt, stream); break;
        case 2: inkling_mix_launch_nq<2>(nq, q, qs1, st, d, n_embd, ds1, ds2, nt, stream); break;
        case 3: inkling_mix_launch_nq<3>(nq, q, qs1, st, d, n_embd, ds1, ds2, nt, stream); break;
        case 4: inkling_mix_launch_nq<4>(nq, q, qs1, st, d, n_embd, ds1, ds2, nt, stream); break;
        case 5: inkling_mix_launch_nq<5>(nq, q, qs1, st, d, n_embd, ds1, ds2, nt, stream); break;
        case 6: inkling_mix_launch_nq<6>(nq, q, qs1, st, d, n_embd, ds1, ds2, nt, stream); break;
        case 7: inkling_mix_launch_nq<7>(nq, q, qs1, st, d, n_embd, ds1, ds2, nt, stream); break;
        case 8: inkling_mix_launch_nq<8>(nq, q, qs1, st, d, n_embd, ds1, ds2, nt, stream); break;
        case 9: inkling_mix_launch_nq<9>(nq, q, qs1, st, d, n_embd, ds1, ds2, nt, stream); break;
        default: GGML_ABORT("inkling mix: unsupported number of states");
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// CONTROL: (1) h[t, r] = D_r . [cur/rms(cur); msg/rms(msg)], one block per (r, t)
//          (2) out = (1 - tanh(F h))*cur + tanh(G h)*msg, one warp per output channel

#define INKLING_CTRL_THREADS 256

static __global__ void inkling_control_h_kernel(
        const float * __restrict__ cur, const float * __restrict__ msg, const float * __restrict__ down,
        float * __restrict__ h, const int n_embd, const int n_rank,
        const int64_t cs1, const int64_t ms1, const int64_t ds1) {
    const int r = blockIdx.x;
    const int t = blockIdx.y;
    const float * c = cur  + t*cs1;
    const float * m = msg  + t*ms1;
    const float * d = down + r*ds1;

    float v[4] = {0.0f, 0.0f, 0.0f, 0.0f}; // |c|^2, |m|^2, d.c, d.m
    for (int k = threadIdx.x; k < n_embd; k += INKLING_CTRL_THREADS) {
        const float ck = c[k];
        const float mk = m[k];
        v[0] += ck*ck;
        v[1] += mk*mk;
        v[2] += d[k]*ck;
        v[3] += d[n_embd + k]*mk;
    }

    __shared__ float red[INKLING_CTRL_THREADS/WARP_SIZE][4];
    const int warp = threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        v[i] = warp_reduce_sum(v[i]);
        if (lane == 0) {
            red[warp][i] = v[i];
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float s[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (int w = 0; w < INKLING_CTRL_THREADS/WARP_SIZE; ++w) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                s[i] += red[w][i];
            }
        }
        h[t*n_rank + r] = s[2]*rsqrtf(s[0]/n_embd + 1e-6f) + s[3]*rsqrtf(s[1]/n_embd + 1e-6f);
    }
}

static __global__ void inkling_control_out_kernel(
        const float * __restrict__ cur, const float * __restrict__ msg, const float * __restrict__ h,
        const float * __restrict__ gain, const float * __restrict__ forget, float * __restrict__ dst,
        const int n_embd, const int n_rank, const int64_t cs1, const int64_t ms1, const int64_t gs1, const int64_t fs1,
        const int64_t os1) {
    const int k = blockIdx.x*blockDim.y + threadIdx.y;
    const int t = blockIdx.y;
    if (k >= n_embd) {
        return;
    }
    const float * ht = h + t*n_rank;
    const float * g  = gain   + k*gs1;
    const float * f  = forget + k*fs1;

    float ag = 0.0f;
    float af = 0.0f;
    for (int r = threadIdx.x; r < n_rank; r += WARP_SIZE) {
        const float hr = ht[r];
        ag += g[r]*hr;
        af += f[r]*hr;
    }
    ag = warp_reduce_sum(ag);
    af = warp_reduce_sum(af);

    if (threadIdx.x == 0) {
        dst[t*os1 + k] = (1.0f - tanhf(af))*cur[t*cs1 + k] + tanhf(ag)*msg[t*ms1 + k];
    }
}

static void inkling_control(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * cur    = dst->src[0];
    const ggml_tensor * msg    = dst->src[1];
    const ggml_tensor * down   = dst->src[2];
    const ggml_tensor * gain   = dst->src[3];
    const ggml_tensor * forget = dst->src[4];

    const int n_embd   = cur->ne[0];
    const int n_rank   = down->ne[1];
    const int n_tokens = cur->ne[1];

    ggml_cuda_pool_alloc<float> h(ctx.pool(), int64_t(n_tokens)*n_rank);
    cudaStream_t stream = ctx.stream();

    inkling_control_h_kernel<<<dim3(n_rank, n_tokens, 1), dim3(INKLING_CTRL_THREADS, 1, 1), 0, stream>>>(
        (const float *) cur->data, (const float *) msg->data, (const float *) down->data, h.get(), n_embd, n_rank,
        cur->nb[1]/sizeof(float), msg->nb[1]/sizeof(float), down->nb[1]/sizeof(float));

    constexpr int wpb = 8;
    inkling_control_out_kernel<<<dim3((n_embd + wpb - 1)/wpb, n_tokens, 1), dim3(WARP_SIZE, wpb, 1), 0, stream>>>(
        (const float *) cur->data, (const float *) msg->data, h.get(), (const float *) gain->data, (const float *) forget->data,
        (float *) dst->data, n_embd, n_rank, cur->nb[1]/sizeof(float), msg->nb[1]/sizeof(float),
        gain->nb[1]/sizeof(float), forget->nb[1]/sizeof(float), dst->nb[1]/sizeof(float));
}

// ---------------------------------------------------------------------------------------------------------------------

bool ggml_cuda_inkling_supported(const ggml_tensor * dst) {
    switch ((ggml_inkling_op) ggml_get_op_params_i32(dst, 0)) {
        case GGML_INKLING_CLIP:
            return true;
        case GGML_INKLING_MIX:
            return dst->src[0]->ne[1] >= 1 && dst->src[0]->ne[1] <= INKLING_MIX_MAX_QUERIES;
        case GGML_INKLING_CONTROL:
            return true;
    }
    return false;
}

void ggml_cuda_op_inkling(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    switch ((ggml_inkling_op) ggml_get_op_params_i32(dst, 0)) {
        case GGML_INKLING_CLIP:    inkling_clip(ctx, dst);    break;
        case GGML_INKLING_MIX:     inkling_mix(ctx, dst);     break;
        case GGML_INKLING_CONTROL: inkling_control(ctx, dst); break;
        default: GGML_ABORT("unknown inkling op");
    }
}
