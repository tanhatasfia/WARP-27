
#include "host_utils.h"
#include "bitpack.cuh"
#include "common.cuh"
#include "adaptive_prefetch_feedback.h"
#include <cerrno>
#include <cstdlib>
#include <climits>
#include <vector>

namespace {
constexpr int THREADS = 128;
constexpr int WARPS = THREADS / 32;
constexpr int RPW = 8;                
constexpr int BN = WARPS * RPW;       
constexpr int TC = 64;                 
constexpr int XPITCH = TC + 1;         
static_assert(RPW % 2 == 0, "RPW must be even for paired epilogues");

struct Proj {
    const uint32_t* w; const void* s; const void* z; const void* b; void* out; int n;
};

struct Params {
    Proj p[3];
    int V, B, K, nchunks, wpr, groups, group_size;
    int split_k, ntiles, jobs, act;
    int tcs_load_warps;
    int* live_sm_ctas;  
    int sm_count;
    int resident_cap;
    float* partial;                   
    int* ws;                          
    const float* rope_cos;            
    const float* rope_sin;
    int head_dim;                   
    int perm_hd;                    
};


template <int MODE>
__device__ __forceinline__ void cb_map(const Params& P, int v, int& proj, int& r) {
    if constexpr (MODE == CB_MODE_GATEUP) { proj = v & 1; r = v >> 1; }
    else if constexpr (MODE == CB_MODE_QKV) {
        const int n0 = P.p[0].n, n1 = P.p[1].n;
        if (v < n0)           { proj = 0; r = v; }
        else if (v < n0 + n1) { proj = 1; r = v - n0; }
        else                  { proj = 2; r = v - n0 - n1; }
    } else { proj = 0; r = v; }
}


#define CB_SEL(P, proj, field) \
    ((proj) == 0 ? (P).p[0].field : ((proj) == 1 ? (P).p[1].field : (P).p[2].field))

template <typename T>
__device__ __forceinline__ float cb_bias(const void* b, int r) {
    return b ? cb_to_float(reinterpret_cast<const T*>(b)[r]) : 0.f;
}


template <typename T, int MODE>
__device__ __forceinline__ void cb_epilogue(const Params& P, int b, int v, float mine, float partner) {
    int proj, r;
    cb_map<MODE>(P, v, proj, r);
    if constexpr (MODE == CB_MODE_GEMV) {
        const float y = cb_apply_act(mine + cb_bias<T>(P.p[0].b, r), P.act);
        reinterpret_cast<T*>(P.p[0].out)[(size_t)b * P.p[0].n + r] = cb_from_float<T>(y);
    } else if constexpr (MODE == CB_MODE_GATEUP) {
        if (proj != 0) return;                          
        const float y = cb_apply_act(mine, CB_ACT_SILU) * partner;
        reinterpret_cast<T*>(P.p[0].out)[(size_t)b * P.p[0].n + r] = cb_from_float<T>(y);
    } else {
        const void* bias = CB_SEL(P, proj, b);
        void* out = CB_SEL(P, proj, out);
        const int n = CB_SEL(P, proj, n);
        float y = mine + cb_bias<T>(bias, r);
        const int hd = proj < 2 ? P.perm_hd : 0;
        const int orig = cb_unperm(r, hd);
        if (P.rope_cos != nullptr && proj < 2) {
            const float yp = partner + cb_bias<T>(bias, r ^ 1);
            const int d = orig % P.head_dim;
            const float c = P.rope_cos[(size_t)b * P.head_dim + d];
            const float s = P.rope_sin[(size_t)b * P.head_dim + d];
            y = cb_rope(y, yp, d, P.head_dim, c, s);
        }
        reinterpret_cast<T*>(out)[(size_t)b * n + orig] = cb_from_float<T>(y);
    }
}

// Zero or an unset variable leaves that control disabled.
int cb_tcs_limit(const char* name, int maximum) {
    const char* value = std::getenv(name);
    if (value == nullptr) return 0;

    char* end = nullptr;
    errno = 0;
    const long limit = std::strtol(value, &end, 10);
    TORCH_CHECK(end != value && *end == '\0' && errno == 0 &&
                limit >= 0 && limit <= maximum,
                name, " must be an integer from 0 to ", maximum);
    return static_cast<int>(limit);
}

template <typename T, int BITS, int MODE, bool THROTTLE_LOADS = false>
__global__ void __launch_bounds__(THREADS) cb_gemv_kernel(const T* __restrict__ x, const Params P) {
    __shared__ float xs[32 * XPITCH];
    __shared__ int s_job;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;

    while (true) {
        const int job = cb_claim_job(P.ws, &s_job);       
        if (job >= P.jobs) break;
        int split, tile, b;
        cb_decode_job(job, P.split_k, P.ntiles, split, tile, b);
        const int c0 = (int)(((long long)P.nchunks * split) / P.split_k);
        const int c1 = (int)(((long long)P.nchunks * (split + 1)) / P.split_k);
        const int v0 = tile * BN + warp * RPW;          

        float acc[RPW];
        #pragma unroll
        for (int r = 0; r < RPW; ++r) acc[r] = 0.f;

        const T* xb = x + (size_t)b * P.K;
        for (int cb = c0; cb < c1; cb += TC) {
            const int nc = min(TC, c1 - cb);
          
            for (int e = threadIdx.x; e < TC * 32; e += THREADS) {
                const int c = e >> 5, j = e & 31;
                const int k = (cb + c) * 32 + j;
                xs[j * XPITCH + c] = (c < nc && k < P.K) ? cb_to_float(xb[k]) : 0.f;
            }
            __syncthreads();

            // Run load/compute in disjoint warp groups only in the throttled variant.
            // Every thread reaches the barriers, including inactive warps and tail lanes.
            const int load_warps = THROTTLE_LOADS ? P.tcs_load_warps : WARPS;
            for (int first = 0; first < WARPS; first += load_warps) {
                if (!THROTTLE_LOADS || (warp >= first && warp < first + load_warps)) {
                    for (int cc = lane; cc < nc; cc += 32) {
                        float xv[32];
                        float xsum = 0.f;
                        #pragma unroll
                        for (int j = 0; j < 32; ++j) { xv[j] = xs[j * XPITCH + cc]; xsum += xv[j]; }
                        const int chunk = cb + cc;
                        const int g = (chunk * 32) / P.group_size;   

                        #pragma unroll
                        for (int r = 0; r < RPW; ++r) {
                            const int v = v0 + r;
                            if (v < P.V) {                         
                                int proj, lr;
                                cb_map<MODE>(P, v, proj, lr);
                                uint32_t w[BITS];
                                cb_load_chunk<BITS>(CB_SEL(P, proj, w) + (size_t)lr * P.wpr + (size_t)chunk * BITS, w);
                                const T* sp = reinterpret_cast<const T*>(CB_SEL(P, proj, s)) + (size_t)lr * P.groups;
                                const T* zp = reinterpret_cast<const T*>(CB_SEL(P, proj, z)) + (size_t)lr * P.groups;
                                float d = 0.f;
                                #pragma unroll
                                for (int j = 0; j < 32; ++j) d = fmaf(xv[j], cb_code_to_float(cb_code<BITS>(w, j)), d);
                                const float s = cb_to_float(sp[g]), z = cb_to_float(zp[g]);
                                acc[r] = fmaf(s, d - z * xsum, acc[r]);
                            }
                        }
                    }
                }
                if constexpr (THROTTLE_LOADS) __syncthreads();
            }
            if constexpr (!THROTTLE_LOADS) __syncthreads();
        }

     
        #pragma unroll
        for (int r = 0; r < RPW; ++r) {
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) acc[r] += __shfl_xor_sync(0xffffffffu, acc[r], off);
        }

        float mine = 0.f, partner = 0.f;
        #pragma unroll
        for (int r = 0; r < RPW; ++r) {
            if (r == lane) mine = acc[r];
            if ((r ^ 1) == lane) partner = acc[r];
        }
        const int v = v0 + lane;
        if (lane < RPW && v < P.V) {
            if (P.split_k == 1) cb_epilogue<T, MODE>(P, b, v, mine, partner);
            else P.partial[((size_t)split * P.B + b) * P.V + v] = mine;
        }
        __syncthreads();                              
    }
    cb_finish(P.ws);
}



constexpr int TC_COPY_SLOTS = 4;
constexpr int TC_COPY_CHUNKS = 8;

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
template <typename T>
__device__ __forceinline__ uint32_t cb_mma_pair(float x, float y) {
    if constexpr (std::is_same<T, __half>::value) {
        return uint32_t(__half_as_ushort(__float2half_rn(x))) |
               (uint32_t(__half_as_ushort(__float2half_rn(y))) << 16);
    } else {
        return uint32_t(__bfloat16_as_ushort(__float2bfloat16_rn(x))) |
               (uint32_t(__bfloat16_as_ushort(__float2bfloat16_rn(y))) << 16);
    }
}

struct CbMmaOperands {
    uint32_t a0, a1, b0, b1;
};

template <typename T, int BITS, int MODE>
__device__ __forceinline__ CbMmaOperands cb_decode_mma_fragment(
    const Params& P, const T* x, const uint32_t* packed, int v0, int lane,
    int chunk, int half) {
    const int col = lane / 4;
    const int k = (lane % 4) * 2;
    const int offset = half * 16;
    const int abs_k = chunk * 32 + offset + k;
    CbMmaOperands f{};
    if (col == 0) {
        f.a0 = cb_mma_pair<T>(abs_k < P.K ? cb_to_float(x[abs_k]) : 0.f,
                             abs_k + 1 < P.K ? cb_to_float(x[abs_k + 1]) : 0.f);
        f.a1 = cb_mma_pair<T>(abs_k + 8 < P.K ? cb_to_float(x[abs_k + 8]) : 0.f,
                             abs_k + 9 < P.K ? cb_to_float(x[abs_k + 9]) : 0.f);
    }
    const int v = v0 + col;
    if (v < P.V) {
        int proj, row;
        cb_map<MODE>(P, v, proj, row);
        const int g = chunk * 32 / P.group_size;
        const size_t meta = (size_t)row * P.groups + g;
        const float s = cb_to_float(reinterpret_cast<const T*>(CB_SEL(P, proj, s))[meta]);
        const float z = cb_to_float(reinterpret_cast<const T*>(CB_SEL(P, proj, z))[meta]);
        uint32_t w[BITS];
        #pragma unroll
        for (int i = 0; i < BITS; ++i) w[i] = packed[col * BITS + i];
        const int j = offset + k;
        f.b0 = cb_mma_pair<T>((cb_code_to_float(cb_code<BITS>(w, j)) - z) * s,
                             (cb_code_to_float(cb_code<BITS>(w, j + 1)) - z) * s);
        f.b1 = cb_mma_pair<T>((cb_code_to_float(cb_code<BITS>(w, j + 8)) - z) * s,
                             (cb_code_to_float(cb_code<BITS>(w, j + 9)) - z) * s);
    }
    return f;
}

template <typename T>
__device__ __forceinline__ void cb_decode_mma(const CbMmaOperands& f, float (&c)[4]) {
    // PTX A registers are ordered (row 0,k0), (row 8,k0),
    // (row 0,k8), (row 8,k8), relative to each lane's group.
    if constexpr (std::is_same<T, __half>::value) {
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                     "{%0,%1,%2,%3}, {%4,%5,%6,%5}, {%7,%8}, {%0,%1,%2,%3};"
                     : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                     : "r"(f.a0), "r"(0u), "r"(f.a1), "r"(f.b0), "r"(f.b1));
    } else {
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                     "{%0,%1,%2,%3}, {%4,%5,%6,%5}, {%7,%8}, {%0,%1,%2,%3};"
                     : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                     : "r"(f.a0), "r"(0u), "r"(f.a1), "r"(f.b0), "r"(f.b1));
    }
}

template <int BYTES>
__device__ __forceinline__ void cb_tc_copy(uint32_t* dst, const uint32_t* src) {
    unsigned address = static_cast<unsigned>(__cvta_generic_to_shared(dst));
    asm volatile("cp.async.ca.shared.global [%0], [%1], %2;"
                 :: "r"(address), "l"(src), "n"(BYTES) : "memory");
}

__device__ __forceinline__ void cb_tc_wait(int newer) {
    switch (newer) {
        case 3: asm volatile("cp.async.wait_group 3;" ::: "memory"); break;
        case 2: asm volatile("cp.async.wait_group 2;" ::: "memory"); break;
        case 1: asm volatile("cp.async.wait_group 1;" ::: "memory"); break;
        default: asm volatile("cp.async.wait_group 0;" ::: "memory"); break;
    }
}
#endif

template <typename T, int BITS, int MODE>
__global__ void __launch_bounds__(THREADS) cb_mma_gemv_kernel(const T* x, const Params P, int distance) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    static_assert(RPW == 8, "one MMA output fragment per warp");
    constexpr int SLOT = TC_COPY_CHUNKS * BN * BITS;
    __shared__ __align__(16) uint32_t packed[TC_COPY_SLOTS * SLOT];
    __shared__ int s_job;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x / 32;
    __shared__ int sm_index;
    __shared__ int live_distance;
    if (P.live_sm_ctas && threadIdx.x == 0) {
        unsigned smid;
        asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
        sm_index = static_cast<int>(smid);
        atomicAdd(&P.live_sm_ctas[sm_index], 1);
    }
    __syncthreads();

    while (true) {
        const int job = cb_claim_job(P.ws, &s_job);
        if (job >= P.jobs) break;
        int split, tile, batch;
        cb_decode_job(job, P.split_k, P.ntiles, split, tile, batch);
        const int c0 = (int)((long long)P.nchunks * split / P.split_k);
        const int c1 = (int)((long long)P.nchunks * (split + 1) / P.split_k);
        const int tiles = (c1 - c0 + TC_COPY_CHUNKS - 1) / TC_COPY_CHUNKS;
        const int base = tile * BN;
        const int v0 = base + warp * RPW;
        const T* xb = x + (size_t)batch * P.K;
        float acc[4] = {};
       
        int next_to_issue = 0;
        int next_to_consume = 0;
        for (int t = 0; t < tiles; ++t) {

            if (P.live_sm_ctas && threadIdx.x == 0) {
                const int active = atomicAdd(&P.live_sm_ctas[sm_index], 0);
                const int cap = max(1, P.resident_cap);
                // Continuous occupancy fraction mapped to a bounded lookahead.
                const float free_fraction = fmaxf(0.f, 1.f - float(active - 1) / float(cap));
                live_distance = max(1, min(3, __float2int_rn(1.f + 2.f * free_fraction)));
            }
            if (P.live_sm_ctas) __syncthreads();
            const int lookahead = P.live_sm_ctas ? live_distance : distance;
          
            const int target_exclusive = min(tiles, t + lookahead + 1);
            while (next_to_issue < target_exclusive &&
                   next_to_issue - next_to_consume < TC_COPY_SLOTS) {
                const int request_tile = next_to_issue;
                uint32_t* dst = packed + (request_tile % TC_COPY_SLOTS) * SLOT;
                for (int e = threadIdx.x; e < TC_COPY_CHUNKS * BN; e += THREADS) {
                    const int chunk = c0 + request_tile * TC_COPY_CHUNKS + e / BN;
                    const int v = base + e % BN;
                    if (chunk < c1 && v < P.V) {
                        int proj, row;
                        cb_map<MODE>(P, v, proj, row);
                        const uint32_t* src = CB_SEL(P, proj, w) +
                                              (size_t)row * P.wpr + (size_t)chunk * BITS;
                        if constexpr (BITS == 3) {
                            #pragma unroll
                            for (int i = 0; i < 3; ++i) cb_tc_copy<4>(dst + e * BITS + i, src + i);
                        } else {
                            cb_tc_copy<BITS * 4>(dst + e * BITS, src);
                        }
                    }
                }
                asm volatile("cp.async.commit_group;" ::: "memory");
                ++next_to_issue;
            }
   
            cb_tc_wait(next_to_issue - t - 1);
            __syncthreads();
            const int first = c0 + t * TC_COPY_CHUNKS;
            const int count = min(c1 - first, TC_COPY_CHUNKS);
            const uint32_t* src = packed + (t % TC_COPY_SLOTS) * SLOT + warp * RPW * BITS;
            CbMmaOperands current = cb_decode_mma_fragment<T, BITS, MODE>(P, xb, src, v0, lane, first, 0);
            for (int i = 0; i < count; ++i) {
                const CbMmaOperands second =
                    cb_decode_mma_fragment<T, BITS, MODE>(P, xb, src, v0, lane, first + i, 1);
                cb_decode_mma<T>(current, acc);
                if (i + 1 < count) {
                    src += BN * BITS;
                    current = cb_decode_mma_fragment<T, BITS, MODE>(P, xb, src, v0, lane, first + i + 1, 0);
                }
                cb_decode_mma<T>(second, acc);
            }
       
            __syncthreads();
            next_to_consume = t + 1;
        }
        
        cb_tc_wait(0);
        __syncthreads();

        if (lane < 4) {
            const int v = v0 + 2 * lane;
            for (int i = 0; i < 2; ++i) {
                if (v + i < P.V) {
                    if (P.split_k == 1) cb_epilogue<T, MODE>(P, batch, v + i, acc[i], acc[i ^ 1]);
                    else P.partial[((size_t)split * P.B + batch) * P.V + v + i] = acc[i];
                }
            }
        }
        __syncthreads();
    }
    if (P.live_sm_ctas) {
        __syncthreads();
        if (threadIdx.x == 0) atomicSub(&P.live_sm_ctas[sm_index], 1);
    }
    cb_finish(P.ws);
#else
    __trap();
#endif
}

template <typename T, int MODE>
__global__ void cb_reduce_kernel(const Params P) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = P.B * P.V;
    if (i >= total) return;
    const int b = i / P.V, v = i - b * P.V;
    float mine = 0.f, partner = 0.f;
    const bool paired = (MODE == CB_MODE_GATEUP) || (MODE == CB_MODE_QKV && P.rope_cos != nullptr);
    for (int s = 0; s < P.split_k; ++s) {
        const float* ps = P.partial + ((size_t)s * P.B + b) * P.V;
        mine += ps[v];
        if (paired && (v ^ 1) < P.V) partner += ps[v ^ 1];
    }
    cb_epilogue<T, MODE>(P, b, v, mine, partner);
}

template <typename T, int BITS, int MODE>
void cb_launch(const torch::Tensor& x, Params P, int64_t rho, const torch::Tensor& ws_in) {
    P.nchunks = (int)cb_chunks(P.K);
    P.split_k = std::max(1, std::min(P.split_k, P.nchunks));
    P.ntiles = (P.V + BN - 1) / BN;
    P.jobs = P.B * P.ntiles * P.split_k;
    int blocks = cb_blocks(P.jobs, rho, cb_sm_count_current());
    if (P.B == 1) {
        const int cta_limit = cb_tcs_limit("WARP_TCS_MAX_CTAS", INT_MAX);
        if (cta_limit > 0) blocks = std::min(blocks, cta_limit);
        P.tcs_load_warps = cb_tcs_limit("WARP_TCS_LOAD_WARPS", WARPS);
    }
    cudaStream_t stream = cb_stream();

    torch::Tensor ws = cb_workspace(ws_in, x);
    P.ws = ws.data_ptr<int>();
    torch::Tensor partial;
    if (P.split_k > 1) {
        partial = torch::empty({P.split_k, P.B, P.V}, x.options().dtype(torch::kFloat32));
        P.partial = partial.data_ptr<float>();
    } else {
        P.partial = nullptr;
    }
    const auto* input = reinterpret_cast<const T*>(x.data_ptr());
    torch::Tensor live_sm_ctas;
    P.live_sm_ctas = nullptr;
    P.sm_count = cb_sm_count_current();
    P.resident_cap = 1;
    bool launched_mma = false;
    if constexpr (!std::is_same<T, float>::value) {
        const int mode = (P.B == 1 && cb_cc_major_current() >= 8)
            ? cb_tcs_limit("WARP_MARLIN_PREFETCH", 5) : 0;
        if (mode > 0) {
            TORCH_CHECK(P.tcs_load_warps == 0 || P.tcs_load_warps == WARPS,
                        "Marlin-style decode uses all copy warps");
            const int longest_slice = (P.nchunks + P.split_k - 1) / P.split_k;
            const int slice_tiles = (longest_slice + TC_COPY_CHUNKS - 1) / TC_COPY_CHUNKS;
            const int max_distance = std::min(3, std::max(0, slice_tiles - 1));
            int distance = std::min(mode, max_distance);
            // Mode 4: per-kernel measured SM-active counter selects distance.
            // The profile is obtained with ncu, not NVML GPU busy or latency.
            if (mode == 4) {
                const cb_sm_prefetch::Key key{x.get_device(), BITS, MODE, P.K, P.V,
                                             P.split_k, blocks, slice_tiles};
                distance = cb_sm_prefetch::Selector::instance().select(key,max_distance);
            }
            if (mode == 5) {
             
                live_sm_ctas = torch::zeros({P.sm_count}, x.options().dtype(torch::kInt32));
                P.live_sm_ctas = live_sm_ctas.data_ptr<int>();
                int max_active = 1;
                C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                    &max_active, cb_mma_gemv_kernel<T, BITS, MODE>, THREADS, 0));
                P.resident_cap = max_active;
            }
            cb_mma_gemv_kernel<T, BITS, MODE><<<blocks, THREADS, 0, stream>>>(input, P, distance);
            launched_mma = true;
        }
    }
    if (!launched_mma && P.tcs_load_warps > 0 && P.tcs_load_warps < WARPS) {
        cb_gemv_kernel<T, BITS, MODE, true><<<blocks, THREADS, 0, stream>>>(input, P);
    } else if (!launched_mma) {
        cb_gemv_kernel<T, BITS, MODE><<<blocks, THREADS, 0, stream>>>(input, P);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    if (P.split_k > 1) {
        const int total = P.B * P.V;
        cb_reduce_kernel<T, MODE><<<(total + 255) / 256, 256, 0, stream>>>(P);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
}

template <int MODE>
void cb_dispatch(const torch::Tensor& x, Params P, int64_t bits, int64_t rho, const torch::Tensor& ws) {
    CB_DISPATCH_FLOAT_TYPES(x.scalar_type(), "warp_gemv", [&] {
        using T = typename CbNative<scalar_t>::type;
        if (bits == 2)      cb_launch<T, 2, MODE>(x, P, rho, ws);
        else if (bits == 3) cb_launch<T, 3, MODE>(x, P, rho, ws);
        else                cb_launch<T, 4, MODE>(x, P, rho, ws);
    });
}

Proj cb_proj(const torch::Tensor& w, const torch::Tensor& s, const torch::Tensor& z,
             const torch::Tensor& bias, bool has_bias, torch::Tensor& out) {
    Proj p{};
    p.w = reinterpret_cast<const uint32_t*>(w.data_ptr<int32_t>());
    p.s = s.data_ptr(); p.z = z.data_ptr();
    p.b = has_bias ? bias.data_ptr() : nullptr;
    p.out = out.data_ptr();
    p.n = (int)w.size(0);
    return p;
}

Params cb_base(const torch::Tensor& x, const torch::Tensor& w, const torch::Tensor& s,
               int64_t K, int64_t group_size, int64_t split_k) {
    Params P{};
    P.B = (int)x.size(0); P.K = (int)K;
    P.wpr = (int)w.size(1); P.groups = (int)s.size(1); P.group_size = (int)group_size;
    P.split_k = (int)split_k; P.act = 0;
    P.rope_cos = nullptr; P.rope_sin = nullptr; P.head_dim = 0; P.perm_hd = 0;
    return P;
}

template <int MODE>
int64_t cb_occupancy(int64_t bits) {
    int blocks = 0;
    const void* f = bits == 2 ? (const void*)cb_gemv_kernel<__half, 2, MODE>
                  : bits == 3 ? (const void*)cb_gemv_kernel<__half, 3, MODE>
                              : (const void*)cb_gemv_kernel<__half, 4, MODE>;
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, f, THREADS, 0));
    return blocks;
}
}


torch::Tensor packed_gemv_cuda(torch::Tensor x, torch::Tensor packed, torch::Tensor scales, torch::Tensor zeros,
                               torch::Tensor bias, int64_t bits, int64_t in_features, int64_t group_size,
                               int64_t split_k, int64_t rho, int64_t act, torch::Tensor workspace) {
    cb_check_x(x, in_features);
    cb_check_weight(x, packed, scales, zeros, bits, in_features, group_size, "packed_gemv");
    TORCH_CHECK(split_k >= 1, "split_k must be >= 1");
    TORCH_CHECK(act >= 0 && act <= 2, "act must be 0 (none), 1 (relu) or 2 (silu)");
    const c10::cuda::CUDAGuard guard(x.device());
    const bool hb = cb_has_bias(bias, x, packed.size(0));
    auto out = torch::empty({x.size(0), packed.size(0)}, x.options());
    if (x.size(0) == 0) return out;
    Params P = cb_base(x, packed, scales, in_features, group_size, split_k);
    P.p[0] = cb_proj(packed, scales, zeros, bias, hb, out);
    P.p[1] = P.p[0]; P.p[2] = P.p[0];
    P.V = P.p[0].n; P.act = (int)act;
    cb_dispatch<CB_MODE_GEMV>(x, P, bits, rho, workspace);
    return out;
}

std::vector<torch::Tensor> fused_qkv_cuda(torch::Tensor x,
    torch::Tensor wq, torch::Tensor sq, torch::Tensor zq, torch::Tensor bq,
    torch::Tensor wk, torch::Tensor sk, torch::Tensor zk, torch::Tensor bk,
    torch::Tensor wv, torch::Tensor sv, torch::Tensor zv, torch::Tensor bv,
    torch::Tensor rope_cos, torch::Tensor rope_sin, int64_t head_dim, int64_t perm_hd,
    int64_t bits, int64_t in_features, int64_t group_size, int64_t split_k, int64_t rho,
    torch::Tensor workspace) {

    cb_check_x(x, in_features);
    cb_check_weight(x, wq, sq, zq, bits, in_features, group_size, "fused_qkv(q)");
    cb_check_weight(x, wk, sk, zk, bits, in_features, group_size, "fused_qkv(k)");
    cb_check_weight(x, wv, sv, zv, bits, in_features, group_size, "fused_qkv(v)");
    TORCH_CHECK(split_k >= 1, "split_k must be >= 1");
    const c10::cuda::CUDAGuard guard(x.device());
    const int64_t B = x.size(0);

    if (perm_hd > 0) {
        TORCH_CHECK(perm_hd % 2 == 0, "perm head_dim must be even");
        TORCH_CHECK(wq.size(0) % perm_hd == 0 && wk.size(0) % perm_hd == 0, "Q/K widths must be multiples of head_dim");
    }
    const bool rope = rope_cos.defined() && rope_cos.numel() > 0;
    torch::Tensor rc, rs;
    if (rope) {
        TORCH_CHECK(perm_hd > 0 && perm_hd == head_dim,
                    "fused RoPE needs Q/K weights stored pair-interleaved with the same head_dim "
                    "(ops.permute_for_rope / install_bitmap(fuse_rope=True))");
        TORCH_CHECK(rope_cos.sizes() == rope_sin.sizes(), "cos/sin shape mismatch");
        TORCH_CHECK(rope_cos.numel() == B * head_dim, "cos/sin must be [B, head_dim] (one row per x row)");
        rc = rope_cos.to(x.device(), torch::kFloat32).contiguous();
        rs = rope_sin.to(x.device(), torch::kFloat32).contiguous();
    }

    torch::Tensor W[3] = {wq, wk, wv}, S[3] = {sq, sk, sv}, Z[3] = {zq, zk, zv}, Bi[3] = {bq, bk, bv};
    std::vector<torch::Tensor> outs;
    for (int i = 0; i < 3; ++i) outs.push_back(torch::empty({B, W[i].size(0)}, x.options()));
    if (B == 0) return outs;

    Params P = cb_base(x, wq, sq, in_features, group_size, split_k);
    for (int i = 0; i < 3; ++i) P.p[i] = cb_proj(W[i], S[i], Z[i], Bi[i], cb_has_bias(Bi[i], x, W[i].size(0)), outs[i]);
    P.V = P.p[0].n + P.p[1].n + P.p[2].n;
    P.perm_hd = (int)perm_hd;
    if (rope) { P.rope_cos = rc.data_ptr<float>(); P.rope_sin = rs.data_ptr<float>(); P.head_dim = (int)head_dim; }
    cb_dispatch<CB_MODE_QKV>(x, P, bits, rho, workspace);
    return outs;
}

torch::Tensor fused_gate_up_cuda(torch::Tensor x, torch::Tensor wg, torch::Tensor sg, torch::Tensor zg,
                                 torch::Tensor wu, torch::Tensor su, torch::Tensor zu,
                                 int64_t bits, int64_t in_features, int64_t group_size, int64_t split_k,
                                 int64_t rho, torch::Tensor workspace) {
    cb_check_x(x, in_features);
    cb_check_weight(x, wg, sg, zg, bits, in_features, group_size, "fused_gate_up(gate)");
    cb_check_weight(x, wu, su, zu, bits, in_features, group_size, "fused_gate_up(up)");
    TORCH_CHECK(wg.sizes() == wu.sizes(), "gate/up packed shapes must match");
    TORCH_CHECK(split_k >= 1, "split_k must be >= 1");
    const c10::cuda::CUDAGuard guard(x.device());
    auto out = torch::empty({x.size(0), wg.size(0)}, x.options());
    if (x.size(0) == 0) return out;
    torch::Tensor none;
    Params P = cb_base(x, wg, sg, in_features, group_size, split_k);
    P.p[0] = cb_proj(wg, sg, zg, none, false, out);
    P.p[1] = cb_proj(wu, su, zu, none, false, out);
    P.p[2] = P.p[0];
    P.V = 2 * P.p[0].n;
    cb_dispatch<CB_MODE_GATEUP>(x, P, bits, rho, workspace);
    return out;
}

int64_t warp_sm_count() { return cb_sm_count_current(); }
int64_t warp_tile_n() { return BN; }


int64_t warp_max_rho_mode(int64_t bits, int64_t mode) {
    if (mode == CB_MODE_QKV) return cb_occupancy<CB_MODE_QKV>(bits);
    if (mode == CB_MODE_GATEUP) return cb_occupancy<CB_MODE_GATEUP>(bits);
    return cb_occupancy<CB_MODE_GEMV>(bits);
}
