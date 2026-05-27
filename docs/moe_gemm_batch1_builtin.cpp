// moe_gemm_batch1_builtin.cpp
// HIP kernel using __builtin_amdgcn_raw_buffer_load_b128 for loads
// and inline asm only for MFMA + epilog stores.
// Supports both with_silu (gemm1, split-K, block=256) and no-silu (gemm2, block=64)
//
// gemm1 (with_silu=1): Grid: [N/2/16, B], Block: [256, 1, 1], output: [B, N/2]
// gemm2 (with_silu=0): Grid: [N/32, B], Block: [64, 1, 1], output: [B, N] (atomic)

#include <hip/hip_runtime.h>

typedef unsigned int uint4v __attribute__((ext_vector_type(4)));
typedef unsigned int uint2v __attribute__((ext_vector_type(2)));
typedef float float4v __attribute__((ext_vector_type(4)));

__global__ __attribute__((amdgpu_flat_work_group_size(64, 256)))
void moe_gemm_batch1_builtin(
    void* __restrict__ p_input,
    void* __restrict__ p_weight,
    void* __restrict__ p_output,
    void* __restrict__ p_topk_ids,
    float* __restrict__ p_topk_weight,
    float* __restrict__ p_w_scale,
    int M, int N, int K, int with_silu)
{
    const int tid = threadIdx.x;
    const int lane_id = tid & 63;
    const int lane_mod_16 = lane_id & 0xf;
    const int lane_div_16 = lane_id >> 4;
    const int warp_id = tid >> 6;
    const int batch_idx = blockIdx.y;

    int expert_id = ((const int*)p_topk_ids)[batch_idx];

    // =========================================================================
    // SECTION 1: Buffer descriptor setup — pure HIP C++ with builtins
    // =========================================================================

    char* a_ptr;
    if (with_silu) {
        a_ptr = (char*)p_input;  // silu: full input [M, K]
    } else {
        a_ptr = (char*)p_input + (uint64_t)batch_idx * K * 2;
    }
    __amdgpu_buffer_rsrc_t rsrc_a = __builtin_amdgcn_make_buffer_rsrc(
        a_ptr, 0, 0x7FFFFFFF, 0x00020000);

    // Weight B: [E, N, K] bf16
    char* b_ptr;
    int soffset_b1;
    if (with_silu) {
        // silu: voffset_b base = 16 * blockIdx.x * K * 2
        b_ptr = (char*)p_weight + (int64_t)expert_id * N * K * 2
              + (int64_t)blockIdx.x * 16 * K * 2;
        // second half: (N/2) * K * 2 bytes offset
        soffset_b1 = (N / 2) * K * 2;
    } else {
        // no-silu: voffset_b base = 32 * blockIdx.x * K * 2
        b_ptr = (char*)p_weight + (int64_t)expert_id * N * K * 2
              + (int64_t)blockIdx.x * 32 * K * 2;
        // second 16-row block: K * 16 * 2 bytes
        soffset_b1 = K * 16 * 2;
    }
    __amdgpu_buffer_rsrc_t rsrc_b = __builtin_amdgcn_make_buffer_rsrc(
        b_ptr, 0, 0x7FFFFFFF, 0x00020000);

    // Per-thread voffsets
    int voffset_a = (tid % 16) * (K * 2) + (tid / 16) * 16;
    int voffset_b = tid * 16;

    // K-step increments depend on mode
    // silu: split-K=4, kb_step=4096, ka_step=256, num_k_steps=K/128
    // no-silu: split-K=1, kb_step=1024, ka_step=64, num_k_steps=K/32
    int k_step_a = with_silu ? 256 : 64;
    int k_step_b = with_silu ? 4096 : 1024;
    int num_k_steps = with_silu ? (K / 128) : (K / 32);

    // =========================================================================
    // SECTION 2: Main GEMM loop — builtin loads + asm MFMA
    // =========================================================================

    float4v acc0 = {0.0f, 0.0f, 0.0f, 0.0f};
    float4v acc1 = {0.0f, 0.0f, 0.0f, 0.0f};

    int soff_a = 0, soff_b = 0;

    // Prolog: load first tile
    uint4v a0 = __builtin_amdgcn_raw_buffer_load_b128(rsrc_a, voffset_a, 0, 0);
    uint4v b00 = __builtin_amdgcn_raw_buffer_load_b128(rsrc_b, voffset_b, 0, 0);
    uint4v b01 = __builtin_amdgcn_raw_buffer_load_b128(rsrc_b, voffset_b, soffset_b1, 0);
    soff_a += k_step_a;
    soff_b += k_step_b;

    // Main loop (unrolled x2)
    int k = 1;
    for (; k + 1 < num_k_steps; k += 2) {
        // Pong loads
        uint4v a1 = __builtin_amdgcn_raw_buffer_load_b128(rsrc_a, voffset_a, soff_a, 0);
        uint4v b10 = __builtin_amdgcn_raw_buffer_load_b128(rsrc_b, voffset_b, soff_b, 0);
        uint4v b11 = __builtin_amdgcn_raw_buffer_load_b128(rsrc_b, voffset_b, soff_b + soffset_b1, 0);
        soff_a += k_step_a;
        soff_b += k_step_b;

        // MFMA on ping
        asm volatile("s_waitcnt vmcnt(3)" ::: "memory");
        {
            uint2v al = {a0.x, a0.y}, ah = {a0.z, a0.w};
            uint2v bl = {b00.x, b00.y}, bh = {b00.z, b00.w};
            uint2v cl = {b01.x, b01.y}, ch = {b01.z, b01.w};
            asm volatile(
                "v_mfma_f32_16x16x16_bf16 %0, %2, %4, %0\n"
                "v_mfma_f32_16x16x16_bf16 %0, %3, %5, %0\n"
                "v_mfma_f32_16x16x16_bf16 %1, %6, %4, %1\n"
                "v_mfma_f32_16x16x16_bf16 %1, %7, %5, %1\n"
                : "+v"(acc0), "+v"(acc1)
                : "v"(bl), "v"(bh), "v"(al), "v"(ah), "v"(cl), "v"(ch)
            );
        }

        // Ping loads
        a0 = __builtin_amdgcn_raw_buffer_load_b128(rsrc_a, voffset_a, soff_a, 0);
        b00 = __builtin_amdgcn_raw_buffer_load_b128(rsrc_b, voffset_b, soff_b, 0);
        b01 = __builtin_amdgcn_raw_buffer_load_b128(rsrc_b, voffset_b, soff_b + soffset_b1, 0);
        soff_a += k_step_a;
        soff_b += k_step_b;

        // MFMA on pong
        asm volatile("s_waitcnt vmcnt(3)" ::: "memory");
        {
            uint2v al = {a1.x, a1.y}, ah = {a1.z, a1.w};
            uint2v bl = {b10.x, b10.y}, bh = {b10.z, b10.w};
            uint2v cl = {b11.x, b11.y}, ch = {b11.z, b11.w};
            asm volatile(
                "v_mfma_f32_16x16x16_bf16 %0, %2, %4, %0\n"
                "v_mfma_f32_16x16x16_bf16 %0, %3, %5, %0\n"
                "v_mfma_f32_16x16x16_bf16 %1, %6, %4, %1\n"
                "v_mfma_f32_16x16x16_bf16 %1, %7, %5, %1\n"
                : "+v"(acc0), "+v"(acc1)
                : "v"(bl), "v"(bh), "v"(al), "v"(ah), "v"(cl), "v"(ch)
            );
        }
    }

    // Handle remaining iteration
    if (k < num_k_steps) {
        uint4v a1 = __builtin_amdgcn_raw_buffer_load_b128(rsrc_a, voffset_a, soff_a, 0);
        uint4v b10 = __builtin_amdgcn_raw_buffer_load_b128(rsrc_b, voffset_b, soff_b, 0);
        uint4v b11 = __builtin_amdgcn_raw_buffer_load_b128(rsrc_b, voffset_b, soff_b + soffset_b1, 0);

        asm volatile("s_waitcnt vmcnt(3)" ::: "memory");
        {
            uint2v al = {a0.x, a0.y}, ah = {a0.z, a0.w};
            uint2v bl = {b00.x, b00.y}, bh = {b00.z, b00.w};
            uint2v cl = {b01.x, b01.y}, ch = {b01.z, b01.w};
            asm volatile(
                "v_mfma_f32_16x16x16_bf16 %0, %2, %4, %0\n"
                "v_mfma_f32_16x16x16_bf16 %0, %3, %5, %0\n"
                "v_mfma_f32_16x16x16_bf16 %1, %6, %4, %1\n"
                "v_mfma_f32_16x16x16_bf16 %1, %7, %5, %1\n"
                : "+v"(acc0), "+v"(acc1)
                : "v"(bl), "v"(bh), "v"(al), "v"(ah), "v"(cl), "v"(ch)
            );
        }

        asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
        {
            uint2v al = {a1.x, a1.y}, ah = {a1.z, a1.w};
            uint2v bl = {b10.x, b10.y}, bh = {b10.z, b10.w};
            uint2v cl = {b11.x, b11.y}, ch = {b11.z, b11.w};
            asm volatile(
                "v_mfma_f32_16x16x16_bf16 %0, %2, %4, %0\n"
                "v_mfma_f32_16x16x16_bf16 %0, %3, %5, %0\n"
                "v_mfma_f32_16x16x16_bf16 %1, %6, %4, %1\n"
                "v_mfma_f32_16x16x16_bf16 %1, %7, %5, %1\n"
                : "+v"(acc0), "+v"(acc1)
                : "v"(bl), "v"(bh), "v"(al), "v"(ah), "v"(cl), "v"(ch)
            );
        }
    } else {
        asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
        {
            uint2v al = {a0.x, a0.y}, ah = {a0.z, a0.w};
            uint2v bl = {b00.x, b00.y}, bh = {b00.z, b00.w};
            uint2v cl = {b01.x, b01.y}, ch = {b01.z, b01.w};
            asm volatile(
                "v_mfma_f32_16x16x16_bf16 %0, %2, %4, %0\n"
                "v_mfma_f32_16x16x16_bf16 %0, %3, %5, %0\n"
                "v_mfma_f32_16x16x16_bf16 %1, %6, %4, %1\n"
                "v_mfma_f32_16x16x16_bf16 %1, %7, %5, %1\n"
                : "+v"(acc0), "+v"(acc1)
                : "v"(bl), "v"(bh), "v"(al), "v"(ah), "v"(cl), "v"(ch)
            );
        }
    }

    // Extract components (prevent CSE bug)
    float c0, c1, c2, c3, c4, c5, c6, c7;
    asm volatile(
        "v_mov_b32 %0, %8\n  v_mov_b32 %1, %9\n"
        "v_mov_b32 %2, %10\n v_mov_b32 %3, %11\n"
        "v_mov_b32 %4, %12\n v_mov_b32 %5, %13\n"
        "v_mov_b32 %6, %14\n v_mov_b32 %7, %15\n"
        : "=v"(c0), "=v"(c1), "=v"(c2), "=v"(c3),
          "=v"(c4), "=v"(c5), "=v"(c6), "=v"(c7)
        : "v"(acc0.x), "v"(acc0.y), "v"(acc0.z), "v"(acc0.w),
          "v"(acc1.x), "v"(acc1.y), "v"(acc1.z), "v"(acc1.w)
    );

    // =========================================================================
    // SECTION 3: Epilog — depends on with_silu
    // =========================================================================

    if (with_silu) {
        // =====================================================================
        // SILU PATH: LDS reduction across 4 waves + silu + store
        // =====================================================================
        __shared__ float lds_buf[4 * 16 * 32];  // 8KB

        uint32_t lds_write_addr = ((lane_mod_16 + warp_id * 16) * 32 + lane_div_16 * 4);
        lds_buf[lds_write_addr + 0]  = c0;
        lds_buf[lds_write_addr + 1]  = c1;
        lds_buf[lds_write_addr + 2]  = c2;
        lds_buf[lds_write_addr + 3]  = c3;
        lds_buf[lds_write_addr + 16] = c4;
        lds_buf[lds_write_addr + 17] = c5;
        lds_buf[lds_write_addr + 18] = c6;
        lds_buf[lds_write_addr + 19] = c7;

        __syncthreads();

        // Each wave reduces 4 rows
        int vrow = lane_div_16 + warp_id * 4;
        float gate0 = lds_buf[(vrow + 0 * 16) * 32 + lane_mod_16];
        float up0   = lds_buf[(vrow + 0 * 16) * 32 + lane_mod_16 + 16];
        float gate1 = lds_buf[(vrow + 1 * 16) * 32 + lane_mod_16];
        float up1   = lds_buf[(vrow + 1 * 16) * 32 + lane_mod_16 + 16];
        float gate2 = lds_buf[(vrow + 2 * 16) * 32 + lane_mod_16];
        float up2   = lds_buf[(vrow + 2 * 16) * 32 + lane_mod_16 + 16];
        float gate3 = lds_buf[(vrow + 3 * 16) * 32 + lane_mod_16];
        float up3   = lds_buf[(vrow + 3 * 16) * 32 + lane_mod_16 + 16];

        float gate_sum = gate0 + gate1 + gate2 + gate3;
        float up_sum = up0 + up1 + up2 + up3;

        // SiLU: gate * sigmoid(gate) * up
        float neg_gate = -1.442695f * gate_sum;
        float exp_val;
        asm volatile("v_exp_f32 %0, %1" : "=v"(exp_val) : "v"(neg_gate));
        float sigmoid = 1.0f / (1.0f + exp_val);
        float silu_out = gate_sum * sigmoid * up_sum;

        // Convert to bf16 and store
        uint32_t silu_bits = __builtin_bit_cast(uint32_t, silu_out) + 0x8000u;

        uint64_t output_base = (uint64_t)p_output + (uint64_t)batch_idx * (N / 2) * 2
                             + (uint64_t)blockIdx.x * 16 * 2;
        uint32_t out_addr = vrow * (N / 2) * 2 + lane_mod_16 * 2;

        if ((uint32_t)vrow < (uint32_t)M) {
            uint32_t out_lo = (uint32_t)(output_base & 0xFFFFFFFF);
            uint32_t out_hi = (uint32_t)(output_base >> 32);
            asm volatile(
                "s_mov_b32 s10, %[out_lo]\n"
                "s_mov_b32 s11, %[out_hi]\n"
                "global_store_short_d16_hi %[addr], %[val], s[10:11]\n"
                :
                : [addr] "v" (out_addr), [val] "v" (silu_bits),
                  [out_lo] "s" (out_lo), [out_hi] "s" (out_hi)
                : "memory", "s10", "s11"
            );
        }
    } else {
        // =====================================================================
        // NO-SILU PATH: Scale + atomic bf16 store
        // =====================================================================
        float topk_w = p_topk_weight[batch_idx];
        c0 *= topk_w; c1 *= topk_w; c2 *= topk_w; c3 *= topk_w;
        c4 *= topk_w; c5 *= topk_w; c6 *= topk_w; c7 *= topk_w;

        uint32_t c0_bits = __builtin_bit_cast(uint32_t, c0) + 0x8000u;
        uint32_t c1_bits = __builtin_bit_cast(uint32_t, c1) + 0x8000u;
        uint32_t c2_bits = __builtin_bit_cast(uint32_t, c2) + 0x8000u;
        uint32_t c3_bits = __builtin_bit_cast(uint32_t, c3) + 0x8000u;
        uint32_t c4_bits = __builtin_bit_cast(uint32_t, c4) + 0x8000u;
        uint32_t c5_bits = __builtin_bit_cast(uint32_t, c5) + 0x8000u;
        uint32_t c6_bits = __builtin_bit_cast(uint32_t, c6) + 0x8000u;
        uint32_t c7_bits = __builtin_bit_cast(uint32_t, c7) + 0x8000u;

        uint32_t packed0 = (c0_bits >> 16) | (c1_bits & 0xFFFF0000u);
        uint32_t packed1 = (c2_bits >> 16) | (c3_bits & 0xFFFF0000u);
        uint32_t packed2 = (c4_bits >> 16) | (c5_bits & 0xFFFF0000u);
        uint32_t packed3 = (c6_bits >> 16) | (c7_bits & 0xFFFF0000u);

        uint64_t output_base = (uint64_t)p_output + (uint64_t)blockIdx.x * 32 * 2;
        uint32_t vaddr = lane_mod_16 * (N * 2) + lane_div_16 * 8;

        if ((uint32_t)lane_mod_16 < (uint32_t)M) {
            uint32_t vaddr1 = vaddr + 4;
            uint32_t vaddr2 = vaddr + 32;
            uint32_t vaddr3 = vaddr + 36;
            uint32_t out_lo = (uint32_t)(output_base & 0xFFFFFFFF);
            uint32_t out_hi = (uint32_t)(output_base >> 32);
            asm volatile(
                "s_mov_b32 s10, %[out_lo]\n"
                "s_mov_b32 s11, %[out_hi]\n"
                "global_atomic_pk_add_bf16 %[a0], %[p0], s[10:11]\n"
                "global_atomic_pk_add_bf16 %[a1], %[p1], s[10:11]\n"
                "global_atomic_pk_add_bf16 %[a2], %[p2], s[10:11]\n"
                "global_atomic_pk_add_bf16 %[a3], %[p3], s[10:11]\n"
                :
                : [a0] "v" (vaddr),  [p0] "v" (packed0),
                  [a1] "v" (vaddr1), [p1] "v" (packed1),
                  [a2] "v" (vaddr2), [p2] "v" (packed2),
                  [a3] "v" (vaddr3), [p3] "v" (packed3),
                  [out_lo] "s" (out_lo), [out_hi] "s" (out_hi)
                : "memory", "s10", "s11"
            );
        }
    }
}
