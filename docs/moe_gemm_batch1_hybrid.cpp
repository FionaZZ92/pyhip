// moe_gemm_batch1_hybrid.cpp
// Hybrid HIP kernel: C++ prolog/epilog + inline asm for buffer loads + MFMA loop
// Supports both with_silu (gemm1, split-K, block=256) and no-silu (gemm2, block=64)
//
// gemm1 (with_silu=1): Grid: [N/2/16, B], Block: [256, 1, 1], output: [B, N/2]
// gemm2 (with_silu=0): Grid: [N/32, B], Block: [64, 1, 1], output: [B, N] (atomic)

#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>
#include "hip/hip_runtime.h"

__global__ __attribute__((amdgpu_flat_work_group_size(64, 256)))
void moe_gemm_batch1_hybrid(
    void* __restrict__ p_input,
    void* __restrict__ p_weight,
    void* __restrict__ p_output,
    void* __restrict__ p_topk_ids,
    float* __restrict__ p_topk_weight,
    float* __restrict__ p_w_scale,
    int M, int N, int K, int with_silu)
{
    // =========================================================================
    // SECTION 1: HIP C++ Prolog — Address computation
    // =========================================================================

    const int tid = threadIdx.x;
    const int lane_id = tid & 63;
    const int lane_mod_16 = lane_id & 0xf;
    const int lane_div_16 = lane_id >> 4;
    const int warp_id = tid >> 6;
    const int batch_idx = blockIdx.y;

    // Load expert_id
    int expert_id = ((const int*)p_topk_ids)[batch_idx];

    // Compute input base pointer
    uint64_t input_base;
    if (with_silu) {
        // silu: input is [M, K], NOT offset by batch in buffer (uses full M rows)
        input_base = (uint64_t)p_input;
    } else {
        // no-silu: input offset by batch_idx (each batch is one row)
        input_base = (uint64_t)p_input + (uint64_t)batch_idx * K * 2;
    }

    // Compute weight base pointer (offset by expert)
    uint64_t weight_base = (uint64_t)p_weight + (int64_t)expert_id * N * K * 2;

    // Compute output base pointer
    uint64_t output_base;
    if (with_silu) {
        // output: [B, N/2], offset = batch_idx * (N/2) * 2 + blockIdx.x * 16 * 2
        output_base = (uint64_t)p_output + (uint64_t)batch_idx * (N / 2) * 2
                    + (uint64_t)blockIdx.x * 16 * 2;
    } else {
        // output: [B, N], offset = blockIdx.x * 32 * 2
        output_base = (uint64_t)p_output + (uint64_t)blockIdx.x * 32 * 2;
    }

    // Buffer sizes for descriptors
    uint32_t input_buf_size = (uint32_t)((uint64_t)M * K * 2);
    uint32_t weight_buf_size = (uint32_t)((uint64_t)K * N * 2);

    // Per-thread voffsets
    // voffset_a: (tid % 16) * stride_A + (tid / 16) * 16
    uint32_t voffset_a = (tid % 16) * (K * 2) + (tid / 16) * 16;

    // voffset_b depends on with_silu
    uint32_t voffset_b0, voffset_b1;
    if (with_silu) {
        // voffset_b[0] = 16 * blockIdx.x * K * 2 + threadIdx.x * 16
        voffset_b0 = (uint32_t)blockIdx.x * 16 * (K * 2) + tid * 16;
        // voffset_b[1] = voffset_b[0] + (N/2) * K * 2
        voffset_b1 = voffset_b0 + (N / 2) * K * 2;
    } else {
        // voffset_b[0] = 32 * blockIdx.x * K * 2 + tid * 16
        voffset_b0 = (uint32_t)blockIdx.x * 32 * (K * 2) + tid * 16;
        // voffset_b[1] = voffset_b[0] + K * 16 * 2
        voffset_b1 = voffset_b0 + K * (16 * 2);
    }

    // Loop parameters differ by mode
    // with_silu: num_split_k=4, kb_advance=4096, ka_advance=256, k_shift=7 (K/128)
    // no_silu:   num_split_k=1, kb_advance=1024, ka_advance=64,  k_shift=5 (K/32)
    uint32_t kb_advance = with_silu ? 0x1000 : 0x400;
    uint32_t ka_advance = with_silu ? 0x100  : 0x40;
    int k_shift = with_silu ? 7 : 5;  // K >> k_shift = loop iterations

    // =========================================================================
    // SECTION 2: Inline ASM — Buffer descriptor setup + Main GEMM loop
    // =========================================================================

    float C0, C1, C2, C3, C4, C5, C6, C7;

    asm volatile(
        // Build buffer descriptor for A
        "   s_mov_b32 s20, %[inp_lo]            \n"
        "   s_mov_b32 s21, %[inp_hi]            \n"
        "   s_mov_b32 s22, %[inp_sz]            \n"
        "   s_mov_b32 s23, 0x20000              \n"
        // Build buffer descriptor for B
        "   s_mov_b32 s24, %[wt_lo]             \n"
        "   s_mov_b32 s25, %[wt_hi]             \n"
        "   s_mov_b32 s26, %[wt_sz]             \n"
        "   s_mov_b32 s27, 0x20000              \n"

        // Initialize scalar offsets
        "   s_mov_b32 s6, 0                     \n"  // soffset_kb
        "   s_mov_b32 s7, 0                     \n"  // soffset_ka

        // Zero accumulators
        "   v_mov_b32 v4, 0                     \n"
        "   v_mov_b32 v5, 0                     \n"
        "   v_mov_b32 v6, 0                     \n"
        "   v_mov_b32 v7, 0                     \n"
        "   v_mov_b32 v8, 0                     \n"
        "   v_mov_b32 v9, 0                     \n"
        "   v_mov_b32 v10, 0                    \n"
        "   v_mov_b32 v11, 0                    \n"

        // Prolog: first loads (ping buffer)
        "   buffer_load_dwordx4 a[0:3], %[va], s[20:23], s7 offen          \n"
        "   buffer_load_dwordx4 v[16:19], %[vb0], s[24:27], s6 offen sc0 nt \n"
        "   buffer_load_dwordx4 v[20:23], %[vb1], s[24:27], s6 offen sc0 nt \n"
        "   s_add_u32 s6, %[kb_adv], s6         \n"
        "   s_add_u32 s7, %[ka_adv], s7         \n"
        "   s_waitcnt vmcnt(3)                  \n"

        // Loop counter
        "   s_mov_b32 s8, 0                     \n"

        // Main loop (unrolled x2 for ping-pong)
        "_hybrid_while_begin:                   \n"
        "   s_ashr_i32 s9, %[K], %[k_shift]    \n"
        "   s_sub_i32 s9, s9, 2                \n"
        "   s_cmp_lt_i32 s8, s9                \n"
        "   s_cbranch_scc0 _hybrid_while_end   \n"

        // --- Pong loads + ping MFMA ---
        "   buffer_load_dwordx4 a[4:7], %[va], s[20:23], s7 offen          \n"
        "   buffer_load_dwordx4 v[24:27], %[vb0], s[24:27], s6 offen sc0 nt \n"
        "   buffer_load_dwordx4 v[28:31], %[vb1], s[24:27], s6 offen sc0 nt \n"
        "   s_add_u32 s6, %[kb_adv], s6         \n"
        "   s_add_u32 s7, %[ka_adv], s7         \n"
        "   s_waitcnt vmcnt(3)                  \n"
        "   v_mfma_f32_16x16x16_bf16 v[4:7], v[16:17], a[0:1], v[4:7]     \n"
        "   v_mfma_f32_16x16x16_bf16 v[4:7], v[18:19], a[2:3], v[4:7]     \n"
        "   v_mfma_f32_16x16x16_bf16 v[8:11], v[20:21], a[0:1], v[8:11]   \n"
        "   v_mfma_f32_16x16x16_bf16 v[8:11], v[22:23], a[2:3], v[8:11]   \n"

        // --- Ping loads + pong MFMA ---
        "   buffer_load_dwordx4 a[0:3], %[va], s[20:23], s7 offen          \n"
        "   buffer_load_dwordx4 v[16:19], %[vb0], s[24:27], s6 offen sc0 nt \n"
        "   buffer_load_dwordx4 v[20:23], %[vb1], s[24:27], s6 offen sc0 nt \n"
        "   s_add_u32 s6, %[kb_adv], s6         \n"
        "   s_add_u32 s7, %[ka_adv], s7         \n"
        "   s_waitcnt vmcnt(3)                  \n"
        "   v_mfma_f32_16x16x16_bf16 v[4:7], v[24:25], a[4:5], v[4:7]     \n"
        "   v_mfma_f32_16x16x16_bf16 v[4:7], v[26:27], a[6:7], v[4:7]     \n"
        "   v_mfma_f32_16x16x16_bf16 v[8:11], v[28:29], a[4:5], v[8:11]   \n"
        "   v_mfma_f32_16x16x16_bf16 v[8:11], v[30:31], a[6:7], v[8:11]   \n"

        "   s_add_i32 s8, 2, s8                \n"
        "   s_branch _hybrid_while_begin       \n"

        "_hybrid_while_end:                    \n"
        // Handle tail
        "   s_ashr_i32 s9, %[K], %[k_shift]    \n"
        "   s_sub_i32 s9, s9, 2                \n"
        "   s_add_i32 s9, 1, s9                \n"
        "   s_cmp_eq_i32 s8, s9                \n"
        "   s_cbranch_scc1 _hybrid_odd_k       \n"

        // Even tail
        "   buffer_load_dwordx4 a[4:7], %[va], s[20:23], s7 offen          \n"
        "   buffer_load_dwordx4 v[24:27], %[vb0], s[24:27], s6 offen sc0 nt \n"
        "   buffer_load_dwordx4 v[28:31], %[vb1], s[24:27], s6 offen sc0 nt \n"
        "   s_waitcnt vmcnt(3)                  \n"
        "   v_mfma_f32_16x16x16_bf16 v[4:7], v[16:17], a[0:1], v[4:7]     \n"
        "   v_mfma_f32_16x16x16_bf16 v[4:7], v[18:19], a[2:3], v[4:7]     \n"
        "   v_mfma_f32_16x16x16_bf16 v[8:11], v[20:21], a[0:1], v[8:11]   \n"
        "   v_mfma_f32_16x16x16_bf16 v[8:11], v[22:23], a[2:3], v[8:11]   \n"
        "   s_waitcnt vmcnt(0)                  \n"
        "   v_mfma_f32_16x16x16_bf16 v[4:7], v[24:25], a[4:5], v[4:7]     \n"
        "   v_mfma_f32_16x16x16_bf16 v[4:7], v[26:27], a[6:7], v[4:7]     \n"
        "   v_mfma_f32_16x16x16_bf16 v[8:11], v[28:29], a[4:5], v[8:11]   \n"
        "   v_mfma_f32_16x16x16_bf16 v[8:11], v[30:31], a[6:7], v[8:11]   \n"
        "   s_branch _hybrid_k_end             \n"

        // Odd tail
        "_hybrid_odd_k:                        \n"
        "   s_waitcnt vmcnt(0)                  \n"
        "   v_mfma_f32_16x16x16_bf16 v[4:7], v[16:17], a[0:1], v[4:7]     \n"
        "   v_mfma_f32_16x16x16_bf16 v[4:7], v[18:19], a[2:3], v[4:7]     \n"
        "   v_mfma_f32_16x16x16_bf16 v[8:11], v[20:21], a[0:1], v[8:11]   \n"
        "   v_mfma_f32_16x16x16_bf16 v[8:11], v[22:23], a[2:3], v[8:11]   \n"

        "_hybrid_k_end:                        \n"
        // Move accumulators to output operands
        "   v_mov_b32 %[c0], v4                 \n"
        "   v_mov_b32 %[c1], v5                 \n"
        "   v_mov_b32 %[c2], v6                 \n"
        "   v_mov_b32 %[c3], v7                 \n"
        "   v_mov_b32 %[c4], v8                 \n"
        "   v_mov_b32 %[c5], v9                 \n"
        "   v_mov_b32 %[c6], v10                \n"
        "   v_mov_b32 %[c7], v11                \n"

        : // outputs
          [c0] "=v" (C0), [c1] "=v" (C1), [c2] "=v" (C2), [c3] "=v" (C3),
          [c4] "=v" (C4), [c5] "=v" (C5), [c6] "=v" (C6), [c7] "=v" (C7)
        : // inputs
          [va]  "v" (voffset_a),
          [vb0] "v" (voffset_b0),
          [vb1] "v" (voffset_b1),
          [inp_lo] "s" ((uint32_t)(input_base & 0xFFFFFFFF)),
          [inp_hi] "s" ((uint32_t)(input_base >> 32)),
          [inp_sz] "s" (input_buf_size),
          [wt_lo]  "s" ((uint32_t)(weight_base & 0xFFFFFFFF)),
          [wt_hi]  "s" ((uint32_t)(weight_base >> 32)),
          [wt_sz]  "s" (weight_buf_size),
          [K]      "s" (K),
          [kb_adv] "s" (kb_advance),
          [ka_adv] "s" (ka_advance),
          [k_shift] "s" (k_shift)
        : // clobbers
          "memory",
          "s6", "s7", "s8", "s9",
          "s20", "s21", "s22", "s23", "s24", "s25", "s26", "s27",
          "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11",
          "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23",
          "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31",
          "a0", "a1", "a2", "a3", "a4", "a5", "a6", "a7"
    );

    // =========================================================================
    // SECTION 3: HIP C++ Epilog — depends on with_silu
    // =========================================================================

    if (with_silu) {
        // =====================================================================
        // SILU PATH: LDS reduction across 4 waves + silu activation + store
        // =====================================================================
        __shared__ float lds_buf[4 * 16 * 32];  // 4 waves × 16 rows × 32 cols = 8KB

        // Each wave writes its partial C_reg[2][4] to LDS
        // LDS layout: [wave_id * 16 + lane_mod_16][lane_div_16 * 4 ... + 3] (+ second tile at +16)
        // Address: (lane_mod_16 + warp_id * 16) * 128 + lane_div_16 * 16
        uint32_t lds_write_addr = ((lane_mod_16 + warp_id * 16) * 32 + lane_div_16 * 4);

        lds_buf[lds_write_addr + 0]  = C0;
        lds_buf[lds_write_addr + 1]  = C1;
        lds_buf[lds_write_addr + 2]  = C2;
        lds_buf[lds_write_addr + 3]  = C3;
        lds_buf[lds_write_addr + 16] = C4;
        lds_buf[lds_write_addr + 17] = C5;
        lds_buf[lds_write_addr + 18] = C6;
        lds_buf[lds_write_addr + 19] = C7;

        __syncthreads();

        // Each wave reduces 4 rows: vrow = lane_div_16 + warp_id * 4
        int vrow = lane_div_16 + warp_id * 4;
        // Read from 4 waves' contributions for this row
        // LDS read: row vrow at columns [lane_mod_16] and [lane_mod_16 + 16]
        // Interleaved read from wave 0,1,2,3 (rows 0-15, 16-31, 32-47, 48-63)
        float gate0 = lds_buf[(vrow + 0 * 16) * 32 + lane_mod_16];
        float up0   = lds_buf[(vrow + 0 * 16) * 32 + lane_mod_16 + 16];
        float gate1 = lds_buf[(vrow + 1 * 16) * 32 + lane_mod_16];
        float up1   = lds_buf[(vrow + 1 * 16) * 32 + lane_mod_16 + 16];
        float gate2 = lds_buf[(vrow + 2 * 16) * 32 + lane_mod_16];
        float up2   = lds_buf[(vrow + 2 * 16) * 32 + lane_mod_16 + 16];
        float gate3 = lds_buf[(vrow + 3 * 16) * 32 + lane_mod_16];
        float up3   = lds_buf[(vrow + 3 * 16) * 32 + lane_mod_16 + 16];

        // Reduce: sum across 4 waves
        float gate_sum = gate0 + gate1 + gate2 + gate3;
        float up_sum = up0 + up1 + up2 + up3;

        // SiLU: gate_sum * sigmoid(gate_sum) * up_sum
        // sigmoid(x) = 1 / (1 + exp(-x))
        float neg_gate = -1.442695f * gate_sum;  // -log2(e) * x for v_exp
        float exp_val;
        asm volatile("v_exp_f32 %0, %1" : "=v"(exp_val) : "v"(neg_gate));
        float sigmoid = 1.0f / (1.0f + exp_val);
        float silu_out = gate_sum * sigmoid * up_sum;

        // Convert to bf16 and store (NOT atomic — single wave writes each element)
        uint32_t silu_bits = __builtin_bit_cast(uint32_t, silu_out);
        silu_bits += 0x8000u;  // round-to-nearest bias

        // Output address: vrow * (N/2) * 2 + lane_mod_16 * 2
        uint32_t out_addr = vrow * (N / 2) * 2 + lane_mod_16 * 2;

        if ((uint32_t)vrow < (uint32_t)M) {
            uint32_t out_lo = (uint32_t)(output_base & 0xFFFFFFFF);
            uint32_t out_hi = (uint32_t)(output_base >> 32);
            asm volatile(
                "   s_mov_b32 s10, %[out_lo]                \n"
                "   s_mov_b32 s11, %[out_hi]                \n"
                "   global_store_short_d16_hi %[addr], %[val], s[10:11]  \n"
                :
                : [addr] "v" (out_addr), [val] "v" (silu_bits),
                  [out_lo] "s" (out_lo), [out_hi] "s" (out_hi)
                : "memory", "s10", "s11"
            );
        }
    } else {
        // =====================================================================
        // NO-SILU PATH: Scale by topk_weight + atomic bf16 store
        // =====================================================================
        float topk_w = p_topk_weight[batch_idx];

        C0 *= topk_w; C1 *= topk_w; C2 *= topk_w; C3 *= topk_w;
        C4 *= topk_w; C5 *= topk_w; C6 *= topk_w; C7 *= topk_w;

        // Convert f32 → bf16 packed pairs
        uint32_t c0_bits = __builtin_bit_cast(uint32_t, C0) + 0x8000u;
        uint32_t c1_bits = __builtin_bit_cast(uint32_t, C1) + 0x8000u;
        uint32_t c2_bits = __builtin_bit_cast(uint32_t, C2) + 0x8000u;
        uint32_t c3_bits = __builtin_bit_cast(uint32_t, C3) + 0x8000u;
        uint32_t c4_bits = __builtin_bit_cast(uint32_t, C4) + 0x8000u;
        uint32_t c5_bits = __builtin_bit_cast(uint32_t, C5) + 0x8000u;
        uint32_t c6_bits = __builtin_bit_cast(uint32_t, C6) + 0x8000u;
        uint32_t c7_bits = __builtin_bit_cast(uint32_t, C7) + 0x8000u;

        uint32_t packed0 = (c0_bits >> 16) | (c1_bits & 0xFFFF0000u);
        uint32_t packed1 = (c2_bits >> 16) | (c3_bits & 0xFFFF0000u);
        uint32_t packed2 = (c4_bits >> 16) | (c5_bits & 0xFFFF0000u);
        uint32_t packed3 = (c6_bits >> 16) | (c7_bits & 0xFFFF0000u);

        // Output address: lane_mod_16 * N * 2 + lane_div_16 * 8
        uint32_t vaddr = lane_mod_16 * (N * 2) + lane_div_16 * 8;

        if ((uint32_t)lane_mod_16 < (uint32_t)M) {
            uint32_t vaddr1 = vaddr + 4;
            uint32_t vaddr2 = vaddr + 32;
            uint32_t vaddr3 = vaddr + 36;
            uint32_t out_lo = (uint32_t)(output_base & 0xFFFFFFFF);
            uint32_t out_hi = (uint32_t)(output_base >> 32);
            asm volatile(
                "   s_mov_b32 s10, %[out_lo]                \n"
                "   s_mov_b32 s11, %[out_hi]                \n"
                "   global_atomic_pk_add_bf16 %[a0], %[p0], s[10:11]  \n"
                "   global_atomic_pk_add_bf16 %[a1], %[p1], s[10:11]  \n"
                "   global_atomic_pk_add_bf16 %[a2], %[p2], s[10:11]  \n"
                "   global_atomic_pk_add_bf16 %[a3], %[p3], s[10:11]  \n"
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
