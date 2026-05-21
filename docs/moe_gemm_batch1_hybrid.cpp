// moe_gemm_batch1_hybrid.cpp
// Hybrid HIP kernel: C++ prolog/epilog + inline asm for buffer loads + MFMA loop
// Produces identical results to the pyhip JIT kernel (moe_gemm_batch1, bf16, no silu)
//
// Grid: [N/32, B], Block: [256, 1, 1]
// Input: [B, K] bf16, Weight: [E, N, K] bf16, Output: [B, N] bf16 (atomic accumulate)

#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>
#include "hip/hip_runtime.h"

__global__ __attribute__((amdgpu_flat_work_group_size(256, 256)))
void moe_gemm_batch1_hybrid(
    void* __restrict__ p_input,
    void* __restrict__ p_weight,
    void* __restrict__ p_output,
    void* __restrict__ p_topk_ids,
    float* __restrict__ p_topk_weight,
    float* __restrict__ p_w_scale,
    int M, int N, int K)
{
    // =========================================================================
    // SECTION 1: HIP C++ Prolog — Address computation
    // =========================================================================

    const int tid = threadIdx.x;
    const int lane_id = tid & 63;
    const int lane_mod_16 = lane_id & 0xf;
    const int lane_div_16 = lane_id >> 4;
    const int batch_idx = blockIdx.y;

    // Load expert_id and topk_weight (scalar loads)
    int expert_id = ((const int*)p_topk_ids)[batch_idx];
    float topk_w = p_topk_weight[batch_idx];

    // Compute input base pointer (offset by batch)
    // p_input_base = p_input + batch_idx * K * 2 (bf16 bytes)
    uint64_t input_base = (uint64_t)p_input + (uint64_t)batch_idx * K * 2;

    // Compute weight base pointer (offset by expert)
    // p_weight_base = p_weight + expert_id * N * K * 2
    uint64_t weight_base = (uint64_t)p_weight + (int64_t)expert_id * N * K * 2;

    // Compute output base pointer (offset by blockIdx.x * 32 bf16 elements)
    uint64_t output_base = (uint64_t)p_output + (uint64_t)blockIdx.x * 32 * 2;

    // Buffer sizes for descriptors
    uint32_t input_buf_size = (uint32_t)((uint64_t)M * K * 2);
    uint32_t weight_buf_size = (uint32_t)((uint64_t)K * N * 2);

    // Per-thread voffsets for buffer loads
    // voffset_a: each lane loads 8 bf16 = 16 bytes from A
    //   (tid % 16) selects M-row, (tid / 16) selects K-offset within tile
    //   NOTE: uses full threadIdx.x, NOT lane_id!
    uint32_t voffset_a = (tid % 16) * (K * 2) + (tid / 16) * 16;

    // voffset_b[0]: first 16-column block of B
    //   blockIdx.x * 32 columns, tid selects 16-byte chunk within K
    uint32_t voffset_b0 = blockIdx.x * 32 * (K * 2) + tid * 16;

    // voffset_b[1]: second 16-column block of B (K*16*2 bytes later)
    uint32_t voffset_b1 = voffset_b0 + K * (16 * 2);

    // =========================================================================
    // SECTION 2: Inline ASM — Buffer descriptor setup + Main GEMM loop
    // The inner loop MUST be asm for:
    //   - buffer_load_dwordx4 with V# descriptors (no HIP C++ API)
    //   - Explicit s_waitcnt for software-pipelined load/compute overlap
    //   - MFMA instruction scheduling (ping-pong registers)
    // =========================================================================

    // Accumulator outputs (8 x float32 = 2 tiles of 4 elements each)
    float C0, C1, C2, C3, C4, C5, C6, C7;

    // The asm block builds buffer descriptors and runs the main K-reduction loop.
    // Uses hardcoded v[4:11] for accumulators (contiguous ranges required by MFMA).
    // Inputs: base pointers, buffer sizes, voffsets, K
    // Outputs: C0-C7 via v_mov at end
    asm volatile(
        // Build buffer descriptor for A: {base_lo, base_hi, size, 0x20000}
        "   s_mov_b32 s20, %[inp_lo]            \n"
        "   s_mov_b32 s21, %[inp_hi]            \n"
        "   s_mov_b32 s22, %[inp_sz]            \n"
        "   s_mov_b32 s23, 0x20000              \n"
        // Build buffer descriptor for B: {base_lo, base_hi, size, 0x20000}
        "   s_mov_b32 s24, %[wt_lo]             \n"
        "   s_mov_b32 s25, %[wt_hi]             \n"
        "   s_mov_b32 s26, %[wt_sz]             \n"
        "   s_mov_b32 s27, 0x20000              \n"

        // Initialize scalar offsets
        "   s_mov_b32 s6, 0                     \n"  // soffset_kb = 0
        "   s_mov_b32 s7, 0                     \n"  // soffset_ka = 0

        // Zero accumulators in v[4:11]
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
        "   s_add_u32 s6, 0x400, s6             \n"  // soffset_kb += 1024
        "   s_add_u32 s7, 0x40, s7              \n"  // soffset_ka += 64
        "   s_waitcnt vmcnt(3)                  \n"

        // Loop counter
        "   s_mov_b32 s8, 0                     \n"  // cur_k = 0

        // Main loop (unrolled x2 for ping-pong)
        "_hybrid_while_begin:                   \n"
        "   s_ashr_i32 s9, %[K], 5             \n"  // K / 32
        "   s_sub_i32 s9, s9, 2                \n"  // K/32 - 2
        "   s_cmp_lt_i32 s8, s9                \n"
        "   s_cbranch_scc0 _hybrid_while_end   \n"

        // --- Iteration body (pong loads + ping MFMA) ---
        "   buffer_load_dwordx4 a[4:7], %[va], s[20:23], s7 offen          \n"
        "   buffer_load_dwordx4 v[24:27], %[vb0], s[24:27], s6 offen sc0 nt \n"
        "   buffer_load_dwordx4 v[28:31], %[vb1], s[24:27], s6 offen sc0 nt \n"
        "   s_add_u32 s6, 0x400, s6             \n"
        "   s_add_u32 s7, 0x40, s7              \n"
        "   s_waitcnt vmcnt(3)                  \n"
        // MFMA: ping A[0:3] x B[16:23]
        "   v_mfma_f32_16x16x16_bf16 v[4:7], v[16:17], a[0:1], v[4:7]     \n"
        "   v_mfma_f32_16x16x16_bf16 v[4:7], v[18:19], a[2:3], v[4:7]     \n"
        "   v_mfma_f32_16x16x16_bf16 v[8:11], v[20:21], a[0:1], v[8:11]   \n"
        "   v_mfma_f32_16x16x16_bf16 v[8:11], v[22:23], a[2:3], v[8:11]   \n"

        // --- Ping loads + pong MFMA ---
        "   buffer_load_dwordx4 a[0:3], %[va], s[20:23], s7 offen          \n"
        "   buffer_load_dwordx4 v[16:19], %[vb0], s[24:27], s6 offen sc0 nt \n"
        "   buffer_load_dwordx4 v[20:23], %[vb1], s[24:27], s6 offen sc0 nt \n"
        "   s_add_u32 s6, 0x400, s6             \n"
        "   s_add_u32 s7, 0x40, s7              \n"
        "   s_waitcnt vmcnt(3)                  \n"
        // MFMA: pong A[4:7] x B[24:31]
        "   v_mfma_f32_16x16x16_bf16 v[4:7], v[24:25], a[4:5], v[4:7]     \n"
        "   v_mfma_f32_16x16x16_bf16 v[4:7], v[26:27], a[6:7], v[4:7]     \n"
        "   v_mfma_f32_16x16x16_bf16 v[8:11], v[28:29], a[4:5], v[8:11]   \n"
        "   v_mfma_f32_16x16x16_bf16 v[8:11], v[30:31], a[6:7], v[8:11]   \n"

        "   s_add_i32 s8, 2, s8                \n"  // cur_k += 2
        "   s_branch _hybrid_while_begin       \n"

        "_hybrid_while_end:                    \n"
        // Handle tail: check if odd or even remaining
        "   s_ashr_i32 s9, %[K], 5             \n"
        "   s_sub_i32 s9, s9, 2                \n"
        "   s_add_i32 s9, 1, s9                \n"
        "   s_cmp_eq_i32 s8, s9                \n"
        "   s_cbranch_scc1 _hybrid_odd_k       \n"

        // Even tail: 2 blocks remaining (pong load + ping MFMA + drain)
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

        // Odd tail: 1 block remaining (just drain ping MFMA)
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
          [K]      "s" (K)
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
    // SECTION 3: HIP C++ Epilog — Scale, convert bf16, atomic store
    // =========================================================================

    // Scale accumulators by topk_weight
    C0 *= topk_w;
    C1 *= topk_w;
    C2 *= topk_w;
    C3 *= topk_w;
    C4 *= topk_w;
    C5 *= topk_w;
    C6 *= topk_w;
    C7 *= topk_w;

    // Convert f32 → bf16 using the same rounding trick as JIT:
    //   add 0x8000 (rounding bias) then take upper 16 bits
    // This is equivalent to __float2bfloat16_rn but ensures bit-exact match
    auto f32_to_bf16_bits = [](float val) -> uint32_t {
        uint32_t bits = __builtin_bit_cast(uint32_t, val);
        bits += 0x8000u;  // round-to-nearest-even bias
        return bits >> 16;
    };

    // Pack pairs of bf16 values into uint32 (low:high)
    // Layout matches JIT: creg_low[i] = (C[2i] >> 16) | (C[2i+1] & 0xFFFF0000)
    uint32_t c0_bits = __builtin_bit_cast(uint32_t, C0) + 0x8000u;
    uint32_t c1_bits = __builtin_bit_cast(uint32_t, C1) + 0x8000u;
    uint32_t c2_bits = __builtin_bit_cast(uint32_t, C2) + 0x8000u;
    uint32_t c3_bits = __builtin_bit_cast(uint32_t, C3) + 0x8000u;
    uint32_t c4_bits = __builtin_bit_cast(uint32_t, C4) + 0x8000u;
    uint32_t c5_bits = __builtin_bit_cast(uint32_t, C5) + 0x8000u;
    uint32_t c6_bits = __builtin_bit_cast(uint32_t, C6) + 0x8000u;
    uint32_t c7_bits = __builtin_bit_cast(uint32_t, C7) + 0x8000u;

    // Pack bf16 pairs: (even >> 16) | (odd & 0xFFFF0000)
    uint32_t packed0 = (c0_bits >> 16) | (c1_bits & 0xFFFF0000u);  // C[0], C[1]
    uint32_t packed1 = (c2_bits >> 16) | (c3_bits & 0xFFFF0000u);  // C[2], C[3]
    uint32_t packed2 = (c4_bits >> 16) | (c5_bits & 0xFFFF0000u);  // C[4], C[5]
    uint32_t packed3 = (c6_bits >> 16) | (c7_bits & 0xFFFF0000u);  // C[6], C[7]

    // Output address computation
    // vaddr = lane_mod_16 * (N * 2) + lane_div_16 * 8
    // (lane_mod_16 selects output row, lane_div_16 * 4 selects 4-element column group)
    uint32_t vaddr = lane_mod_16 * (N * 2) + lane_div_16 * 8;

    // Bounds check (exec mask equivalent)
    if ((uint32_t)lane_mod_16 < (uint32_t)M) {
        // global_atomic_pk_add_bf16: atomically add packed bf16x2 to output
        // Store 4 packed bf16x2 values at different output positions:
        //   [vaddr + 0]:  packed0 (row lane_mod_16, cols 0-1 of this tile)
        //   [vaddr + 4]:  packed1 (row lane_mod_16, cols 2-3 of this tile)
        //   [vaddr + 32]: packed2 (row lane_mod_16, cols 16-17 of this tile)
        //   [vaddr + 36]: packed3 (row lane_mod_16, cols 18-19 of this tile)
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
