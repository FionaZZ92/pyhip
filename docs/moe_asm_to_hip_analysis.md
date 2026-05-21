# Analysis: Which ASM Instructions Can Be Written in HIP C++?

The `moe_gemm_batch1` kernel has **169 assembly instructions**. This document categorizes
each instruction type by whether it can be replaced with standard HIP C++ code that hipcc
will compile to equivalent assembly.

## Instruction Breakdown

| Instruction | Count | HIP C++ Equivalent? | Notes |
|-------------|-------|---------------------|-------|
| `v_mfma_f32_16x16x16_bf16` | 20 | ⚠️ Intrinsic | `__builtin_amdgcn_mfma_f32_16x16x16bf16_1k` |
| `buffer_load_dwordx4` | 12 | ❌ No | Buffer descriptors have no HIP API |
| `s_add_u32` | 11 | ✅ Yes | Integer addition |
| `s_mov_b32` | 10 | ✅ Yes | Assignment / constant load |
| `s_waitcnt` | 9 | ❌ No (implicit) | Compiler inserts automatically |
| `v_mul_f32` | 8 | ✅ Yes | `float * float` |
| `v_mov_b32` | 8 | ✅ Yes | Zero-init / assignment |
| `v_add_u32_e32` | 8 | ✅ Yes | Integer addition |
| `v_add_u32` | 8 | ⚠️ Partial | Add-with-rounding-bias (f32→bf16 trick) |
| `s_lshl_b32` | 8 | ✅ Yes | Left shift |
| `v_lshlrev_b32` | 7 | ✅ Yes | Left shift |
| `v_and_b32` | 7 | ✅ Yes | Bitwise AND |
| `v_lshrrev_b32` | 6 | ✅ Yes | Right shift |
| `s_mul_i32` | 5 | ✅ Yes | Integer multiply |
| `s_load_dwordx2` | 5 | ✅ Yes | Kernel args (auto from params) |
| `s_load_dword` | 5 | ✅ Yes | Kernel args + indirect loads |
| `v_or_b32` | 4 | ✅ Yes | Bitwise OR |
| `global_atomic_pk_add_bf16` | 4 | ⚠️ Intrinsic | `__builtin_amdgcn_global_atomic_fadd_v2bf16` |
| `v_mul_lo_u32` | 3 | ✅ Yes | Integer multiply |
| `s_addc_u32` | 3 | ✅ Yes | Add with carry (64-bit ptr arith) |
| `s_sub_i32` / `s_add_i32` | 4 | ✅ Yes | Integer arithmetic |
| `s_ashr_i32` | 2 | ✅ Yes | Arithmetic right shift |
| `s_nop` | 2 | ❌ No (implicit) | Compiler inserts for hazards |
| `s_branch` / `s_cbranch_*` | 5 | ✅ Yes | Loop/if control flow |
| `v_cmp_*` + exec mask ops | 3 | ✅ Yes | `if (cond)` → exec masking |
| `s_mov_b64 exec` | 1 | ✅ Yes | Exec restore (from if-block) |

## Summary by Category

### ✅ Replaceable with HIP C++ (113 instructions, ~67%)

These map directly to C++ constructs that hipcc compiles identically:

| Category | Instructions | Count | HIP C++ Equivalent |
|----------|-------------|-------|-------------------|
| **Kernel args** | `s_load_dword*` | 10 | Function parameters |
| **Thread ID** | `v_and_b32`, `v_lshrrev_b32` on v0 | 4 | `threadIdx.x & 63`, `threadIdx.x >> 4` |
| **Integer arithmetic** | `s_lshl_b32`, `s_mul_i32`, `s_add_u32`, `v_lshlrev_b32`, `v_mul_lo_u32`, `v_add_u32_e32` | 45 | Shift, multiply, add operations |
| **Accumulator init** | `v_mov_b32 vN, 0` | 8 | `float acc[8] = {0.0f}` |
| **Float multiply** | `v_mul_f32` | 8 | `acc[i] *= weight` |
| **Bit manipulation** | `v_lshrrev_b32`, `v_and_b32`, `v_or_b32` (epilog) | 14 | Bit ops for bf16 pack |
| **Control flow** | branches, compares, exec mask | 8 | `for`, `if` statements |
| **64-bit ptr arith** | `s_addc_u32` | 3 | Compiler handles carries |
| **Assignment** | `s_mov_b32` (non-descriptor) | 5 | Variable assignment |
| **Pointer offset** | `s_add_u32` on ptr lo | 8 | `ptr + offset` |

### ❌ NOT Replaceable (45 instructions, ~27%)

These have no HIP C++ equivalent or require inline asm:

| Category | Instructions | Count | Why |
|----------|-------------|-------|-----|
| **Buffer loads** | `buffer_load_dwordx4` | 12 | Buffer descriptors (V#) are ISA-only. HIP has no API to construct a buffer descriptor and issue structured loads. |
| **Buffer descriptor setup** | `s_mov_b32` (to s[20:23], s[24:27]) | 5 | Part of buffer descriptor construction |
| **Waitcnt** | `s_waitcnt` | 9 | Memory ordering; compiler inserts its own but may differ |
| **NOPs** | `s_nop` | 2 | ISA hazard avoidance |

### ⚠️ Intrinsic Available, But Compiler May Schedule Differently (24 instructions, ~14%)

| Category | Instructions | Count | Intrinsic | Risk |
|----------|-------------|-------|-----------|------|
| **MFMA** | `v_mfma_f32_16x16x16_bf16` | 20 | `__builtin_amdgcn_mfma_f32_16x16x16bf16_1k` | Compiler may reorder MFMAs vs loads, changing overlap |
| **Atomic bf16 add** | `global_atomic_pk_add_bf16` | 4 | `__builtin_amdgcn_global_atomic_fadd_v2bf16` | Works, but type casting is tricky |

---

## What a HIP C++ Version Would Look Like

### Section 1: Prolog — Address Computation (FULLY in HIP C++)

```cpp
__global__ void moe_gemm_batch1_hip(
    void* p_input, void* p_weight, void* p_output,
    void* p_topk_ids, float* p_topk_weight, float* p_w_scale,
    int M, int N, int K)
{
    // Thread/lane identification (replaces sid 9, 23, 24)
    int tid = threadIdx.x;
    int lane_id = tid & 63;
    int lane_mod_16 = lane_id & 0xf;
    int lane_div_16 = lane_id >> 4;

    // Expert ID lookup (replaces sid 19-22)
    int batch_idx = blockIdx.y;
    int expert_id = ((int*)p_topk_ids)[batch_idx];
    float topk_w = p_topk_weight[batch_idx];

    // Input pointer offset (replaces sid 31-34, 41-45)
    char* input_base = (char*)p_input + batch_idx * K * 2;  // bf16 = 2 bytes

    // Weight pointer offset (replaces sid 52-56)
    char* weight_base = (char*)p_weight + (int64_t)expert_id * N * K * 2;

    // Output pointer offset (replaces sid 37-40)
    __bf16* output_base = (__bf16*)p_output + blockIdx.x * 32;

    // Per-thread offsets for MFMA lane mapping (replaces sid 26-30, 35-36)
    int voffset_b0 = (blockIdx.x * 32 * K * 2) + (tid * 16);  // B tile 0
    int voffset_b1 = voffset_b0 + (32 * K * 2);                // B tile 1 (next N-block)
    int voffset_a  = (lane_mod_16 * K * 2) + (lane_div_16 * 16);  // A tile
```

### Section 2: Main Loop — MFMA + Loads (REQUIRES inline asm for buffer loads)

```cpp
    // Accumulator init (replaces sid 11-18) — THIS PART IS HIP C++
    float C[8] = {0.0f};  // 2 tiles × 4 elements

    // === CANNOT DO IN PURE HIP C++ ===
    // The main loop uses buffer_load_dwordx4 with buffer descriptors.
    // There is NO HIP API to:
    //   1. Construct a buffer descriptor (V#): base_ptr + size + flags
    //   2. Issue a buffer_load with per-lane voffset + scalar soffset
    //
    // Alternative: use flat global loads (worse performance!)
    //   __bf16* a_ptr = (__bf16*)(input_base + voffset_a + k * 64);
    //   float4 a_data = *(float4*)a_ptr;  // generates flat_load_dwordx4
    //
    // BUT flat loads are ~10% slower than buffer loads because:
    //   - No bounds checking (buffer desc has size field)
    //   - No cache hint control (sc0, nt flags on buffer_load)
    //   - Different TLB behavior

    // MFMA intrinsic IS available:
    // typedef __attribute__((ext_vector_type(4))) float float4_acc;
    // typedef __attribute__((ext_vector_type(4))) short bf16x4;
    // float4_acc acc = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(b_data, a_data, acc, 0, 0, 0);
```

### Section 3: Epilog — Scale + Pack + Store (MOSTLY HIP C++)

```cpp
    // Scale by topk_weight (replaces sid 129-144) — HIP C++
    for (int i = 0; i < 8; i++)
        C[i] *= topk_w;

    // Convert f32 → bf16 with round-to-nearest (replaces sid 145-156) — HIP C++
    // The asm uses: v_add_u32 v, v, 0x8000; v_lshrrev_b32 v, 16, v
    // which is "add rounding bias then take top 16 bits"
    // HIP equivalent:
    __bf16 result[8];
    for (int i = 0; i < 8; i++)
        result[i] = __float2bfloat16(C[i]);  // compiler does same rounding

    // Pack into bf16x2 pairs (replaces sid 145-156)
    uint32_t packed[4];
    for (int i = 0; i < 4; i++)
        packed[i] = ((uint32_t)__bfloat16_as_ushort(result[2*i+1]) << 16)
                  | __bfloat16_as_ushort(result[2*i]);

    // Atomic store (replaces sid 160-167) — INTRINSIC
    // __builtin_amdgcn_global_atomic_fadd_v2bf16(ptr, packed_val);
    // OR with newer ROCm:
    // atomicAdd((__hip_bfloat162*)output_ptr, val);
```

---

## The Blocking Issue: Buffer Loads

The **12 `buffer_load_dwordx4`** instructions (plus 5 descriptor setup `s_mov_b32`) are the
main obstacle preventing a pure HIP C++ rewrite. Here's why:

### What buffer_load does that flat_load cannot:

1. **Structured addressing**: `buffer_load_dwordx4 dst, voffset, descriptor, soffset offen`
   - `descriptor` (s[20:23]): 128-bit buffer descriptor containing base_addr(64b) + size(32b) + flags(32b)
   - `voffset`: per-lane offset (different for each thread)
   - `soffset`: scalar offset (shared, updated each loop iteration)
   - This gives efficient `base + per_lane + loop_offset` addressing

2. **Bounds checking**: The size field in the descriptor prevents OOB reads (returns 0 instead of faulting)

3. **Cache control flags**: `sc0` (system coherent), `nt` (non-temporal) — guides L2 policy

4. **Performance**: On gfx942, buffer loads can achieve slightly better throughput than flat loads for strided patterns

### Alternatives to buffer_load:

| Approach | Feasibility | Performance Impact |
|----------|-------------|-------------------|
| `*(float4*)(ptr + offset)` | ✅ Works | ~5-10% slower (flat_load_dwordx4) |
| `__builtin_amdgcn_raw_buffer_load_b128` | ⚠️ Fragile | Same perf, but API unstable across ROCm versions |
| Inline asm for just the buffer loads | ✅ Works | Same perf, minimal asm |

---

## Recommended Hybrid Approach

Write a **hybrid kernel** that uses HIP C++ for everything EXCEPT buffer loads:

```cpp
__global__ void moe_gemm_batch1_hybrid(
    void* p_input, void* p_weight, void* p_output,
    void* p_topk_ids, float* p_topk_weight, float* p_w_scale,
    int M, int N, int K)
{
    // === SECTION 1: Pure HIP C++ (prolog) ===
    int tid = threadIdx.x;
    int lane_id = tid & 63;
    int lane_mod_16 = lane_id & 0xf;
    int lane_div_16 = lane_id >> 4;
    int batch_idx = blockIdx.y;

    int expert_id = ((int*)p_topk_ids)[batch_idx];
    float topk_w = p_topk_weight[batch_idx];

    // Compute base pointers
    char* input_ptr = (char*)p_input + (int64_t)batch_idx * K * 2;
    char* weight_ptr = (char*)p_weight + (int64_t)expert_id * N * K * 2;

    // Per-thread offsets
    int voffset_a = (lane_mod_16 * K * 2) + (lane_div_16 * 16);
    int voffset_b0 = (blockIdx.x * 32 * K * 2) + (tid * 16);
    int voffset_b1 = voffset_b0 + (32 * K * 2);  // next 32-column block

    // Accumulator init
    typedef __attribute__((ext_vector_type(4))) float float4_acc;
    float4_acc C0 = {0, 0, 0, 0};
    float4_acc C1 = {0, 0, 0, 0};

    // === SECTION 2: Inline ASM (buffer loads + MFMA + waitcnt) ===
    // Build buffer descriptors
    uint32_t desc_a[4], desc_b[4];
    // ... (must be inline asm to construct V# descriptors)

    // Main K-loop with buffer loads, MFMA, and explicit scheduling
    asm volatile(
        // Buffer descriptor setup + main loop with prefetch
        // ... (the core 80 instructions that MUST be asm)
        : "+v"(C0), "+v"(C1)  // accumulator outputs
        : "v"(voffset_a), "v"(voffset_b0), "v"(voffset_b1),
          "s"(input_ptr), "s"(weight_ptr), "s"(K)
        : "memory"
    );

    // === SECTION 3: Pure HIP C++ (epilog) ===
    // Scale results
    C0 *= topk_w;
    C1 *= topk_w;

    // Convert to bf16 and pack
    // ... (standard HIP bf16 conversion)

    // Bounds check + atomic store
    if (lane_mod_16 < M) {
        // Compute output address
        __bf16* out = (__bf16*)p_output + lane_mod_16 * N + blockIdx.x * 32 + lane_div_16 * 8;
        // Atomic add packed bf16x2
        __builtin_amdgcn_global_atomic_fadd_v2bf16(out, packed0);
        __builtin_amdgcn_global_atomic_fadd_v2bf16(out + 2, packed1);
        __builtin_amdgcn_global_atomic_fadd_v2bf16(out + 32, packed2);
        __builtin_amdgcn_global_atomic_fadd_v2bf16(out + 34, packed3);
    }
}
```

### Instruction count by section:

| Section | ASM Instructions | Can be HIP C++ | Must be ASM |
|---------|-----------------|----------------|-------------|
| Prolog (address setup) | ~55 | **55 (100%)** | 0 |
| Main loop (loads+MFMA) | ~80 | 4 (loop control) | **76 (buffer loads + MFMA + waitcnt)** |
| Epilog (scale+store) | ~34 | **22 (65%)** | 12 (atomics + some packing) |
| **Total** | **169** | **81 (48%)** | **88 (52%)** |

---

## Conclusion

**~48% of the kernel CAN be written in HIP C++** (prolog address math + epilog scaling).
The remaining **~52% must remain as inline asm** because:

1. **Buffer loads (17 instructions)**: No HIP API for buffer descriptors. Could use flat loads
   at a 5-10% performance cost.
2. **MFMA (20 instructions)**: Intrinsic exists but compiler may schedule differently, losing
   the carefully overlapped load/compute pipeline.
3. **Waitcnt (9 instructions)**: Critical for the software-pipelined main loop. Compiler inserts
   its own waitcnts but cannot match hand-tuned overlap.
4. **Atomic bf16 (4 instructions)**: Intrinsic exists; could work but type safety is fragile.

### If you accept flat_load instead of buffer_load:

The kernel becomes **~70% HIP C++** — only the MFMA + waitcnt scheduling in the inner loop
needs inline asm (to preserve the load/compute overlap that gives peak performance).

### If you also accept compiler-scheduled MFMA:

The kernel becomes **~95% HIP C++** — fully expressible with intrinsics. But performance may
degrade by 10-20% due to suboptimal instruction scheduling in the critical inner loop.
