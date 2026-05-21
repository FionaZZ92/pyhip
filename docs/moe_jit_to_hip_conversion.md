# MoE JIT Kernel → HIP Kernel Conversion Guide

This guide walks through converting the `moe_gemm_batch1` PyHIP JIT assembly kernel into a
standalone HIP kernel, verifying correctness and performance are preserved.

## Quick Start

```bash
cd /opt/pyhip
python3 docs/demo_moe_jit_to_hip.py
```

This script does everything end-to-end: generates assembly, exports files, loads the .co,
verifies correctness, and benchmarks both versions.

---

## How It Works

### The moe_gemm_batch1 Kernel

This kernel performs a batched GEMM for Mixture-of-Experts inference (single-token, batch=1):
- **Input**: `[B, K]` bf16 activations
- **Weight**: `[E, N, K]` bf16 expert weights (indexed by `topk_ids`)
- **Output**: `[B, N]` bf16 (accumulated via `global_atomic_pk_add_bf16`)
- **Tiling**: 16×32 MFMA tiles, 4 waves per workgroup, split-N across blockIdx.x

### Generated Assembly Structure

```
 Prolog: Load kernel args (s_load_dword* from kernarg segment)
  - p_input, p_weight, p_output, p_topk_ids, p_topk_weight, p_w_scale, M, N, K

 Setup: Compute buffer descriptors, offsets
  - lane_id = threadIdx.x & 63
  - lane_mod_16, lane_div_16 for MFMA lane mapping
  - voffset_a, voffset_b for buffer addressing
  - expert_id lookup from topk_ids

 Main Loop (_while_begin): K-dimension reduction
  - buffer_load_dwordx4 (A tiles from input, via buffer descriptor)
  - buffer_load_dwordx4 (B tiles from weight, via buffer descriptor)
  - v_mfma_f32_16x16x16_bf16 × 8 per iteration (ping-pong registers)
  - s_waitcnt for load/compute overlap

 Epilog: Scale results, convert f32→bf16, store
  - Multiply accumulator by topk_weight
  - Round-to-nearest bf16 (add 0x8000 bias + shift)
  - global_atomic_pk_add_bf16 to output

 s_endpgm
```

### Resource Usage (from compiler remarks)

| Resource | Value |
|----------|-------|
| VGPRs | 32 |
| AGPRs | 8 (for MFMA accumulators) |
| SGPRs | 34 (28 numbered + 6 system) |
| LDS | 0 bytes |
| Spills | 0 |
| Occupancy | 8 waves/SIMD |
| Scratch | 0 bytes/lane |

---

## Step-by-Step Conversion Process

### Step 1: Generate the Assembly

```python
import os
os.environ['PYHIP_RECOMPILE'] = '1'
os.environ['PYHIP_DUMP_DIR'] = './moe_dump'  # saves pass-by-pass IR

import torch
from pyhip.contrib.moe import moe_gemm_batch1

# Run kernel once to trigger full compilation
moe_gemm_batch1(
    [N // 32, B], [256, 1, 1],
    torch.bfloat16, False,  # compile-time args: weight_dtype, with_silu
    input_ptr, weight_ptr, output_ptr,
    topk_ids_ptr, topk_weight_ptr, w_scale_ptr,
    B, N, K
)
```

**Output files** (in `~/.pyhip/`):
- `.cpp` — HIP kernel source with inline assembly
- `.s` — Final device assembly after all passes (366 lines)
- `.co` — Compiled code object binary (7232 bytes)

**Intermediate IR dumps** (in `./moe_dump/`):
```
0-after_nothing-*.txt           # Raw IR from Python codegen
1-after_pass_remove_dead_bb     # Dead basic block removal
2-after_pass_a2v                # AccVGPR → VGPR conversion
3-after_pass_insert_nop         # ISA-required NOPs
4-after_pass_hide_dependency    # Instruction reordering for latency hiding
5-after_pass_hide_karg_loads    # Hide kernel arg load latency
6-after_pass_cse                # Common subexpression elimination
7-after_pass_dse                # Dead store elimination
8-after_pass_dce                # Dead code elimination
9-after_pass_break_down_gprs    # Split logical GPR arrays
10-after_reg_allocation         # Physical register assignment
```

### Step 2: Examine the Generated .cpp

The `.cpp` is already a valid standalone HIP kernel:

```cpp
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>
#include "hip/hip_runtime.h"
using as3_uint32_ptr = __attribute__((address_space(3))) uint32_t *;

__global__ void moe_gemm_batch1(void* p_input, void* p_weight, void* p_output,
    void* p_topk_ids, float* p_topk_weight, float* p_w_scale, int M, int N, int K)
{
    asm volatile(" ; blockIdx.x %0, blockIdx.y %1, blockIdx.z %2"
                 ::"s"(blockIdx.x),"s"(blockIdx.y),"s"(blockIdx.z));
    asm volatile("\n"
        "    s_load_dwordx2 s[6:7],s[0:1],0x0\n"   // load p_input
        "    s_load_dwordx2 s[8:9],s[0:1],0x8\n"   // load p_weight
        // ... ~180 lines of assembly ...
        "    global_atomic_pk_add_bf16 v1,v6,s[10:11]\n"
        "    s_endpgm\n"
        :: :"memory","s0","s1", ... ,"a7");
}
```

### Step 3: Compile to Standalone .co

```bash
# Option A: From .cpp → .s → .co (full pipeline)
hipcc -x hip --offload-device-only --offload-arch=gfx942 \
    -std=c++20 -O2 moe_gemm_batch1_bf16_nosilu.cpp \
    -S -o moe_gemm_batch1_bf16_nosilu.s

/opt/rocm/llvm/bin/clang++ -x assembler \
    -target amdgcn-amd-amdhsa -mcpu=gfx942 \
    moe_gemm_batch1_bf16_nosilu.s -o moe_gemm_batch1_bf16_nosilu.co

# Option B: Directly use the .co from ~/.pyhip/ (it's the same binary)
cp ~/.pyhip/moe_gemm_batch1-*.co ./moe_gemm_batch1_bf16_nosilu.co

# Option C: From .s only (skip hipcc, use pre-existing assembly)
/opt/rocm/llvm/bin/clang++ -x assembler \
    -target amdgcn-amd-amdhsa -mcpu=gfx942 \
    ~/.pyhip/moe_gemm_batch1-*.s -o moe_gemm_batch1_bf16_nosilu.co
```

### Step 4: Load and Launch the Exported Kernel

```python
from pyhip.core.hiptools import get_all_kernel_args, amdhip_func

# Inspect the .co to find kernel name and argument types
co_path = "moe_gemm_batch1_bf16_nosilu.co"
kernel_args = get_all_kernel_args(co_path)
# → {'moe_gemm_batch1': ('_Z15moe_gemm_batch1PvS_S_S_PfS0_iii',
#     ['void*', 'void*', 'void*', 'void*', 'float*', 'float*', 'int', 'int', 'int'])}

kname = 'moe_gemm_batch1'
sym_name, arg_types = kernel_args[kname]
hip_func = amdhip_func(co_path, sym_name, kname, arg_types)

# Launch (same grid/block as JIT version)
hip_func(
    [N // 32, B], [256, 1, 1],
    input_tensor.data_ptr(),
    weight_tensor.data_ptr(),
    output_tensor.data_ptr(),
    topk_ids.data_ptr(),
    topk_weight.data_ptr(),
    w_scale.data_ptr(),
    B, N, K
)
```

### Using raw HIP C++ (for integration into C++ applications)

```cpp
#include <hip/hip_runtime.h>

int main() {
    hipModule_t module;
    hipFunction_t kernel;

    hipModuleLoad(&module, "moe_gemm_batch1_bf16_nosilu.co");
    // Use llvm-objdump --dynamic-syms to find mangled name
    hipModuleGetFunction(&kernel, module, "_Z15moe_gemm_batch1PvS_S_S_PfS0_iii");

    struct {
        void* p_input; void* p_weight; void* p_output;
        void* p_topk_ids; float* p_topk_weight; float* p_w_scale;
        int M; int N; int K;
    } args = { d_input, d_weight, d_output, d_topk_ids,
               d_topk_weight, d_w_scale, M, N, K };

    size_t arg_size = sizeof(args);
    void* config[] = {
        HIP_LAUNCH_PARAM_BUFFER_POINTER, &args,
        HIP_LAUNCH_PARAM_BUFFER_SIZE, &arg_size,
        HIP_LAUNCH_PARAM_END
    };

    hipModuleLaunchKernel(kernel, N/32, B, 1, 256, 1, 1, 0, 0, nullptr, config);
    hipDeviceSynchronize();
}
```

---

## Correctness Verification

### Why results aren't bit-exact

This kernel uses `global_atomic_pk_add_bf16` to accumulate results from multiple wavefronts
into the same output location. Floating-point atomics are **non-deterministic** — the order
in which wavefronts reach the atomic depends on scheduling.

**Even JIT vs JIT (same binary, two runs) shows  8.0** in bf16 units.diff 

### Verification strategy

```python
# Run both on fresh (zeroed) outputs
output_jit.zero_()
output_hip.zero_()
run_jit()
run_hip()

# Also check JIT self-consistency
output_jit2.zero_()
run_jit_again()

jit_vs_hip = (output_jit - output_hip).abs().max()    # ~8.0
jit_vs_jit = (output_jit - output_jit2).abs().max()   # ~8.0  (same!)

# ✅ PASS if jit_vs_hip <= jit_vs_jit
```

---

## Performance Analysis

From the demo run on MI300X (gfx942):

| Version | Latency | Notes |
|---------|---------|-------|
| JIT kernel | ~59 µs | Includes Python dict lookup + output.zero_() |
| Exported HIP | ~54 µs | Pure hipModuleLaunchKernel |

The JIT version has minimal Python overhead (~5 µs) for cache lookup. The actual GPU kernel
execution is **identical** since both run the same `.co` binary.

For pure kernel-time comparison, use `rocprof`:
```bash
rocprof --stats python3 docs/demo_moe_jit_to_hip.py
# Both will show identical kernel duration in the profiler output
```

---

## Key Assembly Features

### MFMA (Matrix Fused Multiply-Add)
```asm
v_mfma_f32_16x16x16_bf16 v[4:7], v[16:17], a[0:1], v[4:7]
```
- 16×16×16 bf16 MFMA: multiplies 16×16 A (VGPRs) by 16×16 B (AGPRs)
- Accumulates into v[4:7] (4 VGPRs = 16×16 output in f32, 4 elements per lane)
- 8 MFMA instructions per loop iteration (2 N-blocks × 2 K-steps × 2 pingpong)

### Buffer Loads (Structured Addressing)
```asm
buffer_load_dwordx4 a[0:3], v12, s[20:23], s7 offen
```
- Loads 16 bytes (4 DWORDs) into AccVGPR registers
- Uses buffer descriptor `s[20:23]` with base + voffset + soffset addressing
- `offen` flag = offset comes from VGPR (per-lane addressing)

### Atomic Output
```asm
global_atomic_pk_add_bf16 v1, v3, s[10:11]
```
- Atomically adds packed bf16x2 value to global memory
- Enables multiple workgroups to accumulate into the same output

---

## Files Produced

```
docs/
 demo_moe_jit_to_hip.py              # Runnable demo script
 moe_jit_to_hip_conversion.md        # This guide
 moe_export/
   ├── moe_gemm_batch1_bf16_nosilu.cpp # Standalone HIP kernel source
   ├── moe_gemm_batch1_bf16_nosilu.s   # Device assembly (readable)
   └── moe_gemm_batch1_bf16_nosilu.co  # Binary code object (deployable)
 moe_dump/                           # (generated by PYHIP_DUMP_DIR)
 0-after_nothing-*.txt    ├
    ├── 1-after_pass_remove_dead_bb-*
    ├── ...
    └── 10-after_reg_allocation-*
```

---

## Why Not Write a Pure HIP C++ Kernel?

If you wanted to write the equivalent kernel purely in HIP C++ (without inline asm), you would
need to replicate the exact MFMA tiling, buffer addressing, and instruction scheduling. This is
impractical because:

1. **No C++ intrinsic for buffer_load with descriptor** — HIP doesn't expose buffer resource
   descriptors directly; you'd need inline asm for structured buffer loads anyway.

2. **MFMA requires intrinsics** — `__builtin_amdgcn_mfma_f32_16x16x16bf16` exists but gives
   the compiler freedom to schedule differently.

3. **Register allocation** — The compiler makes different choices, potentially increasing
   register pressure and reducing occupancy.

4. **Instruction scheduling** — pyhip's `pass_hide_dependency` manually reorders instructions
   for optimal latency hiding; the compiler may not achieve the same schedule.

**The correct approach is exactly what pyhip does**: write the logic in Python, let the JIT
generate optimized assembly, then export the `.co` for deployment.

---

## Important Caveats

1. **Compile-time args become constants**: The exported kernel has specific values baked in
   (e.g., `weight_dtype=bf16`, `with_silu=False`). Different compile-time args require a
   separate export.

2. **Architecture-specific**: The `.co` only works on the GPU arch it was compiled for.
   Re-assemble the `.s` for other targets:
   ```bash
   /opt/rocm/llvm/bin/clang++ -x assembler -target amdgcn-amd-amdhsa \
       -mcpu=gfx950 moe_kernel.s -o moe_kernel_gfx950.co
   ```

3. **Multiple variants**: A single `@pyhip.jit` kernel may produce multiple `.co` files
   (one per unique set of compile-time args). Export each variant separately.

4. **LDS size**: If the kernel uses LDS, ensure your launch specifies adequate shared memory
   (or rely on the static `__shared__` declaration in the `.cpp`).
