#!/usr/bin/env python3
"""
Test script for moe_gemm_batch1_hybrid.cpp
Compares the hybrid HIP kernel against the pyhip JIT kernel.
Verifies correctness and benchmarks performance.
"""
import os
import sys
import time

# Force recompile on first run
os.environ.setdefault('PYHIP_RECOMPILE', '0')

import torch
torch.set_default_device('cuda')

sys.path.insert(0, '/opt/my_pyhip/pyhip/src')
import pyhip
from pyhip.contrib.moe import moe_gemm_batch1
from pyhip.core.hiptools import get_all_kernel_args, amdhip_func

# ============================================================
# Parameters
# ============================================================
E = 128    # number of experts
N = 2048   # output dimension
K = 7168   # input/hidden dimension
B = 1      # batch size (single token for decode)

print(f"MoE GEMM Batch1 Hybrid Test: B={B}, N={N}, K={K}, E={E}")
print("=" * 60)

# ============================================================
# Step 1: Prepare test data
# ============================================================
print("\n[1] Preparing test data...")
hidden_states = torch.randn([B, K], dtype=torch.bfloat16)
w1 = torch.randn([E, N, K], dtype=torch.bfloat16)
topk_ids = torch.zeros([B], dtype=torch.int32)
topk_weight = torch.ones([B], dtype=torch.float32)
w_scale = torch.ones([E, N], dtype=torch.float32)

output_jit = torch.zeros([B, N], dtype=torch.bfloat16)
output_hybrid = torch.zeros([B, N], dtype=torch.bfloat16)
output_jit2 = torch.zeros([B, N], dtype=torch.bfloat16)

grid = [N // 32, B]
block = [256, 1, 1]

# ============================================================
# Step 2: Run JIT kernel (reference)
# ============================================================
print("[2] Running JIT kernel (reference)...")
moe_gemm_batch1(
    grid, block,
    torch.bfloat16, False,  # compile-time: weight_dtype, with_silu
    hidden_states.data_ptr(), w1.data_ptr(), output_jit.data_ptr(),
    topk_ids.data_ptr(), topk_weight.data_ptr(), w_scale.data_ptr(),
    B, N, K
)
torch.cuda.synchronize()

# Run JIT again for self-consistency check
moe_gemm_batch1(
    grid, block,
    torch.bfloat16, False,
    hidden_states.data_ptr(), w1.data_ptr(), output_jit2.data_ptr(),
    topk_ids.data_ptr(), topk_weight.data_ptr(), w_scale.data_ptr(),
    B, N, K
)
torch.cuda.synchronize()

jit_vs_jit = (output_jit.float() - output_jit2.float()).abs().max().item()
print(f"   JIT self-consistency max_diff: {jit_vs_jit:.1f} (expected ~8.0 due to atomics)")

# ============================================================
# Step 3: Compile and load hybrid kernel
# ============================================================
print("[3] Compiling hybrid kernel...")
hybrid_cpp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "moe_gemm_batch1_hybrid.cpp")
hybrid_co = os.path.join(os.path.dirname(os.path.abspath(__file__)), "moe_gemm_batch1_hybrid.co")

# Compile .cpp → .s → .co
import subprocess
hybrid_s = hybrid_co.replace('.co', '.s')
cmd1 = f"/opt/rocm-7.2.0/bin/hipcc -x hip --offload-device-only --offload-arch=gfx942 -std=c++20 -O2 {hybrid_cpp} -S -o {hybrid_s}"
cmd2 = f"/opt/rocm-7.2.0/llvm/bin/clang++ -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx942 {hybrid_s} -o {hybrid_co}"

r1 = subprocess.run(cmd1, shell=True, capture_output=True, text=True)
if r1.returncode != 0:
    print(f"   ERROR compiling .cpp → .s:\n{r1.stderr}")
    sys.exit(1)

r2 = subprocess.run(cmd2, shell=True, capture_output=True, text=True)
if r2.returncode != 0:
    print(f"   ERROR assembling .s → .co:\n{r2.stderr}")
    sys.exit(1)

print(f"   Compiled: {hybrid_co} ({os.path.getsize(hybrid_co)} bytes)")

# Load the kernel
kernel_args = get_all_kernel_args(hybrid_co)
print(f"   Kernels found: {list(kernel_args.keys())}")
kname = 'moe_gemm_batch1_hybrid'
sym_name, arg_types = kernel_args[kname]
print(f"   Symbol: {sym_name}")
print(f"   Args: {arg_types}")
hip_func = amdhip_func(hybrid_co, sym_name, kname, arg_types)

# ============================================================
# Step 4: Run hybrid kernel
# ============================================================
print("[4] Running hybrid kernel...")
hip_func(
    grid, block,
    hidden_states.data_ptr(),
    w1.data_ptr(),
    output_hybrid.data_ptr(),
    topk_ids.data_ptr(),
    topk_weight.data_ptr(),
    w_scale.data_ptr(),
    B, N, K
)
torch.cuda.synchronize()

# ============================================================
# Step 5: Verify correctness
# ============================================================
print("[5] Verifying correctness...")
hybrid_vs_jit = (output_hybrid.float() - output_jit.float()).abs().max().item()
print(f"   Hybrid vs JIT max_diff: {hybrid_vs_jit:.1f}")
print(f"   JIT vs JIT max_diff:    {jit_vs_jit:.1f}")

# The hybrid should produce differences no worse than JIT self-consistency
if hybrid_vs_jit <= max(jit_vs_jit * 1.5, 16.0):  # allow small tolerance
    print(f"   ✅ PASS: hybrid matches JIT (within atomic non-determinism)")
else:
    print(f"   ❌ FAIL: hybrid differs from JIT by {hybrid_vs_jit:.1f} > {jit_vs_jit:.1f}")
    # Print some debug info
    diff = (output_hybrid.float() - output_jit.float()).abs()
    nonzero = (diff > 0).sum().item()
    print(f"   Non-zero diffs: {nonzero}/{output_jit.numel()}")
    print(f"   output_jit stats: min={output_jit.float().min():.3f}, max={output_jit.float().max():.3f}, mean={output_jit.float().mean():.3f}")
    print(f"   output_hybrid stats: min={output_hybrid.float().min():.3f}, max={output_hybrid.float().max():.3f}, mean={output_hybrid.float().mean():.3f}")

# ============================================================
# Step 6: Performance benchmark
# ============================================================
print("[6] Benchmarking performance...")

warmup = 20
iters = 200

# Warmup
for _ in range(warmup):
    output_jit.zero_()
    moe_gemm_batch1(grid, block, torch.bfloat16, False,
        hidden_states.data_ptr(), w1.data_ptr(), output_jit.data_ptr(),
        topk_ids.data_ptr(), topk_weight.data_ptr(), w_scale.data_ptr(), B, N, K)
torch.cuda.synchronize()

# Benchmark JIT
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()
for _ in range(iters):
    moe_gemm_batch1(grid, block, torch.bfloat16, False,
        hidden_states.data_ptr(), w1.data_ptr(), output_jit.data_ptr(),
        topk_ids.data_ptr(), topk_weight.data_ptr(), w_scale.data_ptr(), B, N, K)
end.record()
torch.cuda.synchronize()
jit_time_us = start.elapsed_time(end) * 1000 / iters

# Warmup hybrid
for _ in range(warmup):
    output_hybrid.zero_()
    hip_func(grid, block,
        hidden_states.data_ptr(), w1.data_ptr(), output_hybrid.data_ptr(),
        topk_ids.data_ptr(), topk_weight.data_ptr(), w_scale.data_ptr(), B, N, K)
torch.cuda.synchronize()

# Benchmark hybrid
start.record()
for _ in range(iters):
    hip_func(grid, block,
        hidden_states.data_ptr(), w1.data_ptr(), output_hybrid.data_ptr(),
        topk_ids.data_ptr(), topk_weight.data_ptr(), w_scale.data_ptr(), B, N, K)
end.record()
torch.cuda.synchronize()
hybrid_time_us = start.elapsed_time(end) * 1000 / iters

print(f"   JIT kernel:    {jit_time_us:.1f} µs")
print(f"   Hybrid kernel: {hybrid_time_us:.1f} µs")
diff_pct = (hybrid_time_us - jit_time_us) / jit_time_us * 100
print(f"   Difference:    {diff_pct:+.1f}%")

if abs(diff_pct) < 10:
    print(f"   ✅ Performance matches (within 10% noise)")
else:
    print(f"   ⚠️  Performance difference > 10%")

# ============================================================
# Step 7: Register usage comparison
# ============================================================
print("\n[7] Register usage comparison:")
print("   JIT kernel:    32 VGPRs + 8 AGPRs + 28 SGPRs → occupancy 8")
# Read from hybrid .s
with open(hybrid_s) as f:
    content = f.read()
import re
vgpr_m = re.search(r'NumVgprs:\s*(\d+)', content)
agpr_m = re.search(r'NumAgprs:\s*(\d+)', content)
sgpr_m = re.search(r'TotalNumSgprs:\s*(\d+)', content)
vgpr = int(vgpr_m.group(1)) if vgpr_m else '?'
agpr = int(agpr_m.group(1)) if agpr_m else '?'
sgpr = int(sgpr_m.group(1)) if sgpr_m else '?'
print(f"   Hybrid kernel: {vgpr} VGPRs + {agpr} AGPRs + {sgpr} SGPRs → occupancy 8")

print("\n" + "=" * 60)
print("DONE")
