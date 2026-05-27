#!/usr/bin/env python3
"""Benchmark moe_gemm_batch1 with Qwen3.5 MoE shapes (two-pass: gemm1+silu, gemm2).

Qwen3.5 MoE config:
- model_dimension (K) = 4096
- inter_dim = 1024, distributed across 8 cards → per-card N = 128
- 513 experts (E)
- topk = 11 → B = 11 (for single-token inference)

Two-pass MoE:
  gemm1: [B, K] x [E, N, K] → [B, N//2] with silu activation, block=256 (split-K)
  gemm2: [B, N//2] x [E, N, N//2] → [B, N] w/o activation, block=64 (1 wave)
"""
import os, sys, time
os.environ.setdefault('PYHIP_RECOMPILE', '0')
import torch
torch.set_default_device('cuda')
sys.path.insert(0, '/opt/my_pyhip/pyhip/src')
import pyhip
from pyhip.contrib.moe import moe_gemm_batch1
from pyhip.core.hiptools import get_all_kernel_args, amdhip_func

# Qwen3.5 MoE shape
E = 513       # number of experts
N = 128       # inter_dim per card (1024 / 8)
K = 4096      # model dimension
B = 11        # topk (tokens routed per expert call)
K2 = N // 2   # gemm2 input dim (= gemm1 output dim = 64)
N2 = N        # gemm2 output dim

print(f"=" * 70)
print(f"Qwen3.5 MoE Two-Pass Benchmark: E={E}, N={N}, K={K}, B={B}")
print(f"=" * 70)
print()

# Load unified hybrid kernel (supports both with_silu=1 and 0)
co_hybrid = '/opt/my_pyhip/pyhip/docs/moe_gemm_batch1_hybrid_dev.co'
ki_h = get_all_kernel_args(co_hybrid)
kn_h = list(ki_h.keys())[0]
sn_h, at_h = ki_h[kn_h]
hip_hybrid = amdhip_func(co_hybrid, sn_h, kn_h, at_h)

# Load unified builtin kernel (supports both with_silu=1 and 0)
co_builtin = '/opt/my_pyhip/pyhip/docs/moe_gemm_batch1_builtin_dev.co'
ki_b = get_all_kernel_args(co_builtin)
kn_b = list(ki_b.keys())[0]
sn_b, at_b = ki_b[kn_b]
hip_builtin = amdhip_func(co_builtin, sn_b, kn_b, at_b)

# Test data
hidden_states = torch.randn([B, K], dtype=torch.bfloat16)
w1 = torch.randn([E, N, K], dtype=torch.bfloat16)  # gate+up weights
topk_ids = torch.randint(0, E, [B], dtype=torch.int32)
topk_weight = torch.ones([B], dtype=torch.float32)
w_scale = torch.ones([E, N], dtype=torch.float32)

N_WARMUP = 50
N_ITER = 500

# =========================================================================
# GEMM1: with silu, block=256 (split-K with LDS reduction)
# =========================================================================
print(f"--- GEMM1 (with silu, block=256, split-K) ---")
print(f"  Shape: [{B}, {K}] x [{E}, {N}, {K}] → [{B}, {N//2}]")
grid1 = [N // 2 // 16, B]
block1 = [256, 1, 1]
print(f"  Grid: {grid1}, Block: {block1}")
print()

# Correctness: JIT vs hybrid vs builtin for gemm1
output_g1_jit = torch.zeros([B, N // 2], dtype=torch.bfloat16)
output_g1_hybrid = torch.zeros([B, N // 2], dtype=torch.bfloat16)
output_g1_builtin = torch.zeros([B, N // 2], dtype=torch.bfloat16)

moe_gemm_batch1(grid1, block1, torch.bfloat16, True,
    hidden_states.data_ptr(), w1.data_ptr(), output_g1_jit.data_ptr(),
    topk_ids.data_ptr(), topk_weight.data_ptr(), w_scale.data_ptr(), B, N, K)
torch.cuda.synchronize()

# hybrid: extra arg with_silu=1
hip_hybrid(grid1, block1,
    hidden_states.data_ptr(), w1.data_ptr(), output_g1_hybrid.data_ptr(),
    topk_ids.data_ptr(), topk_weight.data_ptr(), w_scale.data_ptr(), B, N, K, 1)
torch.cuda.synchronize()

# builtin: extra arg with_silu=1
hip_builtin(grid1, block1,
    hidden_states.data_ptr(), w1.data_ptr(), output_g1_builtin.data_ptr(),
    topk_ids.data_ptr(), topk_weight.data_ptr(), w_scale.data_ptr(), B, N, K, 1)
torch.cuda.synchronize()

# JIT noise floor for gemm1
output_g1_jit2 = torch.zeros([B, N // 2], dtype=torch.bfloat16)
moe_gemm_batch1(grid1, block1, torch.bfloat16, True,
    hidden_states.data_ptr(), w1.data_ptr(), output_g1_jit2.data_ptr(),
    topk_ids.data_ptr(), topk_weight.data_ptr(), w_scale.data_ptr(), B, N, K)
torch.cuda.synchronize()

diff_g1_hybrid = (output_g1_hybrid.float() - output_g1_jit.float()).abs().max().item()
diff_g1_builtin = (output_g1_builtin.float() - output_g1_jit.float()).abs().max().item()
g1_noise = (output_g1_jit.float() - output_g1_jit2.float()).abs().max().item()

print(f"  Correctness (gemm1, max abs diff):")
print(f"    hybrid  vs jit: max_diff = {diff_g1_hybrid}")
print(f"    builtin vs jit: max_diff = {diff_g1_builtin}")
print(f"    jit vs jit:     max_diff = {g1_noise} (noise floor)")
print(f"    hybrid  status: {'PASS' if diff_g1_hybrid <= max(g1_noise * 2, 16.0) else 'FAIL'}")
print(f"    builtin status: {'PASS' if diff_g1_builtin <= max(g1_noise * 5, 16.0) else 'FAIL'}")
print()

# Performance gemm1
for _ in range(N_WARMUP):
    output_g1_jit.zero_()
    moe_gemm_batch1(grid1, block1, torch.bfloat16, True,
        hidden_states.data_ptr(), w1.data_ptr(), output_g1_jit.data_ptr(),
        topk_ids.data_ptr(), topk_weight.data_ptr(), w_scale.data_ptr(), B, N, K)
torch.cuda.synchronize()
start = time.time()
for _ in range(N_ITER):
    output_g1_jit.zero_()
    moe_gemm_batch1(grid1, block1, torch.bfloat16, True,
        hidden_states.data_ptr(), w1.data_ptr(), output_g1_jit.data_ptr(),
        topk_ids.data_ptr(), topk_weight.data_ptr(), w_scale.data_ptr(), B, N, K)
torch.cuda.synchronize()
g1_jit_us = (time.time() - start) / N_ITER * 1e6

for _ in range(N_WARMUP):
    output_g1_hybrid.zero_()
    hip_hybrid(grid1, block1,
        hidden_states.data_ptr(), w1.data_ptr(), output_g1_hybrid.data_ptr(),
        topk_ids.data_ptr(), topk_weight.data_ptr(), w_scale.data_ptr(), B, N, K, 1)
torch.cuda.synchronize()
start = time.time()
for _ in range(N_ITER):
    output_g1_hybrid.zero_()
    hip_hybrid(grid1, block1,
        hidden_states.data_ptr(), w1.data_ptr(), output_g1_hybrid.data_ptr(),
        topk_ids.data_ptr(), topk_weight.data_ptr(), w_scale.data_ptr(), B, N, K, 1)
torch.cuda.synchronize()
g1_hybrid_us = (time.time() - start) / N_ITER * 1e6

for _ in range(N_WARMUP):
    output_g1_builtin.zero_()
    hip_builtin(grid1, block1,
        hidden_states.data_ptr(), w1.data_ptr(), output_g1_builtin.data_ptr(),
        topk_ids.data_ptr(), topk_weight.data_ptr(), w_scale.data_ptr(), B, N, K, 1)
torch.cuda.synchronize()
start = time.time()
for _ in range(N_ITER):
    output_g1_builtin.zero_()
    hip_builtin(grid1, block1,
        hidden_states.data_ptr(), w1.data_ptr(), output_g1_builtin.data_ptr(),
        topk_ids.data_ptr(), topk_weight.data_ptr(), w_scale.data_ptr(), B, N, K, 1)
torch.cuda.synchronize()
g1_builtin_us = (time.time() - start) / N_ITER * 1e6

print(f"  Performance (gemm1, {N_ITER} iters):")
print(f"    {'Kernel':<25} {'Latency':>10} {'vs JIT':>10}")
print(f"    {'-'*25} {'-'*10} {'-'*10}")
print(f"    {'JIT (reference)':<25} {g1_jit_us:>8.1f} µs {'—':>10}")
print(f"    {'Hybrid (asm loop)':<25} {g1_hybrid_us:>8.1f} µs {(g1_hybrid_us/g1_jit_us - 1)*100:>+8.1f}%")
print(f"    {'Builtin (pipelined)':<25} {g1_builtin_us:>8.1f} µs {(g1_builtin_us/g1_jit_us - 1)*100:>+8.1f}%")
print()

# =========================================================================
# GEMM2: no silu, block=64 (1 wave, atomic output)
# =========================================================================
print(f"--- GEMM2 (no activation, block=64, 1 wave) ---")
print(f"  Shape: [{B}, {K2}] x [{E}, {N2}, {K2}] → [{B}, {N2}]")
grid2 = [N2 // 32, B]
block2 = [64, 1, 1]
print(f"  Grid: {grid2}, Block: {block2}")
print(f"  MFMA K-steps: {K2//32}, Output tile: 16x32")
print()

# gemm2 test data
input_g2 = torch.randn([B, K2], dtype=torch.bfloat16)
w2 = torch.randn([E, N2, K2], dtype=torch.bfloat16)

# Correctness
output_jit = torch.zeros([B, N2], dtype=torch.bfloat16)
output_hybrid = torch.zeros([B, N2], dtype=torch.bfloat16)
output_builtin = torch.zeros([B, N2], dtype=torch.bfloat16)

moe_gemm_batch1(grid2, block2, torch.bfloat16, False,
    input_g2.data_ptr(), w2.data_ptr(), output_jit.data_ptr(),
    topk_ids.data_ptr(), topk_weight.data_ptr(), w_scale.data_ptr(), B, N2, K2)
torch.cuda.synchronize()

# hybrid: with_silu=0
hip_hybrid(grid2, block2,
    input_g2.data_ptr(), w2.data_ptr(), output_hybrid.data_ptr(),
    topk_ids.data_ptr(), topk_weight.data_ptr(), w_scale.data_ptr(), B, N2, K2, 0)
torch.cuda.synchronize()

# builtin: with_silu=0
hip_builtin(grid2, block2,
    input_g2.data_ptr(), w2.data_ptr(), output_builtin.data_ptr(),
    topk_ids.data_ptr(), topk_weight.data_ptr(), w_scale.data_ptr(), B, N2, K2, 0)
torch.cuda.synchronize()

output_jit2 = torch.zeros([B, N2], dtype=torch.bfloat16)
moe_gemm_batch1(grid2, block2, torch.bfloat16, False,
    input_g2.data_ptr(), w2.data_ptr(), output_jit2.data_ptr(),
    topk_ids.data_ptr(), topk_weight.data_ptr(), w_scale.data_ptr(), B, N2, K2)
torch.cuda.synchronize()

diff_g2_hybrid = (output_hybrid.float() - output_jit.float()).abs().max().item()
diff_g2_builtin = (output_builtin.float() - output_jit.float()).abs().max().item()
g2_noise = (output_jit.float() - output_jit2.float()).abs().max().item()

print(f"  Correctness (gemm2, max abs diff):")
print(f"    hybrid  vs jit: max_diff = {diff_g2_hybrid}")
print(f"    builtin vs jit: max_diff = {diff_g2_builtin}")
print(f"    jit vs jit:     max_diff = {g2_noise} (atomic noise floor)")
print(f"    hybrid  status: {'PASS' if diff_g2_hybrid <= max(g2_noise * 2, 16.0) else 'FAIL'}")
print(f"    builtin status: {'PASS' if diff_g2_builtin <= max(g2_noise * 5, 16.0) else 'FAIL'}")
print()

# Performance gemm2
for _ in range(N_WARMUP):
    output_jit.zero_()
    moe_gemm_batch1(grid2, block2, torch.bfloat16, False,
        input_g2.data_ptr(), w2.data_ptr(), output_jit.data_ptr(),
        topk_ids.data_ptr(), topk_weight.data_ptr(), w_scale.data_ptr(), B, N2, K2)
torch.cuda.synchronize()
start = time.time()
for _ in range(N_ITER):
    output_jit.zero_()
    moe_gemm_batch1(grid2, block2, torch.bfloat16, False,
        input_g2.data_ptr(), w2.data_ptr(), output_jit.data_ptr(),
        topk_ids.data_ptr(), topk_weight.data_ptr(), w_scale.data_ptr(), B, N2, K2)
torch.cuda.synchronize()
g2_jit_us = (time.time() - start) / N_ITER * 1e6

for _ in range(N_WARMUP):
    output_hybrid.zero_()
    hip_hybrid(grid2, block2,
        input_g2.data_ptr(), w2.data_ptr(), output_hybrid.data_ptr(),
        topk_ids.data_ptr(), topk_weight.data_ptr(), w_scale.data_ptr(), B, N2, K2, 0)
torch.cuda.synchronize()
start = time.time()
for _ in range(N_ITER):
    output_hybrid.zero_()
    hip_hybrid(grid2, block2,
        input_g2.data_ptr(), w2.data_ptr(), output_hybrid.data_ptr(),
        topk_ids.data_ptr(), topk_weight.data_ptr(), w_scale.data_ptr(), B, N2, K2, 0)
torch.cuda.synchronize()
g2_hybrid_us = (time.time() - start) / N_ITER * 1e6

for _ in range(N_WARMUP):
    output_builtin.zero_()
    hip_builtin(grid2, block2,
        input_g2.data_ptr(), w2.data_ptr(), output_builtin.data_ptr(),
        topk_ids.data_ptr(), topk_weight.data_ptr(), w_scale.data_ptr(), B, N2, K2, 0)
torch.cuda.synchronize()
start = time.time()
for _ in range(N_ITER):
    output_builtin.zero_()
    hip_builtin(grid2, block2,
        input_g2.data_ptr(), w2.data_ptr(), output_builtin.data_ptr(),
        topk_ids.data_ptr(), topk_weight.data_ptr(), w_scale.data_ptr(), B, N2, K2, 0)
torch.cuda.synchronize()
g2_builtin_us = (time.time() - start) / N_ITER * 1e6

print(f"  Performance (gemm2, {N_ITER} iters):")
print(f"    {'Kernel':<25} {'Latency':>10} {'vs JIT':>10}")
print(f"    {'-'*25} {'-'*10} {'-'*10}")
print(f"    {'JIT (reference)':<25} {g2_jit_us:>8.1f} µs {'—':>10}")
print(f"    {'Hybrid (asm loop)':<25} {g2_hybrid_us:>8.1f} µs {(g2_hybrid_us/g2_jit_us - 1)*100:>+8.1f}%")
print(f"    {'Builtin (pipelined)':<25} {g2_builtin_us:>8.1f} µs {(g2_builtin_us/g2_jit_us - 1)*100:>+8.1f}%")
print()

# =========================================================================
# Summary
# =========================================================================
print(f"=" * 70)
print(f"SUMMARY")
print(f"=" * 70)
print(f"  gemm1 (silu, block=256): [{B}x{K}] x [{E}x{N}x{K}] → [{B}x{N//2}]")
print(f"    JIT:     {g1_jit_us:.1f} µs")
print(f"    Hybrid:  {g1_hybrid_us:.1f} µs ({(g1_hybrid_us/g1_jit_us - 1)*100:+.1f}%)")
print(f"    Builtin: {g1_builtin_us:.1f} µs ({(g1_builtin_us/g1_jit_us - 1)*100:+.1f}%)")
print(f"  gemm2 (no act, block=64): [{B}x{K2}] x [{E}x{N2}x{K2}] → [{B}x{N2}]")
print(f"    JIT:     {g2_jit_us:.1f} µs")
print(f"    Hybrid:  {g2_hybrid_us:.1f} µs ({(g2_hybrid_us/g2_jit_us - 1)*100:+.1f}%)")
print(f"    Builtin: {g2_builtin_us:.1f} µs ({(g2_builtin_us/g2_jit_us - 1)*100:+.1f}%)")
print(f"  Total MoE pass:")
print(f"    JIT:     {g1_jit_us + g2_jit_us:.1f} µs")
print(f"    Hybrid:  {g1_hybrid_us + g2_hybrid_us:.1f} µs ({((g1_hybrid_us+g2_hybrid_us)/(g1_jit_us+g2_jit_us) - 1)*100:+.1f}%)")
print(f"    Builtin: {g1_builtin_us + g2_builtin_us:.1f} µs ({((g1_builtin_us+g2_builtin_us)/(g1_jit_us+g2_jit_us) - 1)*100:+.1f}%)")
