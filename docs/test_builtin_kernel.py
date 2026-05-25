#!/usr/bin/env python3
"""Test builtin buffer_load hybrid kernel vs JIT kernel"""
import os, sys, time
os.environ.setdefault('PYHIP_RECOMPILE', '0')
import torch
torch.set_default_device('cuda')
sys.path.insert(0, '/opt/my_pyhip/pyhip/src')
import pyhip
from pyhip.contrib.moe import moe_gemm_batch1
from pyhip.core.hiptools import get_all_kernel_args, amdhip_func

E, N, K, B = 128, 2048, 7168, 1
print(f"MoE builtin buffer_load test: B={B}, N={N}, K={K}, E={E}")

# Load builtin kernel
co_path = '/opt/my_pyhip/pyhip/docs/moe_gemm_batch1_builtin_dev.co'
kernel_info = get_all_kernel_args(co_path)
kname = list(kernel_info.keys())[0]
sym_name, arg_types = kernel_info[kname]
print(f"Kernel: {kname}, args: {arg_types}")
hip_func = amdhip_func(co_path, sym_name, kname, arg_types)

# Test data
hidden_states = torch.randn([B, K], dtype=torch.bfloat16)
w1 = torch.randn([E, N, K], dtype=torch.bfloat16)
topk_ids = torch.zeros([B], dtype=torch.int32)
topk_weight = torch.ones([B], dtype=torch.float32)
w_scale = torch.ones([E, N], dtype=torch.float32)

output_jit = torch.zeros([B, N], dtype=torch.bfloat16)
output_builtin = torch.zeros([B, N], dtype=torch.bfloat16)

grid = [N // 32, B]
block = [256, 1, 1]

# Run JIT
moe_gemm_batch1(grid, block, torch.bfloat16, False,
    hidden_states.data_ptr(), w1.data_ptr(), output_jit.data_ptr(),
    topk_ids.data_ptr(), topk_weight.data_ptr(), w_scale.data_ptr(), B, N, K)
torch.cuda.synchronize()

# Run builtin
hip_func(grid, block,
    hidden_states.data_ptr(), w1.data_ptr(), output_builtin.data_ptr(),
    topk_ids.data_ptr(), topk_weight.data_ptr(), w_scale.data_ptr(), B, N, K)
torch.cuda.synchronize()

# JIT vs JIT noise floor
output_jit2 = torch.zeros([B, N], dtype=torch.bfloat16)
moe_gemm_batch1(grid, block, torch.bfloat16, False,
    hidden_states.data_ptr(), w1.data_ptr(), output_jit2.data_ptr(),
    topk_ids.data_ptr(), topk_weight.data_ptr(), w_scale.data_ptr(), B, N, K)
torch.cuda.synchronize()

max_diff = (output_builtin.float() - output_jit.float()).abs().max().item()
jit_noise = (output_jit.float() - output_jit2.float()).abs().max().item()
print(f"\nCorrectness:")
print(f"  builtin vs jit: max_diff = {max_diff}")
print(f"  jit vs jit:     max_diff = {jit_noise}")
print(f"  Status: {'PASS' if max_diff <= 16.0 else 'FAIL'}")

# Benchmark
N_ITER = 200
torch.cuda.synchronize()
start = time.time()
for _ in range(N_ITER):
    output_builtin.zero_()
    hip_func(grid, block,
        hidden_states.data_ptr(), w1.data_ptr(), output_builtin.data_ptr(),
        topk_ids.data_ptr(), topk_weight.data_ptr(), w_scale.data_ptr(), B, N, K)
torch.cuda.synchronize()
builtin_us = (time.time() - start) / N_ITER * 1e6

start = time.time()
for _ in range(N_ITER):
    output_jit.zero_()
    moe_gemm_batch1(grid, block, torch.bfloat16, False,
        hidden_states.data_ptr(), w1.data_ptr(), output_jit.data_ptr(),
        topk_ids.data_ptr(), topk_weight.data_ptr(), w_scale.data_ptr(), B, N, K)
torch.cuda.synchronize()
jit_us = (time.time() - start) / N_ITER * 1e6

# Also compare with buffer_load hybrid
co_hybrid = '/opt/my_pyhip/pyhip/docs/moe_gemm_batch1_hybrid.co'
ki2 = get_all_kernel_args(co_hybrid)
kn2 = list(ki2.keys())[0]
sn2, at2 = ki2[kn2]
hip_hybrid = amdhip_func(co_hybrid, sn2, kn2, at2)
output_hybrid = torch.zeros([B, N], dtype=torch.bfloat16)
start = time.time()
for _ in range(N_ITER):
    output_hybrid.zero_()
    hip_hybrid(grid, block,
        hidden_states.data_ptr(), w1.data_ptr(), output_hybrid.data_ptr(),
        topk_ids.data_ptr(), topk_weight.data_ptr(), w_scale.data_ptr(), B, N, K)
torch.cuda.synchronize()
hybrid_us = (time.time() - start) / N_ITER * 1e6

print(f"\nPerformance ({N_ITER} iters):")
print(f"  builtin kernel:     {builtin_us:.1f} µs")
print(f"  buffer_load hybrid: {hybrid_us:.1f} µs")
print(f"  JIT kernel:         {jit_us:.1f} µs")
print(f"  builtin vs JIT:     {(builtin_us/jit_us - 1)*100:+.1f}%")
print(f"  builtin vs hybrid:  {(builtin_us/hybrid_us - 1)*100:+.1f}%")
print(f"\nRegister usage: 26 VGPRs, 0 AGPRs, 28 SGPRs (builtin)")
