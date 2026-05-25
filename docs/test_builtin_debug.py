#!/usr/bin/env python3
"""Debug: compare single K-step to find where builtin diverges from JIT"""
import os, sys
os.environ.setdefault('PYHIP_RECOMPILE', '0')
import torch
torch.set_default_device('cuda')
sys.path.insert(0, '/opt/my_pyhip/pyhip/src')
import pyhip
from pyhip.contrib.moe import moe_gemm_batch1
from pyhip.core.hiptools import get_all_kernel_args, amdhip_func

# Use small K to isolate
E, N, K, B = 128, 2048, 32, 1  # K=32 → single K step!

co_path = '/opt/my_pyhip/pyhip/docs/moe_gemm_batch1_builtin_dev.co'
kernel_info = get_all_kernel_args(co_path)
kname = list(kernel_info.keys())[0]
sym_name, arg_types = kernel_info[kname]
hip_func = amdhip_func(co_path, sym_name, kname, arg_types)

# Use deterministic input
torch.manual_seed(42)
hidden_states = torch.randn([B, K], dtype=torch.bfloat16)
w1 = torch.randn([E, N, K], dtype=torch.bfloat16)
topk_ids = torch.zeros([B], dtype=torch.int32)
topk_weight = torch.ones([B], dtype=torch.float32)
w_scale = torch.ones([E, N], dtype=torch.float32)

grid = [N // 32, B]
block = [256, 1, 1]

# Run JIT 
output_jit = torch.zeros([B, N], dtype=torch.bfloat16)
moe_gemm_batch1(grid, block, torch.bfloat16, False,
    hidden_states.data_ptr(), w1.data_ptr(), output_jit.data_ptr(),
    topk_ids.data_ptr(), topk_weight.data_ptr(), w_scale.data_ptr(), B, N, K)
torch.cuda.synchronize()

# Run builtin
output_builtin = torch.zeros([B, N], dtype=torch.bfloat16)
hip_func(grid, block,
    hidden_states.data_ptr(), w1.data_ptr(), output_builtin.data_ptr(),
    topk_ids.data_ptr(), topk_weight.data_ptr(), w_scale.data_ptr(), B, N, K)
torch.cuda.synchronize()

# Compute reference with torch
ref = hidden_states.float() @ w1[0].float().T  # [1, N]
ref_bf16 = ref.bfloat16()

diff_jit = (output_jit.float() - ref_bf16.float()).abs()
diff_builtin = (output_builtin.float() - ref_bf16.float()).abs()
diff_j_vs_b = (output_jit.float() - output_builtin.float()).abs()

print(f"K={K}, single K-step")
print(f"  JIT vs torch ref:     max_diff = {diff_jit.max().item():.1f}")
print(f"  builtin vs torch ref: max_diff = {diff_builtin.max().item():.1f}")
print(f"  JIT vs builtin:       max_diff = {diff_j_vs_b.max().item():.1f}")
print(f"\n  First 8 values:")
print(f"    JIT:     {output_jit[0,:8].float().tolist()}")
print(f"    builtin: {output_builtin[0,:8].float().tolist()}")
print(f"    ref:     {ref_bf16[0,:8].float().tolist()}")

# Check if builtin is all zeros
print(f"\n  builtin all zeros? {(output_builtin == 0).all().item()}")
print(f"  builtin nonzero count: {(output_builtin != 0).sum().item()} / {N}")
