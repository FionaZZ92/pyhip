# Converting and Exporting PyHIP JIT Kernels to HIP

## PyHIP's Two Approaches for HIP Kernels

### 1. Assembly JIT Kernels (`@pyhip.jit`)

These are written in Python using the `JIT` class, generating CDNA assembly. The compilation pipeline is:

```
Python JIT code → Structured ASM IR → Optimization passes →
Inline ASM in HIP wrapper → hipcc → .co binary
```

**To export as standalone HIP/.co:**

```python
import os
os.environ['PYHIP_DUMP_DIR'] = './exported_kernels'  # dumps .s and .co files
os.environ['PYHIP_DEBUG_LOG'] = '1'                   # forces recompile + dump

# Call the kernel once to trigger compilation
my_kernel([grid], [block], *args)
# Check ~/.pyhip/ or PYHIP_DUMP_DIR for the generated .s and .co files
```

The `.co` (code object) files can be loaded by any HIP runtime via `hipModuleLoad`.

### 2. HIP Source Kernels (`@pyhip.module`)

These start as `.cpp` or `.s` files and are compiled through:

```
.cpp → hipcc -x hip --offload-device-only -S → .s → clang++ -x assembler → .co
```

**To export:** Simply grab the `.co` from `~/.pyhip/` cache directory.

---

## Practical Suggestions for Converting/Exporting

### A. Export the .co binary directly

```python
import pyhip
import os

# After running kernel at least once, find .co in:
cache_dir = os.getenv("PYHIP_CACHE_DIR", os.path.expanduser("~/.pyhip"))
# .co files here can be loaded with hipModuleLoad() from any HIP app
```

### B. Export as intermediate assembly (.s) for portability

```bash
# Set DUMP_LL=1 env var to also get LLVM IR
DUMP_LL=1 python your_kernel_script.py
# Outputs: ~/.pyhip/kernel_name.ll, .s, .co
```

### C. Convert JIT ASM kernel to standalone HIP C++ kernel

1. Run with `PYHIP_DUMP_DIR=./dump` to get the generated `.s`
2. Wrap the assembly in a HIP host file using inline asm:

   ```cpp
   __global__ void my_kernel(...) {
       asm volatile(
           // paste generated assembly here
       );
   }
   ```

3. Or load the `.co` directly via the HIP runtime API:

   ```c
   hipModule_t module;
   hipModuleLoad(&module, "kernel.co");
   hipFunction_t func;
   hipModuleGetFunction(&func, module, "kernel_symbol_name");
   hipModuleLaunchKernel(func, ...);
   ```

### D. Get kernel symbol names from .co

```bash
/opt/rocm/llvm/bin/llvm-objdump --dynamic-syms --demangle kernel.co
```

This gives you the function signature and mangled symbol name needed for `hipModuleGetFunction`.

### E. Re-target to different GPU arch

```bash
# Reassemble for a different arch
/opt/rocm/llvm/bin/clang++ -x assembler -target amdgcn-amd-amdhsa \
    -mcpu=gfx942 kernel.s -o kernel_gfx942.co
```

---

## Key Considerations

| Concern | Recommendation |
|---------|---------------|
| **Arch portability** | `.co` is arch-specific; re-assemble `.s` for each target (`gfx942`, `gfx950`, etc.) |
| **Register allocation** | pyhip handles this; exported `.s` has physical registers already allocated |
| **LDS usage** | Check `lds_allocator` size in your kernel; declare in host launch |
| **Integration with PyTorch** | Use `torch.cuda.current_stream()` and `.data_ptr()` as pyhip already does |
| **Caching** | pyhip caches by compile-time args hash; change args = new `.co` |
