"""Tile sweep for the MXFP8 grouped GEMM, for the occupancy blog.

Two modes:
  (default)        sweep all configs, print VGPR/AGPR/spill/LDS + wall-clock TFLOP/s
  --config IDX     run ONE config in a tight loop (for rocprofv3 counter collection)

Fixed DSv3-ish shape; only the tile (BLOCK_M/N/K, warps, stages) varies.
"""
import os, sys, argparse, statistics
os.environ.setdefault("TRITON_ALWAYS_COMPILE", "1")
import torch
from torchao.prototype.mx_formats.mx_tensor import to_mx
from torchao.prototype.moe_training.kernels.mxfp8.rocm_mxfp8_mm import triton_mxfp8_grouped_mm
from torchao.prototype.moe_training.utils import generate_jagged_offs

dev = "cuda"
E, M, N, K = 8, 16384, 2048, 11264          # K%256==0, N%32==0, M%32==0 -> cdna4_scale path

# (BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages)
CONFIGS = [
    (64,  128, 256, 4, 2),
    (128, 128, 256, 4, 2),
    (128, 256, 256, 4, 2),
    (256, 128, 256, 8, 2),
    (256, 256, 256, 8, 2),
    (128, 256, 512, 4, 2),
]

def quantize_dim0(x):
    s, d = to_mx(x, elem_dtype=torch.float8_e4m3fn, block_size=32)
    return d, s

def build():
    torch.manual_seed(0)
    A = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
    B = torch.randn(E, N, K, device=dev, dtype=torch.bfloat16)
    A_fp8, A_scale = quantize_dim0(A)
    Bf, Bsf = quantize_dim0(B.reshape(E * N, K))
    B_fp8 = Bf.reshape(E, N, K); B_scale = Bsf.reshape(E, N, K // 32)
    offs = generate_jagged_offs(E, M, multiple_of=32)
    return A_fp8, B_fp8, A_scale, B_scale, offs

def run_cfg(args_in, cfg):
    bm, bn, bk, nw, ns = cfg
    return triton_mxfp8_grouped_mm(*args_in, BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk,
                                   num_warps=nw, num_stages=ns)

def time_ms(fn, iters=50, warmup=10):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s = torch.cuda.Event(True); e = torch.cuda.Event(True)
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return statistics.median(ts)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=int, default=-1)
    ap.add_argument("--iters", type=int, default=50)
    a = ap.parse_args()
    inp = build()
    flops = 2 * M * N * K

    if a.config >= 0:
        cfg = CONFIGS[a.config]
        fn = lambda: run_cfg(inp, cfg)
        for _ in range(10): fn()
        torch.cuda.synchronize()
        for _ in range(a.iters): fn()
        torch.cuda.synchronize()
        print(f"config {a.config} {cfg} ran {a.iters} iters")
        return

    print(f"shape E={E} M={M} N={N} K={K}  (FLOPs={flops/1e9:.0f} GFLOP)\n")
    hdr = f"{'cfg':>3} {'BM':>4}{'BN':>5}{'BK':>5} {'w':>2}{'s':>2}  {'VGPR':>5}{'AGPR':>5}{'spill':>6}{'LDS_KB':>7}  {'us':>8}{'TFLOPs':>9}"
    print(hdr); print("-" * len(hdr))
    for i, cfg in enumerate(CONFIGS):
        bm, bn, bk, nw, ns = cfg
        try:
            out = run_cfg(inp, cfg)            # compile + run
            k = run_cfg.__wrapped__ if False else None
            # grab the compiled kernel object for resource counts
            from torchao.prototype.moe_training.kernels.mxfp8.rocm_mxfp8_mm import _mxfp8_grouped_mm_kernel as KK
            ck = None
            for cache in getattr(KK, "device_caches", {}).values():
                d0 = cache[0]
                if d0:
                    ck = list(d0.values())[-1]
            nreg = getattr(ck, "n_regs", None); nsp = getattr(ck, "n_spills", None)
            shared = ck.metadata.shared if ck is not None else None
            # AGPR from amdgcn text
            agpr = None
            try:
                amd = ck.asm["amdgcn"]
                import re
                m = re.search(r"\.agpr_count:\s*(\d+)", amd); agpr = int(m.group(1)) if m else None
                m2 = re.search(r"\.vgpr_count:\s*(\d+)", amd); nreg = int(m2.group(1)) if m2 else nreg
            except Exception: pass
            t = time_ms(fn := (lambda c=cfg: run_cfg(inp, c)), iters=a.iters)
            tf = flops / (t * 1e-3) / 1e12
            lds = (shared/1024) if shared else 0
            print(f"{i:>3} {bm:>4}{bn:>5}{bk:>5} {nw:>2}{ns:>2}  {str(nreg):>5}{str(agpr):>5}{str(nsp):>6}{lds:>7.1f}  {t*1e3:>8.1f}{tf:>9.1f}")
        except Exception as ex:
            print(f"{i:>3} {bm:>4}{bn:>5}{bk:>5} {nw:>2}{ns:>2}  FAILED: {type(ex).__name__}: {str(ex)[:60]}")

if __name__ == "__main__":
    main()
