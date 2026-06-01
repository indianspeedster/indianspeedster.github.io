"""ILP microbenchmark for the occupancy blog: dense MXFP8 matmul (E=1),
fixed square shape, sweep the per-wave tile (= ILP knob). Bigger tile =>
more independent MFMA fragments per wave => more ILP, lower occupancy.

  default      sweep configs, print VGPR/LDS/spill + wall-clock TFLOP/s
  --config IDX run ONE config in a loop (for rocprofv3 counters)
"""
import os, sys, argparse, statistics
os.environ.setdefault("TRITON_ALWAYS_COMPILE", "1")
import torch
from torchao.prototype.mx_formats.mx_tensor import to_mx
from torchao.prototype.moe_training.kernels.mxfp8.rocm_mxfp8_mm import triton_mxfp8_grouped_mm

dev = "cuda"
M = N = K = 8192                      # dense square; K%256==0, N%32==0, M%32==0
E = 1

# (BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages) -- ascending per-wave tile
CONFIGS = [
    (64,  64,  256, 4, 2),
    (128, 64,  256, 4, 2),
    (128, 128, 256, 4, 2),
    (256, 128, 256, 8, 2),
    (256, 256, 256, 8, 2),
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
    offs = torch.tensor([M], dtype=torch.int32, device=dev)   # single dense group
    return A_fp8, B_fp8, A_scale, B_scale, offs

def run_cfg(inp, cfg):
    bm, bn, bk, nw, ns = cfg
    return triton_mxfp8_grouped_mm(*inp, BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk,
                                   num_warps=nw, num_stages=ns)

def time_ms(fn, iters=60, warmup=12):
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
    ap.add_argument("--iters", type=int, default=60)
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

    print(f"dense MXFP8 matmul  M=N=K={M}  (FLOPs={flops/1e9:.0f} GFLOP)\n")
    hdr = f"{'cfg':>3} {'BM':>4}{'BN':>5}{'BK':>5} {'w':>2}{'s':>2}   {'us':>9}{'TFLOPs':>9}{'%peak':>7}"
    print(hdr); print("-" * len(hdr))
    for i, cfg in enumerate(CONFIGS):
        bm, bn, bk, nw, ns = cfg
        try:
            run_cfg(inp, cfg)
            t = time_ms(lambda c=cfg: run_cfg(inp, c), iters=a.iters)
            tf = flops / (t * 1e-3) / 1e12
            print(f"{i:>3} {bm:>4}{bn:>5}{bk:>5} {nw:>2}{ns:>2}   {t*1e3:>9.1f}{tf:>9.1f}{tf/5000*100:>6.0f}%")
        except Exception as ex:
            print(f"{i:>3} {bm:>4}{bn:>5}{bk:>5} {nw:>2}{ns:>2}   FAILED: {type(ex).__name__}: {str(ex)[:50]}")

if __name__ == "__main__":
    main()
