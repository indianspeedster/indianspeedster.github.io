---
title: "Beating torch.relu with FlyDSL: A Hands-On Guide to Bandwidth-Bound Kernels"
description: "Simon Boehm's CUDA matmul post, but in one dimension. Writing the simplest GPU kernel there is in AMD's FlyDSL, then taking it from 2.9 to 7.4 TB/s on an MI350X — 1.27x torch.relu at its best. Five versions with the code, what each optimization was actually worth, the two that measured negative, and the testing discipline a layout DSL asks for."
date: 2026-08-09
tags: ["GPU", "AMD", "CDNA4", "kernels", "FlyDSL", "MLIR", "bandwidth"]
draft: false
---

The canonical version of this exercise is [Simon Boehm's CUDA matmul post](https://siboehm.com/articles/22/CUDA-MMM): start with a naive kernel, climb an optimization ladder one rung at a time, measure every rung, end up near the vendor library. It's the best thing written on the genre and you should read it.

This is the same exercise **in one dimension**. I wanted to learn [FlyDSL](https://github.com/ROCm/FlyDSL) — AMD's Python DSL for authoring GPU kernels through an MLIR layout-algebra stack — and the thing newcomers bounce off in a layout DSL isn't the arithmetic, it's the addressing. A GEMM makes you learn a two-dimensional tiling hierarchy before you can compile anything. So I picked the smallest kernel that exists and took it as far as it would go: ReLU, `y = max(x, 0)`. One axis. One `max`. Everything you have to understand is about *where the data is*, which is exactly the part FlyDSL is opinionated about, and none of it is hidden behind a blocking strategy.

Five versions, naive to tuned, each measured against `torch.relu` on an MI350X. It ends up **2.9 → 7.4 TB/s**, and between 1.07× and 1.27× `torch.relu` on tensors large enough to be bandwidth-bound. Below about 64 MB of traffic it loses, and not for any reason to do with the kernel — Part 4 has that story. The interesting part is how unevenly the speedup is distributed across the five steps.

> **TL;DR.** ReLU is pure data movement — one `max` per element, no reuse, nothing to overlap — so there is exactly one figure of merit (achieved bandwidth) and a hard ceiling (~8 TB/s of HBM3E). Vectorizing to 128-bit copy atoms is worth **2.4×** and takes you past `torch.relu` on its own. Four further optimizations are worth **4% combined** at the shape I tuned on, two of them measured *negative*, and one turned out to be worth 22% — but only inside a narrow band of working-set sizes that a single-shape benchmark would have missed entirely. Along the way, the layout API asks for a specific kind of test discipline: on a GPU, "the output is correct" and "the kernel is correct" are different claims, and only the first one is what a `torch.allclose` checks.

All five kernels, the benchmark harness and the test suite are in [`flykernels`](https://github.com/indianspeedster/flykernels) — `flykernels/relu/v1_naive.py` through `v5_nontemporal.py`. Everything below was measured on an AMD Instinct MI350X VF (gfx950, CDNA4), ROCm 7.2, `flydsl` 0.3.0, `torch` 2.13.0+rocm7.2.

---

## The machine, the ceiling, and how I measured

Before any kernel, the number it's being graded against.

The MI350X is CDNA4 (gfx950): 256 CUs, 288 GB of HBM3E at a **peak of ~8 TB/s**, fronted by **256 MB of last-level Infinity Cache**. That last figure looks like trivia right now. It is the entire explanation for v5.

ReLU does one `max` per element and touches each element twice — read it, write it. In bf16 that's **one flop per four bytes**, an arithmetic intensity of 0.25. The MI350X does hundreds of TFLOPs. There is no version of this kernel that is compute-bound, no blocking scheme that helps, no reuse to exploit; put it on a roofline and it sits on the memory ceiling at every point. So the ladder has exactly one score:

```
GB/s = 2 · numel · sizeof(dtype) / seconds
```

and one target: 8000 GB/s, which nothing reaches. `torch.relu` gets 6762 at 8192² bf16 — 85% of peak — and that's the practical bar.

**Timing.** CUDA events around a loop of 100 iterations after 20 warmup runs, so per-launch event overhead amortizes and JIT compilation lands in the warmup ([`flykernels/bench.py`](https://github.com/indianspeedster/flykernels/blob/main/flykernels/bench.py)). Run-to-run spread at these sizes is roughly ±1.5%, which is worth remembering when you see a 0.2% "win" below.

**Correctness.** Two independent checks, and the second one is the reason Part 3 exists:

1. **Bit-exact equality** against `torch.relu`. ReLU is exact — a value is either passed through or replaced by zero — so there's no excuse for `allclose`. NaN, ±inf and −0.0 included.
2. **A guard region.** Allocate the output inside a larger buffer, fill the tail with a sentinel, run the kernel, assert the sentinel survived:

```python
SENTINEL = -7.0

def guarded_output(numel, dtype, guard=8192):
    buf = torch.full((numel + guard,), SENTINEL, device="cuda", dtype=dtype)
    return buf[:numel], buf     # write into buf[:numel], then check buf[numel:]
```

Checking the output tensor tells you what landed inside it and *nothing whatsoever* about what landed outside. Four separate times in this exercise I had a kernel that passed check 1 and was still wrong. Three of those were caught by check 2. Part 3 is the fourth.

## The ladder at a glance

bf16 8192×8192 — the shape I tuned on. Peak is 8000 GB/s.

| | version | what changed | GB/s | % of peak | vs torch | Δ vs prev |
| --- | --- | --- | --- | --- | --- | --- |
| v1 | [`v1_naive`](https://github.com/indianspeedster/flykernels/blob/main/flykernels/relu/v1_naive.py) | one element per thread | 2884 | 36% | 0.43× | — |
| v2 | [`v2_vectorized`](https://github.com/indianspeedster/flykernels/blob/main/flykernels/relu/v2_vectorized.py) | 128-bit copy atoms | 6924 | 87% | 1.02× | **2.40×** |
| v3 | [`v3_tuned`](https://github.com/indianspeedster/flykernels/blob/main/flykernels/relu/v3_tuned.py) | swept block size, unroll rejected | 6947 | 87% | 1.03× | +0.3% |
| v4 | [`v4_buffer`](https://github.com/indianspeedster/flykernels/blob/main/flykernels/relu/v4_buffer.py) | hardware bounds checking | 7190 | 90% | 1.06× | +3.5% |
| v5 | [`v5_nontemporal`](https://github.com/indianspeedster/flykernels/blob/main/flykernels/relu/v5_nontemporal.py) | non-temporal stores | **7221** | **90%** | **1.07×** | +0.4% |
| | `torch.relu` | | 6762 | 85% | — | |

The shape of that table is the post. One rung is worth 2.4× and clears the vendor library by itself; the remaining three are worth 4% put together, and one of them is a *negative* result I shipped anyway. If you came here for a long optimization story, ReLU is the wrong kernel — and knowing that in advance, from the arithmetic intensity, is the skill worth having.

The one thing the table hides is v5, which reads as noise here and is worth **22%** two shapes over. That's Part 2's punchline.

The post is in four parts:

- **Part 1 — The two ideas FlyDSL is built on.** Trace-time versus run-time, and layout algebra as "columns of columns". Everything else depends on these.
- **Part 2 — The ladder.** Five versions, what each changed, and what it measured.
- **Part 3 — Testing a kernel, not an output.** The four ways I got a correct answer out of an incorrect kernel.
- **Part 4 — The wall.** Where the time actually goes once the kernel stops being the bottleneck.

---

## Part 1 — The two ideas FlyDSL is built on

### Your Python does not run on the GPU

`@flyc.kernel` does not execute your function on the device. It **traces** it — runs it once, on the host, at compile time — and records the operations into MLIR. What runs on the GPU is the recording.

So every value inside a kernel is one of two kinds, and confusing them is the source of most early friction:

| Kind | Examples | Behaviour |
| --- | --- | --- |
| Compile-time | `BLOCK_THREADS`, `vec`, dtype | Real Python. `if` works, loops unroll, arithmetic happens now. |
| Traced | `fx.thread_idx.x`, `fx.block_idx.x`, `X.shape` | Placeholders. A Python `if` on one of these does not branch per-thread. |

This is why the vector width has to be chosen on the *host* and passed into the kernel builder: it determines the instruction encoding, so it must be a concrete number before tracing begins. A value that varies per call cannot participate. It's also why `fx.range_constexpr` exists — a plain `range()` inside a traced function becomes an `scf.for` loop, not an unroll.

### Layout algebra is "columns of columns"

FlyDSL never asks you to compute an address. Instead you repeatedly split an axis in two and pick a slice. There is one rule worth memorising:

> `logical_divide(t, K)` builds a grid whose columns are the successive K-element runs of `t`.

Slicing with `None` keeps an axis whole; slicing with an index pins it. So `slice(a, (None, bid))` means "keep every in-tile position, select tile number `bid`" — it takes a column.

![relu_01_layout_walk.svg](/blog/flydsl-relu/relu_01_layout_walk.svg)

In one dimension this is as concrete as it gets: `logical_divide` chops a ruler into equal segments, and `slice` picks one. Do it once to get from the tensor to a block's tile, then again to get from the block's tile to a thread's elements. Cut into columns, take my column, cut into columns, take my column. That is the entire data path of the kernel, and it's why the code below is two divides and two slices. In a GEMM you do the same thing on two axes at once with a hierarchy of tiles stacked on top — same algebra, four times the bookkeeping, which is why I'd rather you meet it here.

If your instincts are CUDA-shaped, the mapping is close to one-to-one:

| what you want | CUDA | FlyDSL |
| --- | --- | --- |
| my block / thread index | `blockIdx.x`, `threadIdx.x` | `fx.block_idx.x`, `fx.thread_idx.x` |
| this block's slice of the data | pointer arithmetic | `logical_divide` then `slice` |
| a 128-bit load | cast to `float4` | a copy atom of 128 bits |
| registers to hold it | a local `float4` | `fx.make_rmem_tensor(vec, ty)` |
| don't run off the end | `if (i < n)` | `pred=`, or a buffer descriptor (v4) |

The layout ceremony looks like a lot for something this simple. It pays for itself in Part 2: when the vector width goes from 1 to 8, **only the chunk sizes change** — every divide and slice line stays byte-for-byte identical.

---

## Part 2 — The ladder

### v1 — naive, one element per thread

The goal here is a correct kernel and nothing else.

```python
@flyc.kernel
def relu_kernel(X: fx.Tensor, Y: fx.Tensor):
    bid = fx.block_idx.x
    tid = fx.thread_idx.x
    numel = X.shape.unpack()

    base = bid * BLOCK_THREADS + tid
    in_bounds = fx.make_rmem_tensor(1, fx.Boolean)
    in_bounds[0] = base < numel

    tile = fx.make_layout(BLOCK_THREADS, 1)
    one  = fx.make_layout(1, 1)

    eX = fx.logical_divide(fx.slice(fx.logical_divide(X, tile), (None, bid)), one)
    eY = fx.logical_divide(fx.slice(fx.logical_divide(Y, tile), (None, bid)), one)

    atom = fx.make_copy_atom(ATOM[atom_bits](), elem_ty)
    rX = fx.make_rmem_tensor(1, elem_ty)
    rY = fx.make_rmem_tensor(1, elem_ty)

    fx.copy_atom_call(atom, fx.slice(eX, (None, tid)), rX, pred=in_bounds)

    x = fx.memref_load_vec(rX)
    fx.memref_store_vec(x.maximumf(fx.full_like(x, 0.0)), rY)

    fx.copy_atom_call(atom, rY, fx.slice(eY, (None, tid)), pred=in_bounds)
```

Two details worth pausing on.

`in_bounds` is a one-element register fragment holding this thread's "am I past the end?" flag. It is a *tensor* rather than a bare boolean because `pred=` expects one bit **per copy atom**, and in general a single copy call can drive several atoms. Here it's the degenerate case: one atom, one bit. That stops being degenerate in v3, and the fact that it's per-atom rather than per-element is what [case 2](#2-one-bit-cannot-describe-eight-elements) runs into.

And it's `maximumf`, not `maxnumf`. The first propagates NaN, which is what `torch.relu` does; the second returns the non-NaN operand and would disagree on exactly the inputs nobody writes a test for.

**Result: 2884 GB/s — 36% of peak, 0.43× torch.**

### v2 — vectorize

**Why v1 is slow:** it isn't bandwidth, it's instruction count. Each thread moves **2 bytes per instruction** in bf16. The memory system wants a full 128 bits per lane, so v1 issues eight times more instructions than necessary, each carrying its own issue cost and latency. At 36% of peak, the memory system is idling while the pipe drains address arithmetic.

![relu_02_access_width.svg](/blog/flydsl-relu/relu_02_access_width.svg)

The fix is to give each thread `vec` contiguous elements — 4 for f32, 8 for bf16 — and move them in one 128-bit atom. Here the layout algebra earns its keep, because the change is entirely in the chunk sizes:

```python
# v1
tile = fx.make_layout(BLOCK_THREADS, 1)
one  = fx.make_layout(1, 1)
atom_bits = elem_ty.width

# v2
tile = fx.make_layout(BLOCK_THREADS * vec, 1)
one  = fx.make_layout(vec, 1)
atom_bits = vec * elem_ty.width
```

Every `logical_divide`, `slice` and `copy_atom_call` line is untouched. So is the arithmetic — `maximumf` is already elementwise, so it operates on an 8-wide value with no edit at all.

**Result: 6924 GB/s — 87% of peak, and past torch's 6762.** A 2.4× step. This is the entire optimization story; everything after it is rounding error by comparison.

### v3 — unroll (a negative result)

**Why v2 might be slow:** each thread issues one load, waits on it, computes, stores. Latency is hidden only by occupancy. The textbook remedy is to give each thread several independent vectors and issue *all* the loads before consuming any, so a single thread keeps multiple requests in flight.

I built it, then swept threads-per-block against unroll factor:

| threads / block | unroll 1 | unroll 2 | unroll 4 | unroll 8 |
| --- | --- | --- | --- | --- |
| 64 | 7000 | 6898 | 6759 | 6672 |
| **128** | **7023** | 6906 | 6681 | 6543 |
| 256 | 7020 | 6724 | 6548 | 6492 |
| 512 | 6974 | 6723 | 6504 | 6709 |
| 1024 | 3212 | 6560 | 6639 | 6764 |

Unrolling hurts monotonically in every row.

![relu_05_unroll_tradeoff.svg](/blog/flydsl-relu/relu_05_unroll_tradeoff.svg)

v2 was already at 87% of peak, so extra in-flight loads per thread buy nothing — while the register pressure and the correspondingly smaller grid cost real occupancy. Total memory in flight is roughly unchanged; the number of waves available to hide anything *else* drops by 4×. The premise was wrong: you can't hide latency better on a machine that is already delivering nearly all the bandwidth it has.

The best cell in the sweep is 128 threads with no unroll, and it beats 256 threads by 0.04% — which is to say, by nothing at all. I shipped that configuration because it's the measured optimum, not because it's a win; end to end v3 lands 0.3% over v2, comfortably inside run-to-run noise. I kept the version anyway. A rung that says "I tried the obvious thing and measured it losing" is worth as much as one that won, and the unroll machinery is what the sweep drives.

Two loose ends I'll be honest about. The 1024-thread, unroll-1 cell at 3212 GB/s is the one number in this sweep I can't account for — every neighbour is ~7000, and I left it in rather than quietly trim the row. And blocks above 256 threads are rejected outright unless the kernel declares `known_block_size`, because the AMDGPU default `max_flat_workgroup_size` is 256. The error message names the exact fix, which is more than most toolchains manage.

### v4 — let the hardware check the bounds

**Why v3 is slow:** every thread computes `base = bid * tile + tid * vec`, compares it against `numel`, materialises a predicate register, and carries that predicate into both copies. None of that is data movement. At 87% of peak the remaining 13% is exactly this sort of per-thread overhead, and the machine has a way to do it for free.

`fx.rocdl.make_buffer_tensor` wraps a tensor in an AMD **buffer resource**: a descriptor carrying a base address and a record count. Accesses through it are range-checked in silicon — an out-of-range load returns zero, an out-of-range store is dropped, with no branch, no predicate register, and no index arithmetic that can overflow.

That deletes `in_bounds`, both `pred=` arguments, and the `base` computation. The kernel gets materially shorter, and faster: **7190 GB/s — 90% of peak, 1.06× torch.**

The cost is portability: `fx.rocdl.*` is CDNA-specific, so unlike v1–v3 this will not compile for RDNA. That's the trade — hardware bounds checking for target neutrality.

It also has a trap in it, which is [case 4](#4-a-bounds-check-that-is-off-by-default)'s business.

### v5 — non-temporal stores

**Why v4 is slow:** ReLU streams. Every element is read once and written once and nothing is revisited, so caching either stream is pollution — the lines are evicted long before anyone could reuse them, and they displace whatever else the cache was holding. CDNA buffer atoms take a cache modifier for exactly this: `BufferCopy(bits, 0)` is normal, `BufferCopy(bits, 2)` is non-temporal. I gave the load and the store separate atoms so they could be set independently:

| load | store | GB/s |
| --- | --- | --- |
| cached | cached | 7077 |
| **cached** | **non-temporal** | **7092** |
| non-temporal | cached | 6231 |
| non-temporal | non-temporal | 6244 |

(This is a separate run from the ladder table, which is why the cached/cached row reads 7077 rather than v4's 7190. Same shape, same kernel; that gap is the ±1.5% spread.)

Half the theory held. A non-temporal *store* is worth +0.2% — inside the noise, which is what "this data is never read again" should look like. A non-temporal *load* costs 12%: the input stream evidently does want to pass through cache, presumably because prefetch and coalescing work on cached lines.

I nearly shipped that as another negative result. Then the full benchmark showed v5 beating v4 by 22% at one particular shape, reproducible to three significant figures across repeated runs. Sweeping the working set explains it:

| total traffic | v4 cached | v5 nt-store | gain |
| --- | --- | --- | --- |
| 34 MB | 1419 | 1465 | 1.03× |
| 134 MB | 5805 | 5803 | 1.00× |
| 268 MB | 7150 | 7188 | 1.01× |
| **403 MB** | 6023 | **7325** | **1.22×** |
| **537 MB** | 6035 | **7394** | **1.23×** |
| 805 MB | 6139 | 6306 | 1.03× |
| 1074 MB | 6131 | 6305 | 1.03× |

![relu_07_cache_band.svg](/blog/flydsl-relu/relu_07_cache_band.svg)

The gain exists only in a band — and the *input* footprint inside that band is 201–268 MB (half the traffic, since reads and writes are equal), which brackets the MI350X's 256 MB of last-level cache almost exactly. Below it, the input fits comfortably even while the output stream pollutes the cache, so protecting it changes nothing. Above it, the input cannot fit no matter what you do. Right at the boundary, evicting the write stream is the difference between the read stream staying resident and being thrashed out.

I'd offer that as a hypothesis consistent with the data rather than a proven mechanism — I did not instrument the cache to confirm it. The transferable lesson is narrower and safer: **a single-shape benchmark would have reported this optimization as worthless.**

### Where it landed

| shape (bf16) | v1 | v2 | v3 | v4 | v5 | torch | best ÷ torch |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 8192 × 8192 | 2884 | 6924 | 6947 | 7190 | **7221** | 6762 | 1.07× |
| 16384 × 8192 | 2074 | 6009 | 6084 | 6092 | **7406** | 5819 | 1.27× |
| 32768 × 8192 | 2048 | 6063 | 6128 | 6171 | **6308** | 5809 | 1.09× |
| 4096 × 14336 | 2879 | 6122 | 6131 | 7148 | **7161** | 6668 | 1.07× |

The best number on the board, 7406 GB/s, is 93% of the 8 TB/s peak. 284 tests across the five versions, bit-exact against `torch.relu` on every shape and dtype, NaN, ±inf and −0.0 included.

---

## Part 3 — Testing a kernel, not an output

A layout DSL hands you explicit control over addressing, and the flip side of that control is that a kernel can compute a perfectly correct answer while touching memory it doesn't own. Four separate times in this exercise I had a kernel that passed a bit-exact comparison against `torch.relu` and was nonetheless wrong. The guard region from the methodology section caught three of them.

Each case attaches to a specific rung, so here they are with their addresses:

| | failure | rung | caught by |
| --- | --- | --- | --- |
| 1 | predicate never masks | v1's first draft (the `vectorAdd` idiom) | guard region |
| 2 | one predicate bit, eight elements | v2's first draft | guard region |
| 3 | 64× over-launch, silent until int32 overflows | v2's second draft | a crash, 40s into a benchmark |
| 4 | hardware bounds checking off by default | v4's first draft | guard region — then value comparison |

The shapes these take are worth recognising, so here they are in full.

### 1. Predication that doesn't predicate

I built v1 on the idiom in `examples/01-vectorAdd.py`. It's elegant: build an "identity" tensor whose value at each position *is* its own coordinate, tile it exactly like the data, and compare each thread's coordinate against the tensor shape.

```python
idC = fx.make_view((0, 0), fx.make_identity_layout((M, N)))
cC  = fx.flat_divide(idC, TileMN)[None, None, bid_x, bid_y]
...
thr_pC[a] = fx.elem_less(thr_cC[a], (M, N))
fx.copy(copy_atom, thr_rC, thr_gC, pred=thr_pC)
```

On 0.3.0 / gfx950 the mask comes out permanently open. Running that example verbatim on its own `M, N = 100, 1000`, with a sentinel-filled guard region allocated immediately behind `C`, the writes run **4024 elements past the end of the output** — while `torch.allclose(A + B, C)` still returns `True`. Shapes that divide the tile exactly (`104 × 1024`, say) are unaffected, which is why it isn't obvious.

How I tracked it down, in the order it happened, because the method generalises better than the answer:

1. **Is the predicate plumbing working at all?** Force the predicate fragment to a literal `False`. Zero elements written — so `pred=` is honoured, and the predicate *values* were wrong.
2. **What values?** `fx.printf` from hundreds of lanes interleaves into unreadable soup; I got `blk=(blk=(SIZES pred=blk=(`. The trick that works is to write the values into a tile-aligned tensor **unpredicated**, so every thread has a legal slot, and read it back on the host. Dump the thread id alongside as a check on the readout path itself.

That produced the answer:

![relu_04_coordinate_bug.svg](/blog/flydsl-relu/relu_04_coordinate_bug.svg)

The coordinate tensor's per-tile offset advances by **1 per block index instead of by the tile extent**. Every coordinate therefore stays inside the logical shape, `elem_less(coord, shape)` is true for every thread in every block, and the predicate never masks.

It's a quiet failure by construction: the in-bounds values are all correct, and the surplus blocks write past them into whatever the allocator handed out next — on a fresh allocation, usually slack. An output-only assertion cannot see it.

The workaround is to compute the predicate arithmetically from `bid` and `tid`, which is what v1 above does. You give up some of the abstraction the layout API is selling, but you can verify it directly. (Worth reporting upstream; the identity-tensor route is the more expressive one and it would be good to have it back.)

### 2. One bit cannot describe eight elements

The predicate is one bit **per atom**. At the tail of a tensor, a thread's 8-element vector may be half valid, and there is no way to say so.

![relu_03_straddling_atom.svg](/blog/flydsl-relu/relu_03_straddling_atom.svg)

My first v2 hardcoded `vec = 8` and did exactly this on every size not divisible by 8: correct output, 21 failing guard tests. The fix is a width that always divides the extent:

```python
def pick_vec_width(numel, elem_bits):
    vec = 128 // elem_bits
    while numel % vec:
        vec //= 2
    return vec
```

It costs nothing on real tensors — activation sizes are highly composite, so `vec` stays at maximum — and engages exactly where you'd otherwise corrupt memory. It buys natural 16-byte alignment for free too.

### 3. The over-launch that was harmless until it wasn't

My second v2 had a missing pair of parentheses:

```python
grid_x = (numel + BLOCK_THREADS*vec - 1) // BLOCK_THREADS*vec   # (… // 256) * 8
```

Python reads `//` and `*` left to right, so this divides by the *thread* count and then multiplies — launching **64× too many blocks**. Every excess block computed its predicate, found itself entirely masked, and exited. Results stayed correct. It was pure waste, and invisible.

Until the tensor got large enough:

| 32768 × 8192 bf16, 268M elements | |
| --- | --- |
| blocks launched | 8,388,664 (should be 131,072) |
| highest thread index | 17,179,983,864 |
| int32 max | 2,147,483,647 |

The index wraps negative, which makes `base < numel` true, so a block that should have been entirely masked fires its copy at a garbage address: `CUDA error: an illegal memory access was encountered`, forty seconds into a benchmark run.

Widening the atom in v2 didn't create that bug. It just pushed the arithmetic over 2³¹ and turned silent waste into a crash. The predicate had been quietly doing the grid calculation's job, and the tests it passed were real tests.

### 4. A bounds check that is off by default

v4's first attempt failed the guard tests exactly like an unpredicated kernel. The reason is in the signature:

```python
def make_buffer_tensor(tensor, max_size: bool = True, ...):
    """Construct a new buffer-resource-backed tensor ... for hardware
    OOB-checked loads / stores ...

    ``max_size=True`` (default) sets the descriptor to ``0xFFFFFFFF``.
    """
```

The function advertises hardware OOB checking and defaults to a descriptor spanning all of memory, which checks nothing. You need `max_size=False`.

![relu_06_buffer_descriptor.svg](/blog/flydsl-relu/relu_06_buffer_descriptor.svg)

With checking actually enabled, I expected `pick_vec_width` to become unnecessary — surely the hardware services the valid part of a straddling access. It does not:

| numel | divides by 8 | bytes past end | output correct |
| --- | --- | --- | --- |
| 100,000 | yes | 0 | yes |
| 4,257 | no | 0 | **no** |
| 1,023 | no | 0 | **no** |
| 7 | no | 0 | **no** |

The straddling access is **dropped entirely**, not partially serviced. Memory stays pristine; the valid elements inside that access never get written. Hardware range checking buys memory safety, not correctness — the width rule stays.

This is also the one case of the four that the guard region *could not* catch, because it corrupts the answer rather than the memory. It took the ordinary correctness tests, which had been quietly passing everything for days. The two failure modes are genuinely different, and you want both checks — the guard region catches writes you didn't intend, and the value comparison catches writes you intended and didn't get.

---

## Part 4 — The wall

At 4096 × 4096 every version measures the same. None of them is the reason:

| | host time to submit one call |
| --- | --- |
| FlyDSL `@flyc.jit` | 30.9 µs |
| `torch.relu` | 4.2 µs |

Roughly 25 µs of that sits inside the JIT wrapper's call path — argument marshalling and cache lookup — and it is pure CPU time. Any tensor whose kernel runs in less than that is measuring dispatch, not bandwidth. For ReLU on this machine, that's everything under about 64 MB of total traffic — a 4096 × 4096 bf16 tensor and below. It's also why the 34 MB row in the v5 sweep reads 1419 GB/s instead of something near 7000: that measurement never touched the memory system's limits.

Which means the two largest wins available aren't kernel optimizations at all:

- **Ahead-of-time compilation.** FlyDSL ships an AOT path that skips the JIT and loads from disk cache. Worth ~8× on small tensors — more than every kernel optimization in this post combined.
- **Fusion.** In a real model ReLU never runs as its own kernel. Folded into the preceding GEMM's epilogue it costs approximately zero instead of two bytes per element of HBM traffic. Infinite speedup, by deletion.

---

## What I'd take away

**The ladder for a memory-bound op is short.** One optimization was worth 2.4×; four more were worth 4% combined, and two of those measured negative. If you want a kernel with a deep optimization story, pick one with *reuse* — a reduction, a GEMM — where blocking and shared memory have something to work with. ReLU's ceiling is set by physics you cannot program around, and an arithmetic intensity of 0.25 tells you that before you write a line.

**A passing test is not a correct kernel — know which mechanism made it pass.** My 64× over-launch was covered by a predicate that happened to mask it. The identity-tensor predicate was covered by an allocator that happened to have slack. Both were one unrelated change away from being a crash, and in the first case the unrelated change was widening the copy atom, which pushed an index over 2³¹ forty seconds into a benchmark.

**Check the defaults.** `make_buffer_tensor` offers hardware bounds checking and starts with it off. `maximumf` and `maxnumf` differ only on NaN. `vectorAdd`'s predication is exercised only by shapes that don't divide the tile. None of these are hidden — they're all right there in the signature or the semantics — but a layout DSL gives you enough rope that reading them properly is worth the ten minutes.

**Measure across shapes, not at one.** The non-temporal store looked worthless at 8192² and was worth 22% two shapes later. One number is not a benchmark.

Next up: a reduction, where the ladder actually goes somewhere.

---

## Further reading

- [Simon Boehm, *How to Optimize a CUDA Matmul Kernel*](https://siboehm.com/articles/22/CUDA-MMM) — the two-dimensional version of this exercise, and the reason this post has the shape it does.
- [FlyDSL](https://github.com/ROCm/FlyDSL) — the DSL itself. Start with `examples/01-vectorAdd.py`, with case 1 above in mind.
- [`flykernels`](https://github.com/indianspeedster/flykernels) — all five kernels, the sweep scripts, and the 284 tests.
- [AMD CDNA4 ISA reference](https://www.amd.com/en/search/documentation/hub.html) — buffer resource descriptors, cache modifiers, `max_flat_workgroup_size`.
- [Occupancy math on the MI355X](/blog/occupancy-math-mi355x) — my earlier post on the resource limiters behind v3's sweep.
