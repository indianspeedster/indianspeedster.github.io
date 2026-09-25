---
title: "MXFP8: Microscale Floating Point 8 — How Block-Level Scaling Makes 8-Bit Training Work"
description: "A from-first-principles look at the MXFP8 datatype: why regular FP8 isn't enough, how a shared power-of-two scale per 32-element block restores FP32-like reach, the hardware plumbing on AMD CDNA4, and why the block size is 32."
date: 2026-06-08
tags: ["GPU", "AMD", "FP8", "MXFP8", "quantization", "LLM", "CDNA4"]
draft: false
---

Training LLMs in 8-bit floating point has an obvious appeal: halve the bytes per element relative to BF16, halve memory traffic, and double theoretical compute throughput on the matrix cores. The catch is that eight bits leave very little room for an exponent. The common FP8 variant, E4M3, has four exponent bits and tops out at 448 — too little headroom for activations and weights that spread over many orders of magnitude, and too little *footroom* for small gradients. You can step up to E5M2 (max 57344), but then you've only got two mantissa bits left and your precision drops by half.

**MXFP8 solves this by adding a shared scale per block.** Instead of each 8-bit element standing on its own, 32 elements share a single 8-bit power-of-two scale. The per-element values stay in FP8, and the scale multiplies the whole block. Net effect: across a tensor, FP32-like reach, delivered by 8-bit storage.

> **In one sentence:** MXFP8 stores FP8 values in blocks of 32 elements, where each block shares a power-of-two scaling factor encoded as an 8-bit E8M0 exponent. Every stored element is an FP8 number (E4M3 or E5M2 — the OCP spec defines both; E4M3 is the common choice and the one this post uses); the block scale multiplies the entire block to recover the original magnitude.

> **Why 32?** Every halving of the block size doubles the scale overhead — 3.1% at 32, 6.25% at 16, 12.5% at 8. Every doubling lets a single outlier set the scale for twice as many neighbours. 32 is the point the OCP MX specification settled on after empirical studies across training and inference workloads, and it maps cleanly onto hardware. (Part 4 has the details.)

I want to unpack MXFP8 from silicon to software — the bit layout, the math, how it maps to the matrix cores on AMD MI355X (CDNA4), and the reasoning behind the block size.

> **TL;DR.** MXFP8 packs 8-bit elements into blocks of 32 with one shared E8M0 scale per block. Storage is 1.03 bytes/element (1 byte of data + 1/32 byte of scale) vs 2 bytes for BF16. The block scale extends the representable range from E4M3's [2⁻⁹, 448] to roughly [2⁻¹³⁶, 2¹³⁶] across the tensor — FP32-like reach, though any *single* block still spans only E4M3's width. On CDNA4, scaled MFMA instructions consume MXFP8 operands and their scales natively, at twice the BF16 matrix rate, accumulating in FP32. It's a ~2× memory and compute win for transformer training and inference.

---

## Part 1 — Why FP8 Alone Isn't Enough

An 8-bit floating-point format makes an existential choice: how do you split your 8 bits between exponent and mantissa?

![The bit patterns of E5M2, E4M3 and E8M0: each format's largest value for the two FP8 element types, and the byte 127 (a scale of 1) for the E8M0 scale.](/blog/mxfp8/mxfp8_01_bit_layouts.svg "The three formats MXFP8 is built from. E5M2 spends its bits on range, E4M3 on precision, and the E8M0 scale spends every bit on exponent.")

### E4M3: Range is the bottleneck

E4M3 (4 exponent bits, 3 mantissa bits) is the workhorse. Its positive range spans 2⁻⁹ ≈ 0.00195 (smallest subnormal; the smallest *normal* is 2⁻⁶ ≈ 0.0156) to 448 — about 5.4 orders of magnitude. That's plenty for a well-behaved slice of a tensor, but not for a whole tensor: transformer activations carry outlier channels orders of magnitude larger than the bulk of the values, and gradients shrink to values far below E4M3's floor. Under the OCP definition, E4M3 has no infinities — only `S.1111.111` encodes NaN, and 448 is `S.1111.110` — so there is no headroom above 448: out-of-range values must saturate to ±448 or become NaN.

The workaround in practice: compute a per-tensor maximum absolute value, scale the tensor so that maximum lands near 448, quantize, then fold the scale back in after the GEMM. This solves the overflow problem but introduces a **per-tensor quantization error**: one outlier element sets the scale for everyone, pushing the small elements of the tensor down into E4M3's subnormal range or flushing them to zero.

### E5M2: Precision is the bottleneck

E5M2 (5 exponent bits, 2 mantissa bits) takes the opposite trade. Range jumps to [2⁻¹⁶ ≈ 0.000015, 57344] — about 9.6 orders of magnitude, the same range as FP16 (but far narrower than BF16, which shares FP32's 8-bit exponent). With only two mantissa bits, though, adjacent values are 25% apart, so the worst-case rounding error is 12.5%. That is coarse for weights and activations in the forward pass. It is tolerable for gradients — which is why "hybrid" FP8 recipes use E4M3 forward and E5M2 for gradients, always accumulating in higher precision (FP32 master weights, FP32 GEMM accumulators).

### The fundamental tension

You need wide dynamic range *and* fine per-element precision. E4M3's 3-bit mantissa gives a 12.5% relative step (≤6.25% rounding error), which is good enough per element — but then its range is too narrow. These 8 bits are doing two jobs — conveying *magnitude* and *detail* — and eight bits simply isn't enough for both across a whole tensor.

The insight behind MXFP8 is that these two jobs don't need the same density: **magnitude only needs to be tracked coarsely — per small neighbourhood — while detail varies per element.** Share the magnitude. Keep the detail private.

---

## Part 2 — MXFP8: Block-Level Scaling

### The core idea

Split a tensor into contiguous blocks of 32 elements along the reduction (K) dimension of the GEMM. Each block gets one shared **scale factor** — an 8-bit power-of-two exponent in E8M0 format: the stored byte *E* encodes 2^(E−127), so scales range from 2⁻¹²⁷ to 2¹²⁷ (0xFF is reserved for NaN). The elements themselves stay in FP8 (E4M3 here).

![Four 32-value blocks along K, each with its own scale byte; one block laid out in memory as 1 scale byte plus 32 data bytes; and bars comparing 64 bytes for BF16, 33 for MXFP8 and 32 for unscaled FP8.](/blog/mxfp8/mxfp8_02_block_scaling.svg "Every 32 consecutive values along K get one E8M0 byte. That single byte is the whole cost of the format: 33 bytes per 32 values, against 64 for BF16.")

The dequantized value for element *i* in block *j* is:

> **x̂ᵢ = xᵢ(FP8) × Xⱼ**

where xᵢ(FP8) is the stored 8-bit E4M3 value and Xⱼ = 2^(Eⱼ−127) is the block's shared scale, decoded from its E8M0 byte Eⱼ.

### Storage cost

The math is stark — and it's the single fact that makes MXFP8 attractive:

- 32 × FP8 E4M3 values = 32 bytes
- 1 × E8M0 scale = 1 byte
- Total = **33 bytes** for 32 elements

Compare: raw FP8 (no scaling) would be 32 bytes for the same 32 elements, and BF16 would be 64 bytes. The scale overhead is just 1 extra byte per 32 elements — a **3.125% storage overhead** over raw FP8, or an effective 8.25 bits per element (1.03 bytes/element).

In relative terms:
- Raw FP8 E4M3: 1.0 bytes/element (baseline)
- MXFP8: 1.03 bytes/element (3.1% overhead)
- BF16: 2.0 bytes/element (1.94× larger than MXFP8)

Overhead is one side of the block-size trade: at block size 8 it jumps to 12.5%; at block size 128 it shrinks to 0.8%, but a single outlier then sets the scale for 128 elements.

### Why E8M0 for the scale?

Most people's first question: *why not just use FP16 or FP32 for the scale?* The answer is threefold:

1. **Power-of-two is cheap at dequant.** Multiplying an FP8 value (or a product of two) by 2ⁿ is an exponent adjustment, not a full floating-point multiply. Hardware loves this.
2. **One byte per block.** An FP16 scale would cost 2 bytes (6.25% overhead), and FP32 would cost 4 bytes (12.5% overhead). E8M0 keeps it to 1 byte per 32 elements — 3.125%.
3. **No sign, no mantissa.** The scale has no sign bit (scales are always positive) and no mantissa (only exact powers of two). Every bit goes into the exponent, maximizing the dynamic range per byte of scale storage.

The smallest scale is 2⁻¹²⁷ ≈ 5.88 × 10⁻³⁹ and the largest is 2¹²⁷ ≈ 1.70 × 10³⁸.

Multiply this with E4M3's native range of [0.00195, 448]:

| Scale | Min effective value | Max effective value |
|-------|-------------------|-------------------|
| 2⁻¹²⁷ | 1.1 × 10⁻⁴¹ | 2.6 × 10⁻³⁶ |
| 2⁰ | 0.00195 | 448 |
| 2¹²⁷ | 3.3 × 10³⁵ | 7.6 × 10⁴⁰ |

Across all possible scales, representable magnitudes run from ~10⁻⁴¹ to ~10⁴¹ — comparable to FP32 (whose normal range is ~10⁻³⁸ to ~3.4 × 10³⁸). But keep the important caveat in view: **within any one block**, the representable range is still only E4M3's ~5.4 decades, shifted up or down by that block's scale. MXFP8 doesn't make a single block wider; it lets each block put its narrow window wherever its own data lives.

![Log-scale range bars for FP32, BF16, E5M2 and E4M3, and an MXFP8 row showing three E4M3-width windows at scales 2^-100, 2^0 and 2^100 inside a dashed envelope spanning about 10^-41 to 10^41.](/blog/mxfp8/mxfp8_03_dynamic_range.svg "The range comes from the scale, the precision from the data. Each block keeps E4M3's narrow window; the scale only decides where on the axis that window sits.")

### The quantization algorithm

For a block *b* of 32 elements:

1. Find **m = max(|b₀|, ..., |b₃₁|)**
2. Compute the scale exponent: **k = ⌈log₂(m / 448)⌉**, clamped to [−127, 127], and set **X = 2ᵏ**
3. Quantize each element: **bᵢ(MXFP8) = quantize_E4M3(bᵢ / X)**

Step 2 picks the smallest power of two that brings the block maximum to ≤ 448, so no element overflows, and the scale is always a power of two. Step 3 is standard E4M3 quantization — round to the nearest representable FP8 value. (An all-zero block just gets the minimum scale.)

This round-up rule is one of two common choices. The OCP spec's reference algorithm instead uses **k = ⌊log₂ m⌋ − 8** (8 being E4M3's largest exponent) and saturates anything that lands above 448 — it can clip the block maximum slightly but keeps up to one extra binade of resolution at the bottom of the block. Libraries expose both; PyTorch's torchao, for example, lets you pick the scale-rounding mode.

The magic: elements in *different* blocks can sit at wildly different magnitudes without interfering, because each block chooses its own scale. Within a block, the elements share the scale set by the block's largest value, so the block's small elements are what pay — but the damage is confined to 31 neighbours instead of the whole tensor.

### A worked numerical example

Here's the quantization process on an actual block (simplified to 8 elements for readability, but the same logic applies to 32):

**Original block (FP32):** `[0.02, 0.1, 0.25, 1.2, 3.8, 45.0, 150.0, 1800.0]`

**Step 1 — Find the absolute maximum:** `m = 1800.0`

**Step 2 — Compute the shared scale.** 1800 / 448 = 4.02, and ⌈log₂ 4.02⌉ = 3, so the scale is 2³ = 8 (stored as E8M0 byte 130 = 0x82). After scaling, the maximum becomes 225 ≤ 448.

**Step 3 — Quantize each element.** Divide by the scale, round to the nearest E4M3 value, multiply back:

| Original | ÷ scale (8) | E4M3 quantized | Dequantized | Error |
|----------|------------|----------------|-------------|-------|
| 0.02 | 0.0025 | 0.001953 (subnormal) | 0.015625 | −21.9% |
| 0.10 | 0.0125 | 0.01172 (subnormal) | 0.09375 | −6.3% |
| 0.25 | 0.03125 | 0.03125 | 0.2500 | 0.0% |
| 1.20 | 0.15 | 0.15625 | 1.2500 | +4.2% |
| 3.80 | 0.475 | 0.46875 | 3.7500 | −1.3% |
| 45.00 | 5.625 | 5.5 | 44.00 | −2.2% |
| 150.00 | 18.75 | 18 | 144.0 | −4.0% |
| 1800.00 | 225 | 224 | 1792 | −0.4% |

**Key observation:** every element with a normal E4M3 encoding stays within E4M3's 6.25% rounding bound. The two smallest values fall into the subnormal range — this block spans five orders of magnitude, right at the edge of what one E4M3 window can hold — and lose precision. That's the within-block cost of sharing a scale.

Now compare per-tensor scaling. Suppose this block lives in a tensor whose global maximum is 60000: the per-tensor scale becomes 2⁸ = 256. Under that scale, 0.02, 0.1 and 0.25 all round to **zero** (−100% error), and 1.2 comes back as 1.0 (−16.7%). Block scaling kept all of them; per-tensor scaling threw them away because of an outlier somewhere else in the tensor.

---

## Part 3 — Hardware: Matrix Cores and MXFP8

### CDNA4 (MI355X): scaled MFMA

AMD's CDNA4 matrix cores consume MXFP8 through the *scaled* MFMA (Matrix Fused Multiply-Add) instructions. The FP8 operands and their E8M0 block scales go into the same instruction; the matrix core forms the FP8 products, applies the combined A-scale × B-scale for each 32-wide K block, and accumulates in FP32.

![FP8 data and E8M0 scales loaded from HBM into per-lane registers, then consumed directly by v_mfma_scale_f32_16x16x128_f8f6f4, which applies 2^(sA+sB) per 32-wide K block and accumulates in FP32.](/blog/mxfp8/mxfp8_04_gemm_dataflow.svg "The scales ride into the matrix instruction as operands. No dequantize step exists, so A and B never take up FP32 or BF16 space in HBM or registers.")

There is no separate dequantization pass and no widened FP32 copy of the operands in HBM or registers — the packed FP8 data and the scale bytes are all the kernel ever moves. Because each scale covers 32 consecutive K elements and the instructions' K dimension is a multiple of 32, every instruction consumes a whole number of scale blocks.

The CDNA4 (gfx950) scaled MFMA instructions:

| Instruction | Output tile | K per instruction | Scale blocks along K | Accumulate |
|-------------|-------------|-------------------|----------------------|------------|
| `v_mfma_scale_f32_16x16x128_f8f6f4` | 16×16 | 128 | 4 | FP32 |
| `v_mfma_scale_f32_32x32x64_f8f6f4` | 32×32 | 64 | 2 | FP32 |

The `f8f6f4` suffix is literal: the same instructions take FP8 (E4M3 or E5M2), FP6 or FP4 operands, with the element format selected per operand — so MXFP8, MXFP6 and MXFP4 all run through one datapath.

### Throughput comparison

MI355X peak dense matrix throughput (256 CUs at 2.4 GHz, per AMD's CDNA4 whitepaper):

| Datatype | FLOPs/clk/CU | Peak dense (MI355X) |
|----------|--------------|---------------------|
| FP32 | 256 | ~157 TFLOPS |
| BF16/FP16 | 4096 | ~2.5 PFLOPS |
| FP8 / MXFP8 | 8192 | **~5 PFLOPS** |
| MXFP6 | 16384 | ~10 PFLOPS |
| MXFP4 | 16384 | ~10 PFLOPS |

MXFP8 doubles BF16 compute throughput. Note that on CDNA4 MXFP6 runs at the *same* rate as MXFP4, not half of it. Real kernels land below these peaks: quantization kernels, scale handling, epilogues and — for small-M or otherwise low-arithmetic-intensity GEMMs — memory bandwidth all eat into the ideal 2×.

### Where MXFP8 fits in the quantization landscape

It helps to place MXFP8 alongside the other reduced-precision formats:

| Format | Scale granularity | Scale type | Block size | Bytes/elem |
|--------|------------------|------------|------------|------------|
| FP8 E4M3 (per-tensor) | Tensor | FP32 | Entire tensor | 1.0 |
| Block FP8 (1×128) | Block | FP32 | 128 | 1.03 |
| **MXFP8 (OCP)** | Block | E8M0 | 32 | 1.03 |
| MXFP6 (OCP) | Block | E8M0 | 32 | 0.78 |
| MXFP4 (OCP) | Block | E8M0 | 32 | 0.53 |
| BF16 | None | — | 1 | 2.0 |

**The key differentiator:** MXFP8 uses E8M0 (power-of-two) scales on 32-element blocks. Block FP8 uses FP32 scales on 128-element blocks — 4× coarser granularity for the *same* storage overhead (4 bytes per 128 elements = 1 byte per 32), though its full-precision scales avoid power-of-two rounding. Per-tensor FP8 is the crudest: one scale for the entire tensor, vulnerable to the single-outlier problem.

![Three 4 by 64 grids with one outlier: under per-tensor scaling all 256 cells are affected, under per-row scaling the outlier's row of 64, and under MXFP8 only the outlier's 32-value block.](/blog/mxfp8/mxfp8_05_scale_locality.svg "The real argument for small blocks: the scale granularity decides how far one outlier's damage spreads. MXFP8 caps it at 32 values.")

---

## Part 4 — Why Block Size 32?

The block size is a choice, and 32 isn't arbitrary — but it isn't a closed-form optimum either. It's the balance point between two costs that move in opposite directions.

### Smaller blocks: better outlier containment, more overhead

A block's scale is set by its largest element, so every other element in the block is quantized relative to that maximum. When an outlier lands in a block, the block's small values get pushed down toward E4M3's subnormal range — exactly what happened to 0.02 and 0.1 in the worked example. The smaller the block, the fewer innocent neighbours share an outlier's scale, and the more likely a block's values fit comfortably inside one ~5.4-decade E4M3 window.

The price is storage and bandwidth: overhead is 8 bits / (8·B bits) = 1/B.

| Block size | Scale overhead | Elements sharing one outlier's scale |
|------------|----------------|--------------------------------------|
| 8 | 12.5% | 8 |
| 16 | 6.25% | 16 |
| **32** | **3.125%** | **32** |
| 64 | 1.56% | 64 |
| 128 | 0.78% | 128 |

### Larger blocks: cheaper, but outliers spread

Going the other way buys little: from 32 to 128, overhead only falls by 2.3 percentage points, while an outlier's reach grows fourfold. The power-of-two scale compounds this — it can waste up to one binade of range relative to an exact scale, a cost that hurts more when one scale must cover more values.

### What settled it

The OCP MX block size came out of empirical work — Rouhani et al., *Microscaling Data Formats for Deep Learning* (2023) — which swept block sizes and element formats across training and inference workloads and found 32 kept accuracy close to FP32 baselines with small overhead. It is also hardware-friendly: 32 FP8 elements are 32 bytes, and the matrix-instruction K dimensions (64, 128) are whole multiples of it.

---

## Part 5 — MXFP6 and MXFP4: The Road Ahead

The MX specification scales down further:

| Format | Element bits | Scale bits | Block size | Bytes/elem | Relative to BF16 |
|--------|-------------|-----------|------------|------------|-----------------|
| BF16 | 16 | — | 1 | 2.0 | 1× |
| FP8 E4M3 | 8 | — | 1 | 1.0 | 2× |
| **MXFP8** | 8 | 8 | 32 | 1.03 | 1.94× |
| MXFP6 | 6 | 8 | 32 | 0.78 | 2.56× |
| MXFP4 | 4 | 8 | 32 | 0.53 | 3.76× |

MXFP6 and MXFP4 trade more precision for more compression. MXFP6 comes in two variants: E2M3 (3 mantissa bits, the same 12.5% relative step as E4M3 but much less range) and E3M2 (2 mantissa bits, a 25% step like E5M2, with per-block scaling supplying the range). MXFP4 uses E2M1 — a single mantissa bit, so adjacent values are 50% apart and rounding error reaches 25% — which is why it is used primarily for inference.

On CDNA4, MXFP6 and MXFP4 run through the same scaled MFMA instructions as MXFP8, at twice the MXFP8 rate, but the real-world training convergence story for these formats is still being written. The quantization error at 4 bits per element starts interacting with optimizer dynamics in non-trivial ways — a topic for another post.

---

## The Mental Model

Here's the intuition to carry with you:

> FP8 gives every value its own exponent.
> Block Floating Point gives every block one exponent.
> **MXFP8 sits in the middle: every value has a local FP8 exponent, while every block gets an additional shared E8M0 scale.**

The per-element E4M3 exponent handles fine-grained magnitude variation within the block. The shared E8M0 scale shifts the entire block's representable window up or down by powers of two. Together they cover FP32-like range across the tensor, at 8-bit storage density, with outlier damage confined to a single 32-element block.

That's the block-scaling bet, and it pays off.

---

## References

- [OCP Microscaling Formats (MX) Specification v1.0](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)
- [Rouhani et al., "Microscaling Data Formats for Deep Learning" (2023)](https://arxiv.org/abs/2310.10537)
- [AMD, "Introducing AMD CDNA 4 Architecture" (whitepaper)](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-4-architecture-whitepaper.pdf)
- [ROCm Composable Kernel MXFP8 GEMM](https://github.com/ROCm/composable_kernel)
- [AITER: Block-Scaled GEMM in ROCm](https://github.com/ROCm/aiter)
- [PyTorch torchao (MX formats)](https://github.com/pytorch/ao)

---

*Feedback? Thoughts? Find me on [LinkedIn](https://www.linkedin.com/in/shekhar-p-aa90249a/) or [GitHub](https://github.com/indianspeedster).*
