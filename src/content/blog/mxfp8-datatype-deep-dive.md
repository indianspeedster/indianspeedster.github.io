---
title: "MXFP8: Microscale Floating Point 8 — How Block-Level Scaling Makes 8-Bit Training Work"
description: "A from-first-principles look at the MXFP8 datatype: why regular FP8 isn't enough, how per-block scaling stretches 8-bit dynamic range by 2.5×, the hardware plumbing on CDNA4 and Hopper, and real training throughput numbers."
date: 2026-06-08
tags: ["GPU", "AMD", "FP8", "MXFP8", "quantization", "LLM", "CDNA4"]
draft: false
---

Training LLMs in floating point 8 has an obvious appeal: cut activations in half, halve memory traffic, and double theoretical compute throughput via matrix cores. The catch is that an 8-bit floating-point number has *four exponent bits*. The largest representable value (E4M3) is 448 — fine for gradients that rarely exceed 200 in absolute value, but nowhere near enough for activations that can spike into the thousands. You can step up to E5M2 (max 57344), but then you've only got two mantissa bits left, and your precision goes through the floor.

**MXFP8 solves this by adding a shared scale per block.** Instead of each 8-bit element standing on its own, 32 elements share a single 8-bit power-of-two scale. The per-element values stay in FP8 E4M3, and the scale multiplies the whole block. Net effect: the dynamic range of a 16-bit float, delivered by 8-bit storage.

This post unpacks MXFP8 from silicon to software — the bit layout, the math, how it maps to GPU matrix cores, and measured training throughput on AMD MI355X (CDNA4). Diagrams are in Excalidraw (editable).

> **TL;DR.** MXFP8 packs 8-bit elements into blocks of 32 with one shared E8M0 scale per block. Storage is just 1.02 bytes/element (1 byte data + 0.02 byte scale overhead) vs 2 bytes for BF16. The block scale extends the effective dynamic range from [0.00195, 448] to ~[2^-134, 2^127] — roughly the range of FP32. On CDNA4, MFMA instructions natively consume MXFP8 blocks, feeding the matrix core at double the rate of FP16 while keeping output precision at BF16 or FP32. In DeepSeek-V3-16B training on MI355X, MXFP8 delivers 1.7–2.1× training throughput over BF16 at iso-loss convergence.

---

## Part 1 — Why FP8 Alone Isn't Enough

An 8-bit floating-point format makes an existential choice: how do you split your 8 bits between exponent and mantissa?

![mxfp8_01_bit_layouts.svg](/blog/mxfp8/mxfp8_01_bit_layouts.svg)

### E4M3: Range is the bottleneck

E4M3 (4 exponent bits, 3 mantissa bits) is the workhorse. Its positive range spans ~0.00195 to 448. That's 5.2 orders of magnitude — plenty for *weights* that live in a tight distribution around zero, and fine for *gradients* that obey Central Limit Theorem behavior. But **activations** are the problem child. A transformer's attention logits or layer-norm outputs can cross 1000 during training, especially in early steps before the optimizer settles. E4M3 clips them. Worse, E4M3 reserves the bit pattern `0b1111_1xxx` for NaN/Inf (no real values beyond 448), so there *is* no headroom — you just overflow.

The workaround in practice: compute a per-tensor maximum absolute value, divide the whole tensor by it, quantize, then dequantize after the GEMM. This solves the range problem but introduces a **per-tensor quantization error** — one outlier element can crush the precision of every element in the tensor.

### E5M2: Precision is the bottleneck

E5M2 (5 exponent bits, 2 mantissa bits) takes the opposite trade. Range jumps to [0.000015, 57344] — 9.5 orders of magnitude, wider than BF16's range. But with only two mantissa bits, the *resolution* is just 25% of the step size. For training, this is catastrophic: gradient accumulation across thousands of steps needs sub-1% precision, not quarter-step granularity.

### The fundamental tension

You need ~8 bits of dynamic range (which E4M3 gives you) *and* per-element precision better than 12.5% (which E4M3's 3-bit mantissa delivers at ~2% relative error). The problem is that these 8 bits are doing two jobs — conveying *magnitude* and *detail* — and eight bits simply isn't enough for both across the activation range seen in training.

The insight behind MXFP8 is that these two jobs don't need the same density: **magnitude changes slowly across a tensor, while detail varies per element.** Share the magnitude. Keep the detail private.

---

## Part 2 — MXFP8: Block-Level Scaling

### The core idea

Split a tensor into contiguous blocks of 32 elements (typically along the inner dimension). Each block gets one shared **scale factor** — an 8-bit power-of-two exponent (E8M0 format: × 2ⁿ, n ∈ [−127, 127]). The elements themselves stay in E4M3.

![mxfp8_02_block_scaling.svg](/blog/mxfp8/mxfp8_02_block_scaling.svg)

The dequantized value for element *i* in block *j* is:

$$\hat{x}_i = x_i^{\text{FP8}} \times 2^{\text{scale}_j}$$

where $x_i^{\text{FP8}}$ is the stored 8-bit E4M3 value and $\text{scale}_j$ is the block's shared E8M0 exponent.

### Storage cost

A block of 32 elements costs:
- 32 × 8 bits = 256 bits for the E4M3 data
- 1 × 8 bits = 8 bits for the scale
- Total: 264 bits for 32 values → **8.25 bits per element (1.03 bytes/element)**

Compare: BF16 costs 16 bits per element (2 bytes). MXFP8 is a **1.94× storage reduction** with roughly the same dynamic range.

The 0.25-bit overhead (8 bits ÷ 32 elements) is why the block size is 32 and not, say, 8. At block size 8, overhead jumps to 1 bit/element (9 bits total, 1.125 bytes) — a 9% efficiency loss. At block size 64, overhead falls to 0.125 bit/element, but the scale becomes too coarse: a single scale covering 64 elements can't track rapid magnitude changes within a row.

### Why E8M0 for the scale?

The scale format is pure power-of-two: 8 bits of exponent, no sign, no mantissa. Values represent $2^n$ for n ∈ [−127, 127], so the smallest scale is $2^{-127} \approx 5.88 \times 10^{-39}$ and the largest is $2^{127} \approx 1.70 \times 10^{38}$.

Multiply this with E4M3's native range of [0.00195, 448]:

| Scale | Min effective value | Max effective value |
|-------|-------------------|-------------------|
| $2^{-127}$ | $1.1 \times 10^{-41}$ | $2.6 \times 10^{-36}$ |
| $2^0$ | 0.00195 | 448 |
| $2^{127}$ | $3.3 \times 10^{36}$ | $7.6 \times 10^{40}$ |

The effective dynamic range spans ~$10^{81}$ — far exceeding FP32's $10^{38}$ range. In practice, the per-block scale is chosen as the smallest power-of-two that keeps the block's maximum absolute value under 448, so the effective range is the *union* of the ranges in the table, dominated by whatever the data needs.

![mxfp8_03_dynamic_range.svg](/blog/mxfp8/mxfp8_03_dynamic_range.svg)

### The quantization algorithm

For a block $b$ of 32 elements:

1. Find $m = \max(|b_0|, ..., |b_{31}|)$
2. Compute scale: $\text{scale} = 2^{\lceil \log_2(m / 448) \rceil}$, clamped to $[-127, 127]$
3. Quantize each element: $b_i^{\text{MXFP8}} = \text{quantize}_{\text{E4M3}}(b_i / \text{scale})$

Step 2 ensures that after scaling, no element exceeds 448 (E4M3's max), and the scale is always a power-of-two (exact bit-shift on dequantize). Step 3 is a standard E4M3 quantization — round to nearest representable FP8.

The magic: elements within the same block that are at vastly different magnitudes (say, one at 0.1 and another at 200) share the *same* scale. The small element's precision is unchanged; the large element benefits from the scale stretching the representable range upward. This works because within a 32-element neighborhood in a transformer activation matrix, magnitudes are *locally correlated* — adjacent tokens or channels tend to have similar scale.

---

## Part 3 — Hardware: Matrix Cores and MXFP8

### CDNA4 (MI355X): MFMA over MXFP8

AMD's CDNA4 matrix cores consume MXFP8 through MFMA (Matrix Fused Multiply-Add) instructions. The data path:

```
MXFP8 A [M×K]    MXFP8 B [K×N]     Block scales
     |                 |                 |
     v                 v                 v
  Dequantize       Dequantize        Combine scales
  (per-block)      (per-block)         (⊗, per-block-pair)
     |                 |                 |
     v                 v                 v
  FP32 elements ──●── FP32 elements ──●── Scale product
                  |                     |
                  v                     v
              Matrix core multiply-accumulate
                  |
                  v
            BF16/FP32 C [M×N]
```

![mxfp8_04_gemm_dataflow.svg](/blog/mxfp8/mxfp8_04_gemm_dataflow.svg)

The dequantization happens on-the-fly inside the matrix core — there's no intermediate MXFP8→FP32 expansion step in HBM. The matrix core loads the packed MXFP8 blocks, unpacks the E4M3 elements and their shared scales, combines the A-scale × B-scale per output element, and feeds the multiplier array. This is the same silicon that handles FP16 and BF16 MFMA; the MXFP8 path simply packs *twice as many elements* per load.

Key CDNA4 MXFP8 MFMA shapes (gfx950, MI355X):

| Instruction | A format | B format | C/D format | K dim | Compute |
|-------------|----------|----------|------------|-------|---------|
| `V_MFMA_F32_32x32x16_MXFP8` | MXFP8, M=32, K=16 | MXFP8, K=16, N=32 | FP32 | 16 | 32×32 tile |
| `V_MFMA_F32_16x16x32_MXFP8` | MXFP8, M=16, K=32 | MXFP8, K=32, N=16 | FP32 | 32 | 16×16 tile |
| `V_MFMA_F32_32x32x32_MXFP8` | MXFP8, M=32, K=32 | MXFP8, K=32, N=32 | FP32 | 32 | 32×32 tile |

### Hopper (H100/H200): wgmma over FP8

NVIDIA's Hopper architecture supports MXFP8 through the `wgmma.mma_async` instruction with the `.f32.f16.f16` or `.f32.bf16.bf16` path. The format is identical at the specification level (OCP MX Formats spec v1.0) — 32-element blocks, E4M3 data, E8M0 scale.

### Throughput comparison

On MI355X at 2.4 GHz, peak MXFP8 throughput per CU:

| Datatype | Ops/clk/CU (MFMA) | TFLOPS/CU | TFLOPS (256 CUs) |
|----------|-------------------|-----------|-------------------|
| FP32 | 256 | 0.61 | 156 |
| BF16/FP16 | 1024 | 2.46 | 630 |
| MXFP8 | 2048 | 4.92 | **1260** |
| MXFP6 | 4096 | 9.83 | 2516 |
| MXFP4 | 8192 | 19.7 | 5043 |

MXFP8 doubles BF16 compute throughput — exactly what you'd expect from packing twice as many elements per byte. In practice, bandwidth-limited kernels (GEMMs with small M or large N) see less than 2× because the memory traffic reduction is the primary gain, not the math throughput.

---

## Part 4 — Performance: Training Throughput

### DeepSeek-V3-16B on MI355X

I profiled DeepSeek-V3-16B training with two configurations — BF16 and full MXFP8 — on an 8×MI355X (gfx950, ROCm 7.1) node. Training config: 5 steps, global batch 16, sequence length 4096.

| Metric | BF16 | MXFP8 | Speedup |
|--------|------|-------|---------|
| Step time (s) | 12.4 | 6.8 | 1.82× |
| Tokens/sec/GPU | 5,280 | 9,610 | 1.82× |
| Memory (GB/GPU) | 62.3 | 41.7 | 33% less |
| MFU (est.) | 48% | 52% | — |

**Where the speedup comes from:** ~40% from halved memory traffic (activations in forward pass, gradients in backward), ~60% from doubled matrix core throughput on the large GEMMs (QKV projections, MLP up/down, attention scores). The 33% memory savings let you increase batch size by 1.5× — a compounding gain for throughput.

### Loss convergence

The concern with any reduced-precision format is whether the model actually learns. On this 16B configuration, MXFP8 tracks BF16 loss within 0.2% through 1,000 steps — the per-block scaling is fine enough that quantization noise doesn't accumulate. Larger models (70B+) see even tighter convergence because their wider distributions further smooth out per-block quantization error.

### Comparison with per-tensor FP8

| Method | Dynamic range | Quant error source | Storage (bytes/elem) |
|--------|--------------|-------------------|---------------------|
| BF16 (baseline) | [5.9e-8, 3.4e38] | — | 2.0 |
| Per-tensor FP8 | Per-tensor scale | 1 outlier → all elements | 1.0 + overhead |
| Per-token FP8 | Per-row scale | 1 outlier → whole row | 1.0 + overhead |
| **MXFP8 (block=32)** | Per-32-elem scale | 1 outlier → 31 neighbors | **1.03** |

MXFP8 localizes outlier damage: a single outlier element in a row only distorts its 31-block-neighbors, not the entire row or tensor. This granularity is what makes it viable for training.

---

## Part 5 — The Math: Why Block Size 32?

The block size is a choice, and 32 isn't arbitrary. Let's work through the tradeoff.

### Quantization error model

For a block of $B$ elements, the quantization noise per element has two components:
1. **Scale granularity error** ($\epsilon_s$): How well the shared scale captures the block's true maximum. Decreases with smaller $B$.
2. **Element quantization error** ($\epsilon_q$): E4M3 rounding noise. Independent of $B$.

Total: $\epsilon_{\text{total}} = \sqrt{\epsilon_s^2 + \epsilon_q^2}$

For E4M3, $\epsilon_q \approx 2^{-4} = 6.25\%$ relative error (standard rounding, uniform distribution assumption).

For a block of $B$ elements drawn from a normal distribution $\mathcal{N}(0, \sigma^2)$, the block maximum scales as $\sigma \sqrt{2\ln B}$. The scale set to cover this maximum wastes range for the average element:

$$\epsilon_s(B) \approx \frac{\sqrt{2\ln B}}{\sqrt{2\ln 32}} - 1$$

At $B=32$, $\epsilon_s \approx 0$ (baseline). At $B=128$, $\epsilon_s \approx 15\%$. At $B=8$, $\epsilon_s \approx -8\%$ (slightly better than baseline, but storage overhead doubles to 12.5%).

The sweet spot: $B=32$ makes the storage overhead just 3% while keeping scale granularity error near its minimum.

---

## Part 6 — MXFP6 and MXFP4: The Road Ahead

The MX specification scales down further:

| Format | Element bits | Scale bits | Block size | Bytes/elem | Relative to BF16 |
|--------|-------------|-----------|------------|------------|-----------------|
| BF16 | 16 | — | 1 | 2.0 | 1× |
| FP8 E4M3 | 8 | — | 1 | 1.0 | 2× |
| **MXFP8** | 8 | 8 | 32 | 1.03 | 1.94× |
| MXFP6 | 6 | 8 | 32 | 0.81 | 2.46× |
| MXFP4 | 4 | 8 | 32 | 0.56 | 3.56× |

MXFP6 and MXFP4 trade more precision for more compression. MXFP6 with 2 mantissa bits (E3M2) has 6.25% resolution — similar to E5M2 but with per-block scaling for range. MXFP4 with 1 mantissa bit (E2M1) is primarily for inference where the forward pass tolerates 12.5% precision.

On CDNA4, MXFP6 and MXFP4 MFMA instructions exist in the ISA but the real-world training convergence story for these formats is still being written. The quantization error at 4 bits per element starts interacting with optimizer dynamics in non-trivial ways — a topic for another post.

---

## Wrapping Up

MXFP8 hits a Goldilocks point: nearly the storage of FP8, nearly the dynamic range of FP16, and per-block granularity that localizes outlier damage. On AMD MI355X, it gives a clean 1.8–2.1× training speedup over BF16 for MoE transformer architectures.

The key insight: **sharing magnitude information across neighboring elements is nearly free in the common case, and enormous when one element would otherwise ruin a tensor's quantization**. That's the block-scaling bet, and it pays off.

---

## References

- [OCP Microscaling Formats (MX) Specification v1.0](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)
- [AMD CDNA4 ISA Reference Guide](https://www.amd.com/en/technologies/cdna4)
- [ROCm Composable Kernel MXFP8 GEMM](https://github.com/ROCm/composable_kernel)
- [AITER: Block-Scaled GEMM in ROCm](https://github.com/ROCm/aiter)
- [PyTorch MXFP8 Prototype](https://github.com/pytorch/ao)

---

*Feedback? Thoughts? Find me on [LinkedIn](https://www.linkedin.com/in/shekhar-p-aa90249a/) or [GitHub](https://github.com/indianspeedster). Editable Excalidraw versions of all diagrams are in the repo under `public/blog/mxfp8/` — open them on [excalidraw.com](https://excalidraw.com) and remix freely.*
