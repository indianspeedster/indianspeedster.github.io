---
title: "MXFP8: Microscale Floating Point 8 — How Block-Level Scaling Makes 8-Bit Training Work"
description: "A from-first-principles look at the MXFP8 datatype: why regular FP8 isn't enough, how per-block scaling stretches 8-bit dynamic range by 2.5×, the hardware plumbing on CDNA4 and Hopper, and the block-size theory behind the design."
date: 2026-06-08
tags: ["GPU", "AMD", "FP8", "MXFP8", "quantization", "LLM", "CDNA4"]
draft: false
---

Training LLMs in floating point 8 has an obvious appeal: cut activations in half, halve memory traffic, and double theoretical compute throughput via matrix cores. The catch is that an 8-bit floating-point number has *four exponent bits*. The largest representable value (E4M3) is 448 — fine for gradients that rarely exceed 200 in absolute value, but nowhere near enough for activations that can spike into the thousands. You can step up to E5M2 (max 57344), but then you've only got two mantissa bits left, and your precision goes through the floor.

**MXFP8 solves this by adding a shared scale per block.** Instead of each 8-bit element standing on its own, 32 elements share a single 8-bit power-of-two scale. The per-element values stay in FP8 E4M3, and the scale multiplies the whole block. Net effect: the dynamic range of a 16-bit float, delivered by 8-bit storage.

> **In one sentence:** MXFP8 stores FP8 values in blocks of 32 elements, where each block shares a power-of-two scaling factor encoded as an 8-bit E8M0 exponent. Every stored element is an E4M3 FP8 number; the block scale multiplies the entire block to recover the original magnitude.

This post unpacks MXFP8 from silicon to software — the bit layout, the math, how it maps to GPU matrix cores on AMD MI355X (CDNA4) and NVIDIA Hopper, and the block-size theory behind the design. Diagrams are in Excalidraw (editable).

> **TL;DR.** MXFP8 packs 8-bit elements into blocks of 32 with one shared E8M0 scale per block. Storage is just 1.02 bytes/element (1 byte data + 0.02 byte scale overhead) vs 2 bytes for BF16. The block scale extends the effective dynamic range from [0.00195, 448] to ~[2⁻¹³⁴, 2¹²⁷] — roughly the range of FP32. On CDNA4, MFMA instructions natively consume MXFP8 blocks, feeding the matrix core at double the rate of FP16 while keeping output precision at BF16 or FP32. It's a ~2× memory and compute win for transformer training and inference.

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

Split a tensor into contiguous blocks of 32 elements (typically along the inner dimension). Each block gets one shared **scale factor** — an 8-bit power-of-two exponent (E8M0 format: × 2ⁿ, n ranges from −127 to 127). The elements themselves stay in E4M3.

![mxfp8_02_block_scaling.svg](/blog/mxfp8/mxfp8_02_block_scaling.svg)

The dequantized value for element *i* in block *j* is:

> **x̂ᵢ = xᵢ(FP8) × 2^scaleⱼ**

where xᵢ(FP8) is the stored 8-bit E4M3 value and scaleⱼ is the block's shared E8M0 exponent.

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

The overhead is why the block size is 32 — at block size 8, overhead jumps to 12.5%; at block size 128, it shrinks to 0.8% but the scale becomes too coarse to track rapid magnitude changes within a row.

### Why E8M0 for the scale?

Most people's first question: *why not just use FP16 or FP32 for the scale?* The answer is threefold:

1. **Power-of-two is free at dequant.** An E8M0 scale represents 2ⁿ for n ∈ [−127, 127]. Multiplying an E4M3 value by 2ⁿ is a single integer exponent addition — no floating-point multiply needed. Hardware loves this.
2. **One byte per block.** An FP16 scale would cost 2 bytes (6.25% overhead), and FP32 would cost 4 bytes (12.5% overhead). E8M0 keeps it to 1 byte — critical for staying under 2% effective overhead.
3. **No sign, no mantissa.** The scale has no sign bit (values are always positive) and no mantissa (only exact powers of two). Every bit goes into the exponent, maximizing the dynamic range per byte of scale storage.

Values represent 2ⁿ for n ∈ [−127, 127], so the smallest scale is 2⁻¹²⁷ ≈ 5.88 × 10⁻³⁹ and the largest is 2¹²⁷ ≈ 1.70 × 10³⁸.

Multiply this with E4M3's native range of [0.00195, 448]:

| Scale | Min effective value | Max effective value |
|-------|-------------------|-------------------|
| 2⁻¹²⁷ | 1.1 × 10⁻⁴¹ | 2.6 × 10⁻³⁶ |
| 2⁰ | 0.00195 | 448 |
| 2¹²⁷ | 3.3 × 10³⁶ | 7.6 × 10⁴⁰ |

The effective dynamic range spans ~10⁸¹ — far exceeding FP32's 10³⁸ range. In practice, the per-block scale is chosen as the smallest power-of-two that keeps the block's maximum absolute value under 448, so the effective range is the *union* of the ranges in the table, dominated by whatever the data needs.

![mxfp8_03_dynamic_range.svg](/blog/mxfp8/mxfp8_03_dynamic_range.svg)

### The quantization algorithm

For a block *b* of 32 elements:

1. Find **m = max(|b₀|, ..., |b₃₁|)**
2. Compute scale: **scale = 2^⌈log₂(m / 448)⌉**, clamped to [−127, 127]
3. Quantize each element: **bᵢ(MXFP8) = quantize_E4M3(bᵢ / scale)**

Step 2 ensures that after scaling, no element exceeds 448 (E4M3's max), and the scale is always a power-of-two (exact bit-shift on dequantize). Step 3 is a standard E4M3 quantization — round to nearest representable FP8.

The magic: elements within the same block that are at vastly different magnitudes (say, one at 0.1 and another at 200) share the *same* scale. The small element's precision is unchanged; the large element benefits from the scale stretching the representable range upward. This works because within a 32-element neighborhood in a transformer activation matrix, magnitudes are *locally correlated* — adjacent tokens or channels tend to have similar scale.

### A worked numerical example

Here's the quantization process on an actual block (simplified to 8 elements for readability, but the same logic scales to 32):

**Original block (FP32):** `[0.1, 0.25, 0.5, 1.2, 3.8, 12.0, 45.0, 150.0]`

**Step 1 — Find the absolute maximum:** `m = max(|0.1|, ..., |150.0|) = 150.0`

**Step 2 — Compute the shared scale.** The scale must bring 150.0 down to ≤ 448 (E4M3's max). Since 150 < 448 already, the scale is 2⁰ = 1.0. If the maximum were, say, 3600, the scale would be 2^⌈log₂(3600/448)⌉ = 2³ = 8.0.

**Step 3 — Quantize each element.** Divide by the scale, then round to the nearest E4M3 representable value:

| Original | ÷ scale (1.0) | E4M3 quantized | Dequantized | Error |
|----------|--------------|----------------|-------------|-------|
| 0.10 | 0.10 | 0.1016 | 0.1016 | +1.6% |
| 0.25 | 0.25 | 0.2500 | 0.2500 | 0.0% |
| 0.50 | 0.50 | 0.5000 | 0.5000 | 0.0% |
| 1.20 | 1.20 | 1.1875 | 1.1875 | −1.0% |
| 3.80 | 3.80 | 3.7500 | 3.7500 | −1.3% |
| 12.00 | 12.00 | 12.000 | 12.000 | 0.0% |
| 45.00 | 45.00 | 44.000 | 44.000 | −2.2% |
| 150.00 | 150.00 | 144.000 | 144.000 | −4.0% |

**Key observation:** The small values (0.1, 0.25) preserve their E4M3 precision perfectly — the scale isn't dominated by the large outlier at 150. The worst error is 4% at the largest element, well within E4M3's typical quantization bound. This is the localization property: 150's magnitude doesn't crush 0.1's precision because they share a scale that properly covers both.

> 💡 If this were per-tensor scaling, the maximum across the *entire tensor* might be 5000, forcing scale = 2⁴ = 16. Every element would be divided by 16, making 0.1 → 0.00625 — below E4M3's minimum normal (0.00195) and losing several bits of precision. Block scaling prevents this.

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

### Where MXFP8 fits in the quantization landscape

It helps to place MXFP8 alongside the other reduced-precision formats you'll encounter:

| Format | Scale granularity | Scale type | Block size | Bytes/elem |
|--------|------------------|------------|------------|------------|
| FP8 E4M3 (per-tensor) | Tensor | FP32 | Entire tensor | 1.0 |
| Block FP8 | Block | FP32 | 128 | 1.03 |
| **MXFP8 (OCP)** | Block | E8M0 | 32 | 1.03 |
| MXFP6 (OCP) | Block | E8M0 | 32 | 0.81 |
| MXFP4 (OCP) | Block | E8M0 | 32 | 0.56 |
| BF16 | None | — | 1 | 2.0 |

**The key differentiator:** MXFP8 uses E8M0 (power-of-two) scales on 32-element blocks. Block FP8 uses FP32 scales on 128-element blocks — 4× coarser granularity and 4× more scale storage. Per-tensor FP8 is the crudest: one scale for the entire tensor, vulnerable to the single-outlier problem.

![mxfp8_05_scale_locality.svg](/blog/mxfp8/mxfp8_05_scale_locality.svg)

---

## Part 4 — The Math: Why Block Size 32?

The block size is a choice, and 32 isn't arbitrary. Let's work through the tradeoff.

### Quantization error model

For a block of **B** elements, the quantization noise per element has two components:
1. **Scale granularity error (εₛ):** How well the shared scale captures the block's true maximum. Decreases with smaller B.
2. **Element quantization error (ε_q):** E4M3 rounding noise. Independent of B.

Total: **ε_total = √(εₛ² + ε_q²)**

For E4M3, ε_q ≈ 2⁻⁴ = 6.25% relative error (standard rounding, uniform distribution assumption).

For a block of B elements drawn from a normal distribution N(0, σ²), the block maximum scales as σ·√(2·ln B). The scale set to cover this maximum wastes range for the average element:

> **εₛ(B) ≈ √(2·ln B) / √(2·ln 32) − 1**

At B=32, εₛ ≈ 0 (baseline). At B=128, εₛ ≈ 15%. At B=8, εₛ ≈ −8% (slightly better than baseline, but storage overhead doubles to 12.5%).

The sweet spot: B=32 makes the storage overhead just 3% while keeping scale granularity error near its minimum.

---

## Part 5 — MXFP6 and MXFP4: The Road Ahead

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

## Why MI355X Kernel Engineers Should Care

If you write GEMM kernels for CDNA4 — especially grouped GEMMs for MoE — MXFP8 is becoming unavoidable. Here's what changes in your kernel:

### 1. MXFP8 is becoming the common currency

Models quantized to MXFP8 show up everywhere: vLLM and SGLang inference, torchao training, and Primus-Turbo pre-training. If your kernel doesn't consume MXFP8 blocks, you're leaving 2× throughput on the table.

### 2. Scale movement is a first-order performance concern

A grouped GEMM with 64 experts, each processing a different token batch, must move not just the FP8 data but also the per-block E8M0 scales from HBM into registers. At 3% overhead per matrix, this sounds trivial — but in a fused MoE kernel where the scale feeds the MFMA instruction as an operand, scale layout in memory determines whether you get coalesced loads or scattered 1-byte reads. Get this wrong and the scale-load latency dominates the kernel.

### 3. Scale layout is table stakes for GEMM kernel design

On CDNA4, the MFMA MXFP8 instructions expect scales interleaved with the data in a specific block-major layout. The hardware loads the 32-element FP8 block and its 1-byte E8M0 scale together — but only if the memory layout matches. A kernel that naive-interprets MXFP8 data as plain FP8 will compute garbage (the scale byte lands in the data stream).

### 4. The register budget shifts

When you switch from BF16 to MXFP8, your operand size halves — but you now need a register to hold the combined A-scale × B-scale product per output tile. This is a small constant overhead (one extra register per tile), but it shifts the VGPR budget slightly. For tiles at the edge of occupancy limits (e.g., 256 VGPRs/wave), this one register can tip you into a lower occupancy tier.

> **Bottom line for kernel authors:** MXFP8 isn't just "FP8 with extra bytes." The scales are first-class operands that affect memory layout, load patterns, and register allocation. Factor them into your kernel design from day one.

---

## The Mental Model

Here's the intuition to carry with you:

> FP8 gives every value its own exponent.
> Block Floating Point gives every block one exponent.
> **MXFP8 sits in the middle: every value has a local FP8 exponent, while every block gets an additional shared E8M0 scale.**

The per-element E4M3 exponent handles fine-grained magnitude variation within the block. The shared E8M0 scale shifts the entire block's representable range up or down by powers of two. Together they cover the full FP32 range, at 8-bit storage density, with outlier damage confined to 31-element neighborhoods.

That's the block-scaling bet, and it pays off.

---

## References

- [OCP Microscaling Formats (MX) Specification v1.0](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)
- [AMD CDNA4 ISA Reference Guide](https://www.amd.com/en/technologies/cdna4)
- [ROCm Composable Kernel MXFP8 GEMM](https://github.com/ROCm/composable_kernel)
- [AITER: Block-Scaled GEMM in ROCm](https://github.com/ROCm/aiter)
- [PyTorch MXFP8 Prototype](https://github.com/pytorch/ao)

---

*Feedback? Thoughts? Find me on [LinkedIn](https://www.linkedin.com/in/shekhar-p-aa90249a/) or [GitHub](https://github.com/indianspeedster). Editable Excalidraw versions of all diagrams are in the repo under `public/blog/mxfp8/` — open them on [excalidraw.com](https://excalidraw.com) and remix freely.*
