---
title: "Occupancy Math on the AMD MI355X (CDNA4): A From-First-Principles Guide"
description: "A from-first-principles guide to wavefront occupancy on AMD's MI355X (CDNA4): the hardware resource budget, the three limiters that cap it, worked MXFP8 grouped-GEMM examples, and why peak throughput often lives at low occupancy."
date: 2026-05-31
tags: ["GPU", "AMD", "CDNA4", "kernels", "occupancy"]
draft: false
---

Ask a GPU kernel engineer how their kernel is doing and *occupancy* comes up within a sentence or two. It's the number everyone quotes and the dial everyone reaches for — and, in my experience, the metric people understand least. Most treat it as an opaque percentage the profiler hands back. It isn't. Occupancy is fully derivable by hand from a kernel's resource usage and a handful of fixed hardware limits, and being able to do that derivation changes how you tune.

This is a from-first-principles guide to occupancy math on the AMD Instinct MI355X (CDNA4, gfx950). We'll build it from the silicon up: what the hardware budget actually is, which three resources cap how many wavefronts can go resident on a SIMD, and how to compute a kernel's occupancy ceiling on paper and then confirm it with rocprofv3. The worked examples lean on MXFP8 grouped GEMM tiles — the kind of MoE kernel where these numbers decide whether you reach peak.

The post is in three parts:

- **Part 1 — The MI355X architecture.** The CU, the four SIMDs, the wavefront, the Matrix Core, and the memory hierarchy that feeds them — the resources the math counts.
- **Part 2 — Occupancy math.** The definition, the three limiters (VGPRs, SGPRs, LDS), the per-SIMD vs per-CU split that trips everyone up, worked examples, and how to measure occupancy for real.
- **Part 3 — Better performance at lower occupancy.** The twist at the end: once you can compute the ceiling, why reaching for it is often the wrong move. Little's Law, ILP versus occupancy, and a tile sweep where occupancy drops while throughput climbs.

Occupancy is worth understanding precisely — both so you can fix the kernels that are genuinely occupancy-limited, and so you can recognize the ones that aren't.

---

## Part 1 — The MI355X architecture

Occupancy is, start to finish, a story about how a fixed pool of resources gets divided among the work you launch. So before any of the math lands, you need a clear mental model of the chip — and in particular, *which resources are private and which are shared*. We'll build it top-down and keep returning to that one distinction, because it's the hinge the whole calculation turns on.

![mi355x_01_chip_hierarchy.svg](/blog/occupancy-mi355x/mi355x_01_chip_hierarchy.svg)

### The chip, top down

The MI355X is CDNA4, ISA target gfx950. At the top level it's eight Accelerator Complex Dies (XCDs) stitched together over fourth-gen Infinity Fabric, totaling 256 Compute Units — 32 per XCD — clocked up to 2.4 GHz. Wrapped around the dies are 288 GB of HBM3E at 8 TB/s, fronted by 256 MB of last-level Infinity Cache. Each XCD carries its own L2 slice, which is why scheduling that keeps a tile's traffic on one die (XCD-aware swizzling) pays off — but that's a locality story, not an occupancy one.

For occupancy, the only unit you reason about is the Compute Unit. Everything above it governs how data moves; occupancy is decided one CU at a time. So that's where we zoom in.

### Inside a Compute Unit

A CU is four SIMD units plus shared infrastructure. Each SIMD is 64 lanes wide and owns a private register file. The pieces that matter:

- **VGPRs — 512 per SIMD, private.** The vector register file, allocated per wave. This is usually the resource that caps occupancy.
- **AccVGPRs — a separate accumulator pool, also private to the SIMD.** MFMA instructions accumulate here. Because it's separate from the VGPR file, spending it doesn't directly evict your general-purpose registers — a fact that matters enormously in Part 3.
- **SGPRs — ~800 per SIMD.** Scalar registers, allocated per wave. Rarely the binding limit.
- **LDS — 160 KB, shared across all four SIMDs.** One physical scratchpad per CU. It's the cooperation mechanism for a workgroup, and on CDNA4 it's 2.5× the size of CDNA3's 64 KB.

Here's the distinction to burn in: **the register files are per-SIMD; the LDS is per-CU.** A wave that lands on SIMD 0 cannot touch SIMD 1's registers, but every wave in a workgroup — wherever it lands — sees the same LDS. That single asymmetry is why the three occupancy limiters in Part 2 don't share a denominator.

![mi355x_02_cu_anatomy.svg](/blog/occupancy-mi355x/mi355x_02_cu_anatomy.svg)

### Threads, lanes, and wavefronts

The hardware doesn't execute threads one at a time. It executes **wavefronts**: bundles of 64 threads that run in lockstep on a SIMD's 64 lanes. AMD's wavefront is exactly NVIDIA's warp, just 64 wide instead of 32 — and "wave" and "wavefront" are the same thing. A 256-thread workgroup is therefore four waves, not 256 independent units of scheduling.

A SIMD holds up to **8 wavefronts resident** at once. Resident means their registers are live and reserved; only one wave issues per cycle, and the scheduler hides latency by switching to a different ready wave whenever the current one stalls. Occupancy is just the ratio of resident waves to that maximum of 8 — counted per SIMD, capped at 32 per CU. Crucially, occupancy says nothing about whether those waves are doing useful work; it only counts how many are parked. Hold that thought for Part 3.

![mi355x_03_wavefront_model.svg](/blog/occupancy-mi355x/mi355x_03_wavefront_model.svg)

### The Matrix Core

The reason any of this exists on an AI accelerator is the Matrix Core — the MFMA engine. CDNA4 overhauled it with native FP8, FP6, and FP4 support and roughly doubled per-CU matrix throughput versus CDNA3. The MI355X peaks at about 5 PFLOPs of MXFP8 and 10 PFLOPs of MXFP6/FP4 dense matrix throughput, with structured sparsity pushing FP4 past 20 PFLOPs. The instruction you'll meet in MoE GEMM kernels is `v_mfma_scale_f32_16x16x128_f8f6f4`, which folds per-block microscaling directly into the matrix op.

The mechanical detail that connects back to occupancy: MFMA reads its operands from the VGPR file and accumulates into AccVGPRs. The matrix engine is fed by registers — nothing slower can keep it busy at peak. That's exactly the tension the memory hierarchy makes concrete.

### The memory hierarchy

From fastest and smallest to slowest and largest: the register file (VGPR/AccVGPR, per-SIMD) → LDS (160 KB/CU) → L1 → L2 (per XCD) → 256 MB Infinity Cache → 288 GB of HBM3E at 8 TB/s. Each step down trades bandwidth for capacity.

The gap that matters is the very first one. Only the register file delivers operands fast enough to sustain the Matrix Core at peak; LDS, despite being on-chip and despite CDNA4's generous 160 KB, is meaningfully slower and prone to bank conflicts. Keeping the hot accumulation register-resident — not in LDS — is what lets a kernel hit the matrix peak, and it's the seed of the argument that closes the post.

![mi355x_04_memory_hierarchy.svg](/blog/occupancy-mi355x/mi355x_04_memory_hierarchy.svg)

### CDNA3 → CDNA4, in one box

What actually changed for kernel authors moving from MI300X/MI325X to MI355X:

- **LDS: 64 KB → 160 KB per CU.** The single biggest occupancy-relevant change; LDS-bound tiles get dramatically more headroom.
- **Matrix Core: ~2× per-CU throughput**, plus native FP6/FP4 (FP6 runs at FP4 rates).
- **VGPRs: unchanged at 512/SIMD.** Despite a common assumption, CDNA4 did *not* double the vector register file. If you've heard otherwise, that's the myth to drop before doing the math.

With the resources and their boundaries in hand, we can finally count them. That's Part 2.

---

## Part 2 — Occupancy math

Occupancy has a one-line definition: it's the number of wavefronts resident on a SIMD divided by the maximum that SIMD can hold (8). Report it per SIMD or scale it to the CU (max 32 waves) — same ratio either way. The number you get is set by whichever resource runs out first, and there are four candidates.

### The limiters

Each limiter has the same shape — a fixed hardware budget divided by what one unit of your kernel consumes, floored to a whole number — but they count different things:

- **VGPRs** hold every per-thread value that's live at once: the accumulator tile, loaded operands, loop variables. Bigger tiles and deeper unrolling cost more.
- **SGPRs** hold values uniform across a wave — base addresses, loop bounds, predicates. Usually cheap.
- **LDS** holds the workgroup's shared staging buffers: the tiles you cooperatively load before feeding the matrix core.
- **Workgroup slots and barriers** are spent once per resident workgroup, no matter how large or small it is.

As formulas:

```
VGPR limit:  floor( 512    / VGPRs-per-thread   )  ->  waves / SIMD
SGPR limit:  floor( ~800   / SGPRs-per-wave     )  ->  waves / SIMD
LDS  limit:  floor( 160 KB / LDS-per-workgroup  )  ->  workgroups / CU
WG   limit:  max workgroups resident / CU          (hard cap + barrier slots)
```

Your occupancy is the minimum of the four, then clamped by the hardware caps (8 waves/SIMD, 32 waves/CU). In practice VGPRs or LDS bind; SGPRs almost never do unless you have heavy scalar address math.

The fourth limiter is workgroup-level. A CU can hold only a fixed number of resident workgroups, enforced in part by **barrier resources** — every workgroup of more than one wavefront needs a hardware barrier to implement `__syncthreads` / `s_barrier`, and a CU has a limited pool of them. This one stays invisible until your workgroups get *small*: a swarm of tiny one- or two-wave workgroups can exhaust the workgroup or barrier slots while VGPRs, SGPRs, and LDS still have room to spare. With the 256-thread (4-wave) tiles typical of grouped GEMM it rarely binds — but for fine-grained kernels it's the limiter people forget, right up until the profiler shows occupancy stuck below what the register and LDS math predicts.

![mi355x_05_four_limiters.svg](/blog/occupancy-mi355x/mi355x_05_four_limiters.svg)

### The unit mismatch that trips everyone up

Notice the limiters don't all produce the same quantity. VGPRs and SGPRs are per-SIMD resources, so they yield **waves per SIMD**. LDS is a per-CU resource — one scratchpad shared by all four SIMDs, as we saw in Part 1 — so it yields **workgroups per CU**. You can't take a `min` across different units; you have to convert first.

The bridge is the workgroup's wave count and how those waves land on SIMDs. A 256-thread workgroup is 4 waves, and the hardware spreads those 4 waves one-per-SIMD across the CU. So each resident workgroup contributes one wave to every SIMD, which lets you restate the LDS limit (workgroups/CU) in the same waves/SIMD currency as the register limits — and *then* take the minimum. Larger workgroups change the conversion factor: an 8-wave (512-thread) workgroup puts 2 waves on each SIMD, so each workgroup costs two waves/SIMD instead of one.

Concretely: suppose LDS allows 6 resident workgroups per CU and each workgroup is 4 waves. Those 4 waves spread one-per-SIMD, so 6 workgroups put 6 waves on every SIMD — under the cap of 8, so the LDS limit *expressed in waves/SIMD* is 6. Only now can you line it up against the VGPR limit (also in waves/SIMD) and take the smaller. Compare a raw `6 workgroups/CU` against a `5-wave/SIMD` register limit directly and you're comparing apples to oranges.

### A worked example

Take a realistic MXFP8 grouped-GEMM tile: a 256-thread workgroup (4 waves) using 96 VGPRs per thread, 48 SGPRs per wave, and 32 KB of LDS per workgroup. Run the three limiters:

```
VGPR:  floor(512 / 96) = 5 waves / SIMD
SGPR:  floor(800 / 48) = 16 waves / SIMD     (not binding)
WG:    workgroup / barrier cap on workgroups/CU   (not binding for a 4-wave tile)
LDS:   depends on the LDS budget, which is exactly what changed across generations
```

So compare the two generations directly:

```
CDNA3 (MI300X, 64 KB LDS)
  LDS:  floor(64 / 32) = 2 workgroups/CU  ->  2 waves/SIMD
  min( VGPR 5, SGPR 16, LDS 2 ) = 2 waves/SIMD
  occupancy = 2 / 8 = 25%        <- LDS-bound

CDNA4 (MI355X, 160 KB LDS)
  LDS:  floor(160 / 32) = 5 workgroups/CU  ->  5 waves/SIMD
  min( VGPR 5, SGPR 16, LDS 5 ) = 5 waves/SIMD
  occupancy = 5 / 8 = 62.5%      <- register-bound
```

Same kernel, same tile — 25% on CDNA3, 62.5% on CDNA4. And notice *what* changed: on MI300X the kernel was strangled by LDS; the 160 KB scratchpad on MI355X lifts that ceiling until the VGPR limit (5) becomes the constraint instead. More LDS didn't just raise the number — it relocated the bottleneck. That relocation is the whole reason to do this math by hand rather than read a profiler percentage and shrug.

![mi355x_06_worked_example.svg](/blog/occupancy-mi355x/mi355x_06_worked_example.svg)

### Granularity: where hand-math drifts from reality

The floor divisions above assume your kernel uses exactly the registers you think it does. It doesn't — the hardware allocates registers in fixed-size blocks, so your effective count is always rounded *up* to the next block boundary. VGPRs round to a small granularity (check your `objdump` output; commonly 4–8 on CDNA), SGPRs to roughly 16, and LDS to its own granularity too.

That rounding has teeth. Suppose the compiler reports **100 VGPRs per thread**. By hand you'd expect `floor(512 / 100) = 5` waves/SIMD, or 62.5%. But with a granularity of 8, 100 rounds up to **104**, and `floor(512 / 104) = 4` waves — 50%. You silently lost a wave to four registers you didn't know you were spending.

The flip side is free occupancy. Trim that same kernel back under the boundary — to **96 VGPRs** — and `floor(512 / 96) = 5` waves returns you to 62.5%. Shaving a handful of registers to drop below a granularity boundary is one of the cheapest occupancy wins there is, and it's exactly why you read the *rounded* number out of the binary rather than trusting the one in your head. When hand math and the profiler disagree by a single wave, granularity is almost always the reason.

### How to measure it

Two ways, and you want both. Statically, ask the compiler and the binary what the kernel actually consumes:

```
# resource usage at compile time
hipcc -Rpass-analysis=kernel-resource-usage kernel.cpp

# or read it straight out of the code object
llvm-objdump --disassemble --mcpu=gfx950 kernel.hsaco | grep -iE "vgpr|sgpr|lds|granulated"
roc-obj-ls a.out
```

Those give you the inputs to the formulas above — the *theoretical* occupancy ceiling. For what the kernel achieved at runtime, read the wavefront-occupancy counter with rocprofv3, alongside `VALUUtilization` and `MemUnitBusy` so you can tell whether occupancy is even your problem.

### Theory vs. the profiler

Here is a caveat that becomes the launch point for Part 3. The number you derive by hand is a *ceiling* — the most waves that could be resident given resources. The number rocprofv3 reports is an *average over time*, and it can land above or below your hand figure for entirely mundane reasons: granularity rounding, partial final workgroups, waves draining at the kernel's edges, the sampling window. It's common to derive a 62.5% ceiling and watch the profiler report something a few points off. Neither number is wrong; they answer different questions.

Which raises the question the rest of this post exists to answer. If occupancy is this slippery to even pin down — and if all it counts is *how many waves are parked*, not whether they're doing anything — why do we treat maximizing it as the goal? That's Part 3.

---

## Part 3 — Better performance at lower occupancy

Part 2 ended on a question: if occupancy only counts how many waves are parked, why chase it? Here is the answer, and it's the most useful idea in this whole post.

Occupancy is *one* mechanism for hiding latency — not the only one. When a wave stalls on a memory load or a long MFMA, the scheduler covers the gap by issuing from another resident wave. More resident waves means more stalls you can paper over. That's thread-level parallelism (TLP), and it's real. But it isn't the only parallelism the hardware can exploit, and treating it as the goal blinds you to the other source.

This argument isn't new — Vasily Volkov made it for NVIDIA's Fermi in his 2010 GTC talk, *Better Performance at Lower Occupancy*. CDNA4 changes the hardware, not the logic.

### Little's Law: how much parallelism do you actually need?

The parallelism required to fully hide latency is given by Little's Law:

```
parallelism-in-flight = latency × throughput
```

To keep a functional unit saturated you need enough independent operations in flight to cover its latency at its issue rate. For the matrix core: if an MFMA takes L cycles to retire and the unit accepts a new one every T cycles, you need roughly L/T independent MFMAs in flight at all times. Fall short and the unit idles between dependent ops; meet it and you're at peak — regardless of *how* you supplied those independent ops.

That last clause is the whole game. Little's Law asks for independent operations in flight. It does not ask for waves.

### Two ways to get it: TLP and ILP

There are two ways to put L/T independent MFMAs in flight:

- **TLP (occupancy):** many resident waves, each issuing one MFMA. The independence comes from *different waves*.
- **ILP:** fewer waves, each issuing several *independent* MFMAs — multiple accumulator tiles advanced in parallel inside one wave. The independence comes from *within the wave*.

Both satisfy Little's Law. And here's the CDNA-specific kicker: MFMA accumulates into AccVGPRs, a register pool separate from the architected VGPRs. Holding several independent accumulator tiles for ILP draws down the *accumulator* pool — it doesn't evict the operand registers the way piling everything into one pool would. ILP is comparatively cheap on this architecture, which is exactly why you can afford to spend registers on it instead of on more waves.

![mi355x_07_tlp_vs_ilp.svg](/blog/occupancy-mi355x/mi355x_07_tlp_vs_ilp.svg)

### Hiding arithmetic latency with fewer waves

You can watch this happen with a microbenchmark. Write a kernel whose inner loop issues K independent MFMA chains (`#pragma unroll`, K separate accumulator tiles), run it at varying occupancy, and plot the fraction of MXFP8 peak you reach. The shape is consistent across GPUs: at K=1 (no ILP) you need high occupancy to approach peak; raise K to 2–4 and you hit peak at *2–4 waves/SIMD* instead of 8. The independent work the matrix core needs came from inside the wave, so you no longer need a crowd of waves to supply it.

*(The curve here is yours to measure — matrix-engine busy vs waves/SIMD at K = 1, 2, 4.)*

### Hiding memory latency with fewer waves

The same logic covers memory. Little's Law for HBM:

```
bytes-in-flight = HBM latency × bandwidth
```

At 8 TB/s, even a sub-microsecond HBM latency implies only a few megabytes in flight across the entire 256-CU GPU — on the order of tens of KB per CU. You don't need 32 waves to reach that; a handful of waves each issuing **wide, vectorized `buffer_load`s**, with several loads outstanding before the `s_waitcnt`, will saturate the bus. The lever is bytes-per-wave-in-flight (vector width × outstanding loads), not wave count. Fetch more per wave and you hide the same latency with fewer of them.

### The register/LDS bandwidth gap — and what 160 KB is really for

Recall from Part 1 that only the register file feeds the matrix core at full rate; LDS is slower and bank-conflict-prone. CDNA4's headline 160 KB of LDS is a real gift, but it's tempting to misread it as "fast memory you should pack data into." The accumulator belongs in AccVGPRs, full stop — that's the only thing that sustains peak MFMA.

So what *is* the extra LDS for? Depth. Use it to stage more of the operand stream ahead of the matrix core — deeper software-pipelined prefetch, more buffered K-steps — so the compute units never wait on HBM. The 160 KB buys a longer runway to keep loads ahead of math, which is precisely what lets a low-occupancy, big-tile kernel stay fed. It's a latency-hiding budget, not an accumulator substitute.

### The trap: cranking occupancy can starve the matrix core

Now the punchline. The register file is fixed at 512 VGPRs/SIMD, so occupancy and per-wave register budget are in direct, zero-sum tension:

```
more waves/SIMD  ->  fewer VGPRs per wave  ->  smaller accumulator tile
                 ->  lower arithmetic intensity (more LDS/HBM traffic per MFMA)
                 ->  matrix core waits on data  ->  throughput DOWN
```

Past the point where you have just enough waves (plus ILP) to cover latency, every additional wave you buy by shrinking the tile is a wave you didn't need — paid for with arithmetic intensity you did. That's how a kernel at 75% occupancy loses to the same kernel reworked to 37.5%: the low-occupancy version holds a bigger register-resident tile, does more compute per byte moved, and keeps the matrix core saturated through ILP. For MFMA-bound kernels the sweet spot is routinely 2–4 waves/SIMD, not 8.

![mi355x_08_occupancy_trap.svg](/blog/occupancy-mi355x/mi355x_08_occupancy_trap.svg)

### Case study: an MXFP8 grouped GEMM tile sweep

Put it together on a real kernel. Take a grouped GEMM and sweep the per-wave output tile, recording at each step the AccVGPR usage, waves/SIMD, occupancy, and achieved TFLOP/s.

![mi355x_09_sweep.svg](/blog/occupancy-mi355x/mi355x_09_sweep.svg)

As the tile grows, AccVGPR per wave rises and occupancy falls — yet throughput climbs, because each wave now carries more independent MFMA work and a higher arithmetic intensity. It climbs until a knee: push the tile too far and you either spill registers (catastrophic) or thin the wave count below what's needed to hide memory latency. The peak sits at low occupancy, on the far side of where the "maximize occupancy" advice would have told you to stop.

The numbers on that curve are yours to fill from rocprofv3 — but the shape is the point, and it's the same shape Volkov measured more than fifteen years ago.

### So what should you actually optimize?

Not occupancy. Optimize for keeping the matrix core fed: enough parallelism in flight to cover latency, sourced as much from ILP and wide loads as from waves; the biggest register-resident tile that doesn't spill; LDS spent on pipeline depth, not as an accumulator. Then measure the right thing — matrix-engine and VALU utilization, not the occupancy percentage.

Occupancy is one input to the latency-hiding equation. It was never the answer.

---

## Conclusion: a workflow, not a number

We covered the CDNA4 hardware and its private-vs-shared resource split (Part 1), the four limiters and how to compute occupancy by hand (Part 2), and why that ceiling is rarely the target (Part 3). The throughline: occupancy is a *diagnostic*, not an objective — it tells you how much TLP is resident, which is useful precisely because it lets you reason about whether TLP is what you're short on.

So here's how to use all of it. Next time you open a kernel:

1. **Read the real resource usage** from the binary (`-Rpass-analysis=kernel-resource-usage`, `llvm-objdump --mcpu=gfx950`) — not the numbers in your head. Mind the granularity rounding.
2. **Compute the four limiters**, convert them to a common unit, and take the minimum. The minimum is your ceiling; the *argmin* is your binding resource.
3. **Check whether occupancy is even your problem** — pull matrix-engine and VALU utilization from rocprofv3. If the matrix core is already saturated, occupancy is a distraction. Stop here.
4. **If you're latency-bound, ask where the parallelism should come from.** Usually the cheap fix is more ILP (independent accumulator tiles) or wider loads, not more waves.
5. **If you're matrix-bound and under-fed, spend the register file on the tile, not on waves.** Grow the register-resident tile until just before it spills; let occupancy fall; use the 160 KB LDS for pipeline depth.

![mi355x_10_decision_flowchart.svg](/blog/occupancy-mi355x/mi355x_10_decision_flowchart.svg)

The kernels that win on MI355X treat the 512-VGPR file and the matrix core as the scarce resources — and treat occupancy as the readout that tells them which knob just moved.

> **MI355X occupancy cheat sheet** — 512 VGPR/SIMD (private) · separate AccVGPR pool · ~800 SGPR/SIMD · 160 KB LDS/CU (shared) · max 8 waves/SIMD, 32/CU · limiters: VGPR, SGPR, LDS, workgroup (+barriers) · occupancy = min(limiters), then clamp.

*Compute the ceiling so you know where it is — then decide, deliberately, how far below it to stop.*
