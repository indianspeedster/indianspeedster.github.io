---
title: "The Tensor Data Mover: How CDNA5 Moves Tiles Without Touching a VGPR"
description: "CDNA5 adds a dedicated DMA engine that walks up-to-5D tensors and lands tiles straight in LDS, ignoring the EXEC mask and addressing nothing per lane. What the D# descriptor holds, how tile addressing and out-of-bounds clamping work, LDS padding for bank conflicts and transpose, cluster multicast, iteration and gather, and how TENSORcnt lets the copy overlap the math."
date: 2026-09-02
tags: ["GPU", "AMD", "CDNA5", "kernels", "TDM", "LDS", "GEMM"]
draft: false
---

Every GEMM kernel you have ever written spends most of its instruction budget on something that isn't multiplication. You compute a per-lane global address, issue a load, wait on the load counter, write the value to LDS, hit a barrier, and only then does the matrix core get to do its one job. The arithmetic is a rounding error in the instruction stream; the addressing is the kernel.

CDNA5 moves that work into hardware. The **Tensor Data Mover** (TDM) is a DMA engine sitting next to each pair of SIMDs that takes a description of a tensor and a tile within it, walks the tensor in global memory on its own, and writes the tile straight into LDS. It uses no VGPRs. It ignores the EXEC mask. It runs in parallel with whatever the wave does next.

This post is a close reading of §10.11 of the *"CDNA5" Instruction Set Architecture* manual, plus the pieces scattered through the rest of the document that you need in order to make sense of it — the counter model in §5.7, workgroup clusters in §2.3, multicast loads in §10.7, and the async LDS barrier in §11.2.2. I wanted to understand the whole feature as a programming model rather than as a table of bitfields, and the manual, quite reasonably, presents it as a table of bitfields.

> **TL;DR.** TDM is a per-SIMD-pair DMA engine driven by a **Tensor DMA Descriptor (D#)** built out of 12 or 20 SGPRs. Two instructions — `TENSOR_LOAD_TO_LDS` and `TENSOR_STORE_FROM_LDS` — hand it that descriptor and return immediately; completion is tracked with a new counter, `TENSORcnt`, drained by `S_WAIT_TENSORCNT`. The descriptor describes a tensor of up to **5 dimensions** and a tile within it, and the engine handles the address walk, out-of-bounds clamping (reads return zero, writes are dropped), optional **LDS padding** to kill bank conflicts or set up a transpose, **multicast** of one tile into the LDS of up to 16 workgroups in a cluster, **iteration** over strided rows, and a 2D **gather/scatter** mode with up to 16 row indices per instruction. It can also fire an LDS async-barrier arrive on completion, so consumer waves can be woken without the producer polling. The whole point is that the tile for iteration *n+1* moves while the matrix core chews on tile *n*.

Nine diagrams below carry the load; the prose around each one says what it is showing and which part of the manual it comes from. Every figure is hand-drawn in Excalidraw, and the `.excalidraw` source sits next to the `.svg` in [`public/blog/cdna5-tdm/`](https://github.com/indianspeedster/indianspeedster.github.io/tree/main/public/blog/cdna5-tdm) if you want to pull one apart or reuse it.

> **This is a documentation read, not a benchmark.** Everything below is derived from AMD's published CDNA5 ISA manual. I have not run a single instruction of this on hardware — CDNA5 parts weren't in my hands when I wrote this. Where I say something is fast, I mean the architecture is built so that it should be; where I extrapolate beyond what the manual states, I say so. Nothing here is endorsed or reviewed by AMD, and where I flag something as a spec ambiguity that's my reading, not an official erratum.

The post is in five parts:

- **Part 1 — Why a DMA engine.** The three ways to fill LDS on CDNA5, and what each one costs you.
- **Part 2 — The programming model.** Where TDM sits in the WGP, the two instructions, the D#, and the `TENSORcnt` ordering rules that are the whole reason this is worth using.
- **Part 3 — Addressing.** Tensor versus tile, the 5D address walk, out-of-bounds semantics, and LDS padding.
- **Part 4 — The optional modes.** Cluster multicast, descriptor iteration, gather and scatter.
- **Part 5 — Synchronization and the sharp edges.** The async barrier arrive, the pipelined GEMM shape it enables, and a checklist of everything the manual says will bite you.

---

## Part 1 — Why a DMA engine

CDNA5 gives you three ways to get a block of global memory into LDS. They are not variations on a theme; they sit at genuinely different points on a cost curve.

**The VGPR round trip.** The classical path: `GLOBAL_LOAD_B128` into registers, `S_WAIT_LOADCNT`, `DS_STORE_B128` out to LDS. Every lane computes its own address, the data lands in the register file, and then you pay a second instruction and a second trip through the LDS pipe to put it where you actually wanted it. The register pressure is real — a deep software pipeline needs registers for tiles in flight, and on a matrix-core kernel those are the same registers the accumulator wants. This is the path that makes occupancy tuning miserable.

**Async global→LDS.** §10.8 documents `GLOBAL_LOAD_ASYNC_TO_LDS_B{8,32,64,128}` and the matching store. These skip the register file entirely: one VGPR supplies the global address, another supplies the LDS address, and the data goes straight from memory into LDS without ever being architecturally visible. Completion is tracked by `ASYNCcnt` and drained with `S_WAIT_ASYNCCNT`. This is a big improvement and it is still per-lane — you are computing 32 addresses in VGPRs, and the amount of data one instruction can move is capped at 32 lanes × 16 bytes.

**TDM.** One instruction, no VGPRs, no per-lane anything, and the amount of data it moves is bounded by the tile you described, not by the wave width. From the manual:

> Tensor instructions are encoded as VIMAGE or VGLOBAL instructions but use no VGPRs and are not performed per-lane. The EXEC mask is ignored: tensor instructions are issued no matter if EXEC==0 or not, and it makes no difference which lanes are enabled or disabled.

That sentence is the entire pitch. A TDM instruction is closer to a scalar instruction than a vector one — it is a *command to a coprocessor*, parameterized by a descriptor in SGPRs, and it decouples the size of the transfer from the shape of the wave that requested it.

The cost structure inverts accordingly. The per-lane paths get more expensive as the tile grows, because you issue more instructions. TDM has a fixed setup cost — you have to populate 12 or 20 SGPRs — that is amortized over the whole tile, and if the tile shape is loop-invariant you populate them once outside the loop and re-issue the instruction with an updated base address each iteration.

![Three columns comparing the VGPR round trip, async global-to-LDS copies and the Tensor Data Mover, each showing the path from global memory down to LDS and the instruction, register and counter cost.](/blog/cdna5-tdm/tdm_01_three_paths.svg "The three paths differ in where the data stops on its way to LDS. The VGPR round trip parks it in the register file and pays a second instruction to push it out again. The async copy skips the register file but still computes one address per lane. TDM takes neither: the shape of the transfer lives in a descriptor, so the instruction count stops scaling with the tile.")

Read the middle row as the real distinction. It is not that TDM is faster at moving a given 128 bits — it is that the first two columns make the *wave* responsible for addressing, and the third does not.

Here is the same idea as a table:

| | VGPR round trip | Async to LDS | TDM |
|---|---|---|---|
| Instructions per tile | 2 per chunk + waits | 1 per chunk | 1 total |
| VGPRs consumed | data + addresses | addresses only | **none** |
| Address computation | per lane, in VALU | per lane, in VALU | in the DMA engine |
| Per-instruction volume | 32 lanes × ≤16 B | 32 lanes × ≤16 B | the whole tile |
| Honors EXEC | yes | yes | **no** |
| Counter | `LOADcnt` + `DScnt` | `ASYNCcnt` | `TENSORcnt` |
| Multi-dimensional | you write the loop | you write the loop | up to 5D in hardware |

If you have worked with Hopper's Tensor Memory Accelerator, the shape of this will be familiar: a descriptor-driven copy engine that understands multi-dimensional tiles and signals completion through a barrier in shared memory. The details differ substantially — CDNA5's descriptor lives in SGPRs rather than in memory, the dimensionality and the padding model are different, and TDM's gather mode has no direct TMA analogue — but the architectural bet is the same one. Data movement for matrix kernels is regular enough to be described declaratively, so stop making the shader core do it imperatively.

---

## Part 2 — The programming model

### Where it lives

§1.2.2.1 places the engine precisely:

> Each pair of SIMDs connects to the WGP$ and a Tensor Data Mover (TDM) that can move large blocks of structured data between external memory and LDS.

![A workgroup processor containing two SIMD pairs, each with its own Tensor Data Mover beneath it, both writing into a shared LDS and WGP cache block, with a bidirectional path to global memory.](/blog/cdna5-tdm/tdm_02_where_it_lives.svg "Two SIMD pairs, two TDMs, one LDS. The engine sits between the SIMDs that issue descriptors and the LDS that receives tiles, and it reaches global memory through the ordinary cache hierarchy.")

So the granularity is a **SIMD pair**, not a whole WGP and not a single SIMD. That matters for a mental model of contention: waves on the two SIMDs sharing a TDM share its issue bandwidth, and the manual notes there are config registers (`TDM_CONTROL`) governing "arbitration and TDM policies" — the arbitration exists because multiple waves will be queueing descriptors at one engine. The manual doesn't document the policy knobs, so I won't guess at them.

The destination is the LDS half of the WGP's local memory unit. §11.1 gives the budget: 320 KB of LDS and 64 KB of WGP$ sharing the same physical array, split into **64 banks of 4 bytes each**. Hold on to the bank count; it comes back in Part 3 when we get to padding.

### The two instructions

There are exactly two:

```
TENSOR_LOAD_TO_LDS      // global memory -> LDS
TENSOR_STORE_FROM_LDS   // LDS -> global memory
```

They are encoded in the VIMAGE format, but almost every VIMAGE field is repurposed or forced to a constant. The fields that carry meaning are the four `VADDR` slots, which despite the name hold **SGPR addresses**, not VGPR addresses:

| Field | Meaning |
|---|---|
| `VADDR0` | SGPR base of a 4-SGPR block: D# group 0 |
| `VADDR1` | SGPR base of an **8**-SGPR block: D# group 1 |
| `VADDR2` | SGPR base of a 4-SGPR block: D# group 2, or `NULL` |
| `VADDR3` | SGPR base of a 4-SGPR block: D# group 3, or `NULL` |
| `VADDR4` | unused — set to `NULL` (`0x7C`) |
| `SCOPE`, `TH`, `NV` | memory scope, temporal hint, non-volatile |

![The four descriptor groups laid out as columns — group 0 with four SGPRs, group 1 with eight, groups 2 and 3 with four each — listing their fields, plus callouts showing how iterate_enable and gather_mode reinterpret groups 2 and 3.](/blog/cdna5-tdm/tdm_03_descriptor.svg "The D# in full. The two callouts on the right are the part that makes this a tagged union rather than a struct: turning on iteration or gather silently changes what the group 2 and 3 registers mean.")

Add it up: a 2D transfer needs groups 0 and 1, which is **12 SGPRs**; anything up to 5D needs all four groups, which is **20 SGPRs**. `VADDR2` and `VADDR3` must both be `NULL` or both be non-`NULL` — you cannot supply group 2 without group 3. If you do pass `NULL`, the effect is defined as though you had pointed at a block of zeroed SGPRs, and the manual explicitly permits pointing `VADDR2` and `VADDR3` at the *same* SGPR block if you need group 3 to be a copy of group 2.

Twelve to twenty SGPRs is not nothing — a wave has 106 general SGPRs — but it is a one-time cost for a descriptor you will typically build once and re-issue many times, and it is coming out of the scalar file rather than the vector file that your accumulator tile is fighting over.

### The counter, and why it is the point

CDNA5 tracks outstanding memory work with a family of counters — `LOADcnt`, `STOREcnt`, `DScnt`, `KMcnt`, `ASYNCcnt`, `XCNT` — and adds one for this:

> **TENSORcnt** — 6 bits — Tensor (matrix) DMA operation count. Incremented by 1 for each TDM transfer issued. Decremented by 1 for each operation completed.

Six bits, so up to 63 outstanding transfers per wave, which is far more pipelining depth than any sane kernel will use. You drain it with `S_WAIT_TENSORCNT N`, which blocks until `TENSORcnt <= N` — note the `<=`, which is what makes multi-stage pipelining expressible. `S_WAIT_TENSORCNT 1` with two transfers in flight means "wait for the older one, keep the newer one running."

The ordering rules in §10.11.1 are worth quoting in full because they define exactly what you may and may not assume:

> Tensor-Done is returned once per instruction, not per memory transfer. Tensor instructions complete in-order with other Tensor instructions from the same wave (both loads and stores stay in-order), but are unordered with instructions from other waves. Tensor instructions are unordered with respect to other types of memory instructions.

Three separate guarantees, and the third one is a trap.

1. **Completion is per-instruction.** The engine may split your tile into a hundred memory transactions; you get one decrement when all of them have landed. You never see partial completion.
2. **Same-wave TDM ops are FIFO**, loads and stores together. Issue a load then a store, and `S_WAIT_TENSORCNT 1` retires the load. This is a stronger guarantee than `ASYNCcnt` gives you — for async ops, loads and stores complete out of order with respect to each other.
3. **TDM is unordered against every other memory type.** A `GLOBAL_STORE` issued before a `TENSOR_LOAD_TO_LDS` from the same wave has no defined ordering against it. If a TDM load must observe a prior store, you need an explicit wait on the *store's* counter first. `S_WAIT_TENSORCNT` orders TDM against TDM and nothing else.

There is also a small but easy-to-miss restriction: **tensor instructions may not occur in a clause.** If your assembler or compiler is grouping memory instructions into clauses for issue efficiency, TDM ops must sit outside them.

One more consequence of `TENSORcnt` shows up in an unrelated corner of the spec, §3.2.2.1 on instruction skipping. The hardware may skip vector memory instructions when `EXEC == 0`, but only if `LOADcnt`, `STOREcnt`, `ASYNCcnt` **and** `TENSORcnt` are all zero. An outstanding TDM transfer suppresses that optimization for the whole wave. Not something to design around, but a nice illustration of how deeply the counter is wired into the wave state machine.

---

## Part 3 — Addressing

### Tensor versus tile

The descriptor names two objects, and keeping them straight is most of the battle.

The **tensor** is the logical array in global memory. It is described by `tensor_dim[0..4]` — the extent along each of up to five dimensions, in elements — and `tensor_dim[0..3]_stride` — the distance from one line to the next, also in elements. Strides are separate from extents precisely so that they can be *larger*, which is how you describe a sub-region of a bigger array, or an array with padding between rows.

The **tile** is the sub-region you actually want moved, described by `tile_dim[0..4]` and a destination `lds_addr`.

And then there is the field that catches everyone:

> `global_addr` — Global memory address of the start of **the tile within the tensor** (not the start of the tensor).

![A grid representing a 2D tensor with two shaded padding columns on the right, a blue tile highlighted inside it, a red dot marking where global_addr points, and brackets labelling tensor_dim0, tensor_dim0_stride, tile_dim0 and tile_dim1.](/blog/cdna5-tdm/tdm_04_tensor_tile.svg "tensor_dim0 is how much real data a row holds; tensor_dim0_stride is how far apart rows actually are, which is how padded and sub-region layouts get described. The red dot is the field that moves as you step the tile.")

`global_addr` points at the tile's first element, not the tensor's origin. `tensor_dim` exists purely so the engine knows where the *tensor* ends for bounds-checking purposes. Which means that stepping a tile across a matrix is done by bumping `global_addr` — two SGPRs of arithmetic between iterations — while every other field stays put. That is a deliberate and rather nice piece of design: the loop-varying part of the descriptor is one 57-bit field.

Element size is `data_size`, encoded as log2 in a 2-bit field: `0`→1 byte, `1`→2, `2`→4, `3`→8. The manual's own pseudocode does the decode explicitly with `dataSize = (1 << D#.data_size)`, while the address formulas a page earlier write `D#.data_size *` as if the field held the byte count directly. Read the formulas as using the decoded size; the pseudocode is the authoritative version.

### The address walk

For a 2D tensor:

```
global_addr[2d] = D#.global_addr + dataSize * (x + y * D#.tensor_dim0_stride)
```

and each additional dimension adds a term with the next stride:

```
global_addr[3d] = D#.global_addr + dataSize * (x
                + y * D#.tensor_dim0_stride
                + z * D#.tensor_dim1_stride)

global_addr[4d] = D#.global_addr + dataSize * (x
                + y * D#.tensor_dim0_stride
                + z * D#.tensor_dim1_stride
                + zz * D#.tensor_dim2_stride)
```

Note that `tensor_dimN_stride` is documented as "the total stride across the first N+1 dimensions" — these are cumulative strides, not per-dimension increments to be multiplied together. `tensor_dim1_stride` already includes whatever `tensor_dim0_stride` contributed. If you are used to writing `z * dim0 * dim1`, that multiplication is *not* what the hardware does; you supply the product yourself.

The strides are wide: 48 bits, in elements. At 8-byte elements that is a 51-bit byte stride, comfortably more than the 57-bit `global_addr` field can reach. These fields are not going to be your limit.

The manual's own pseudocode for the 3D load is the clearest statement of what the engine actually does:

```
Laddr = D#.lds_addr        // LDS write pointer
bytesStored = 0            // bytes written since last pad
dataSize = (1 << D#.data_size)

for Z = 0..D#.tile_dim2
  for Y = 0..D#.tile_dim1
    Maddr = D#.global_addr + dataSize * (y * tensor_dim0_stride
                                       + z * tensor_dim1_stride)
    for X = 0..D#.tile_dim0
        LDS[Laddr] = Memory[Maddr]      // dataSize sequential bytes
        Maddr += dataSize
        Laddr += dataSize
        bytesStored += dataSize
        if (D#.pad_enable && (bytesStored >> 3) >= (1 << D#.pad_interval))
            bytesStored = 0
            Laddr += D#.pad_amount
```

Three things to read out of this.

**The LDS side is dense and the global side is strided.** `Laddr` only ever advances by `dataSize` (plus padding). The tile arrives in LDS packed, in row-major order, regardless of how scattered it was in memory. There is no LDS-side stride field. If you want a non-contiguous LDS layout, padding is the only lever you have.

**Innermost dimension 0 is contiguous in both spaces.** The engine reads `tile_dim0` consecutive elements from consecutive addresses. Dimension 0 is where your coalescing lives, so it wants to be the fast-varying dimension of the memory layout — the same rule as always, just now expressed in a descriptor field instead of in your indexing code.

**The loop bounds are written inclusively.** `for X = 0..D#.tile_dim0` reads as `tile_dim0 + 1` iterations, which contradicts `tile_dim0` being "tile dimension 0 in elements." I read this as loose pseudocode for a half-open range — a tile of `tile_dim0` elements — but it is exactly the kind of off-by-one I would verify against real hardware before trusting a hand-built descriptor.

### Out of bounds is defined, not undefined

This is my favorite part of the design, and it is easy to skim past:

> Reads from portions of a tile that extend beyond the right (positive address) end of a tensor dimension in global memory return zero, and writes to those portions of the tile are dropped.

![Two panels, each showing a tile straddling the right and bottom edge of a tensor. On the load side the cells outside the tensor are filled with zeros; on the store side they are crossed out to show the writes being dropped.](/blog/cdna5-tdm/tdm_05_out_of_bounds.svg "The same edge tile under a load and under a store. Neither faults, neither needs a predicated slow path, and neither can touch memory belonging to a neighbouring tile.")

Ragged edges are handled in hardware, with the *right* semantics. A GEMM whose M or N is not a multiple of the tile size normally needs either a separate epilogue kernel, a predicated slow path, or pre-padded buffers. With TDM you issue the same descriptor for the edge tile, the engine zero-fills the overhang, and the zeros contribute nothing to the accumulation. Stores get the mirror-image treatment: the out-of-range writes are simply dropped, so you cannot corrupt a neighbour by storing a full tile over a partial region.

Two caveats. It is one-sided — "the right (positive address) end" — so this protects you from running off the end, not from a negative or underflowing `global_addr`. And it is checked against `tensor_dim`, which means `tensor_dim` has to be honest. If you set it to the tile size out of convenience, you have disabled the feature.

### LDS padding: bank conflicts and transposes

![Two panels of five LDS rows with bank numbers in each cell. Without padding every row starts at bank 0, highlighted as a red column. With one DWORD of padding per row the starting banks step 0, 1, 2, 3, 4 down a green diagonal.](/blog/cdna5-tdm/tdm_06_lds_padding.svg "Why padding exists. Densely packed rows put every row start in the same bank, so a column access serializes 64 ways. One DWORD of skew per row spreads those starts across banks — and the copy engine inserts it for free, mid-transfer.")

The last two lines of the pseudocode are the padding mechanism, and they solve a specific, familiar problem.

LDS has 64 banks of 4 bytes. A column access into a densely packed tile whose row length is a multiple of 64 DWORDs hits the same bank on every row — a 64-way conflict, serialized. The classical fix is to pad each row by a DWORD or two so successive rows start in different banks. Doing that with per-lane stores means either address arithmetic on every store or a swizzle; TDM does it in the copy engine, for free:

- `pad_enable` — on or off.
- `pad_interval` — how many DWORDs to write before inserting padding. 3 bits, encoded as a power of two: `0`→2 DWORDs, `1`→4, `2`→8, … `7`→256.
- `pad_amount` — how many DWORDs to skip. 7 bits, offset by one: `0`→1 DWORD, `1`→2, … `127`→128.

Two encoding details will bite you. Both fields are offset or log-encoded, so writing the number you want gets you something else — `pad_amount = 1` skips *two* DWORDs. And the padding is expressed in DWORDs while everything else in the descriptor is in `data_size` elements; for an FP8 tile, one unit of padding is four elements.

The manual is emphatic about three constraints:

- Padding applies to **memory→LDS only**. There is no de-padding on the way out; `pad_enable` is ignored for `TENSOR_STORE_FROM_LDS`. A padded round trip needs you to unpad by hand.
- `pad_amount` and `pad_interval` must **both** be zero or **both** be non-zero. "Other values produce unpredictable results" — not zero-padding, not an error. Since `pad_amount = 0` means one DWORD and `pad_interval = 0` means two DWORDs, the natural "smallest padding" configuration is a pair of zeros, which is exactly the illegal combination unless `pad_enable` is also off. Set `pad_enable` deliberately.
- `tile_dim0` must be a multiple of 4 bytes for padding to work properly. At 1-byte elements that means multiples of 4; at 2-byte, multiples of 2.

Padding that would run past the workgroup's LDS allocation is ignored rather than faulting — a quiet failure mode worth knowing about, since it means an over-padded descriptor degrades into a differently-laid-out tile rather than an error.

The manual also notes padding is there "to lay out the data in a pattern that may be easier for matrix transpose." That is the second use: the WMMA path wants A and B fragments in a particular orientation, and CDNA5 has `DS_LOAD_TR16_B128` and friends (§11.2.4) to transpose on the way from LDS into VGPRs. Choosing a padding stride that makes the transposing load conflict-free is a real tuning knob, and it costs you nothing at copy time.

---

## Part 4 — The optional modes

Three modes sit on top of the basic copy, each flipped on by a descriptor bit.

### Multicast: one memory read, sixteen LDS destinations

§2.3 introduces **workgroup clusters** — up to 16 workgroups scheduled onto the same shader engine, each on its own WGP, able to synchronize through a cluster-wide barrier. §10.7 explains why you would want that, in one sentence:

> In GEMM applications, it is common to have multiple workgroups request the same data from memory.

![A single global memory read entering the TDM, which fans the same tile out to the LDS of four workgroups inside a dashed cluster boundary, annotated as scaling to sixteen.](/blog/cdna5-tdm/tdm_07_multicast.svg "One read of the shared operand, N copies delivered. The mask in the descriptor picks which workgroups of the cluster receive it, and the engine switches from GLOBAL_LOAD_ASYNC to CLUSTER_LOAD_ASYNC to do it.")

If sixteen workgroups are each computing a different output tile in the same row band, they all need the same A-panel. Sixteen separate reads of the same bytes is a waste of a scarce resource. So TDM can broadcast:

> If the `D#.workgroup_mask` is non-zero and the tensor instruction is `TENSOR_LOAD_TO_LDS`, the TDM moves the data between global memory and LDS using `CLUSTER_LOAD_ASYNC` ops instead of the usual `GLOBAL_LOAD_ASYNC`.

`workgroup_mask` is a 16-bit field, one bit per workgroup-in-cluster. Set it, and the tile lands in the LDS of every masked workgroup off a single trip to memory. In the limit that is a 16× reduction in bytes moved for the shared operand.

The constraints:

- **Load only.** For `TENSOR_STORE_FROM_LDS` the field is ignored. Broadcasting a store would mean sixteen writers to one address; the asymmetry is inherent.
- **`workgroup_mask` must be zero when the wave is not in a cluster.** The manual states this twice, once in §10.11 and once in the `VADDR1` field description ("`SGPR[VADDR1][15:0]` must be zero when not running in a cluster"), which suggests it is a real correctness hazard rather than a benign no-op.
- The underlying multicast mechanism (§10.7) is a **rendezvous with a timeout**: every masked workgroup is expected to make the same request, and data returns when all have arrived or the timeout fires. Late arrivals get their own separate broadcast. So the win degrades gracefully into "no win" if the workgroups are badly skewed, rather than into a hang — but it does mean the benefit depends on the cluster staying roughly in lockstep.
- `early_timeout` in the descriptor forwards a flag to the GL1 telling it to return data as soon as it arrives from GL2, to whichever requesters have shown up. That is the knob for "I would rather not wait for stragglers."
- Multicast loads force-miss the WGP$ regardless of scope and temporal hints. The broadcast path bypasses the local cache by construction.

### Iteration: strided row extraction

`iterate_enable` lets one descriptor be replayed several times with automatic base-address bumps:

> Iteration allows pulling out every Nth row from the tensor and storing them compacted into LDS as multiple smaller consecutive tiles.

The extra fields are `global_addr_increment` and `lds_addr_increment` (both in elements) and `iterate_count` (0 means iterate once, 255 means 256 times). It works on 2D and 3D tensors only.

This is a strided-slice primitive. Sub-sampling a feature map, gathering every k-th row of a matrix, deinterleaving a packed layout — the sort of thing that otherwise costs you an outer loop and a descriptor rebuild per step.

The cost is register real estate. Turning iteration on **overloads three group-2 fields**: `tensor_dim3` becomes `lds_addr_increment`, `tensor_dim2_stride` becomes `global_addr_increment`, and `tile_dim3` becomes `iterate_count`. Since those are exactly the fields a 4D or 5D tensor needs, iteration and high dimensionality are effectively mutually exclusive — consistent with iteration being restricted to 2D and 3D. And `iterate_enable` is ignored outright when gather mode is on.

### Gather and scatter: row indices in the descriptor

The most surprising mode. Set `Gather Mode` in group 0, and descriptor groups 2 and 3 stop being tensor dimensions and become **a list of row indices**:

- 16-bit indices: 16 rows per instruction.
- 32-bit indices: 8 rows per instruction.

`tile_dim1` is redefined to say how many of those indices are valid. Each row is fetched as `height=1 by width=tile_dim0`. `tile_dim2` and `tile_dim1_stride` are ignored. `tensor_dim1` still does bounds checking as usual — so an out-of-range index gives you a zero-filled row rather than a fault.

![A 2D tensor with four non-adjacent rows highlighted, an index list held in descriptor groups 2 and 3, and those four rows landing compacted and adjacent in LDS.](/blog/cdna5-tdm/tdm_08_gather.svg "Gather mode spends the group 2 and 3 registers on row indices instead of tensor dimensions. Scattered rows in memory arrive contiguous in LDS, in the order the index list names them.")

That is an embedding-table lookup, or an MoE expert-row gather, expressed as a single instruction. And because the same index list applies to `TENSOR_STORE_FROM_LDS`, you get **scatter** for free: the indices generate the destination addresses instead of the source ones.

The one caveat is precise and slightly alarming:

> Random index selection is supported; indices are not required to be strictly increasing and may repeat. However, correct handling of out-of-bounds (OOB) accesses only occurs when each index is greater than or equal to the previous index.

Read that carefully. Arbitrary index orders work. Out-of-bounds *handling* is only correct for non-decreasing index lists. So the zero-fill guarantee — the thing you were relying on in Part 3 — is conditional in gather mode. If your indices are unsorted, you must guarantee every one of them is in range yourself. Given that gather mode is aimed squarely at workloads where indices come from data (token IDs, routing decisions), that is a constraint you have to design around rather than assume away: either sort the indices, or clamp them before they reach the descriptor.

Gather is 2D only.

---

## Part 5 — Synchronization and the sharp edges

### Waking the consumers without polling

`S_WAIT_TENSORCNT` is per-wave and blocking, which is the right tool when the wave that issued the copy is also the wave that consumes it. In a real producer/consumer kernel it usually isn't: one wave issues the loads, many waves consume the tile.

So the descriptor can fire a barrier arrive on completion. Set `atomic_barrier_enable` and give it `atomic_barrier_address` — a 64-bit-aligned LDS location, stored as bits [18:3] with the low three bits implied — and:

> A tensor op may optionally request that an LDS atomic occur after the tensor completes by setting `D#.atomic_barrier_enable`. When set, after a tensor operation completes it sends `DS_ATOMIC_ASYNC_BARRIER_ARRIVE` to signal its completion.

CDNA5's LDS barriers (§11.2.2) are split-phase counting barriers living in LDS: a pending count and a phase, decremented by arrivals, and when the count rolls under, the hardware **sends a wake-up signal to sleeping waves in the workgroup** and reloads the count. Consumer waves can therefore `S_SLEEP` on the barrier instead of spinning, and the DMA engine itself performs the arrive.

The producer wave is now fully out of the loop. It issues a descriptor and moves on; the engine does the copy and signals the barrier; the consumers wake up. Nobody polls, nobody spins on a flag in LDS, and no wave burns issue slots waiting.

The manual adds one ordering note that is easy to misread: "The TDM sends `DS_ATOMIC_ASYNC_BARRIER_ARRIVE` to keep ordering of operations between descriptors of the same wave." The arrive is not only a consumer-facing signal, it is also how the engine keeps successive descriptors from the same wave ordered against each other at the LDS end.

### The shape it is all for

Put the pieces together and the intended inner loop falls out:

```
  // outside the loop: build the D# once
  s_mov_b32   ...                      // tensor dims, strides, tile dims,
  ...                                  // data_size, padding, barrier address

  // prologue: get the pipeline started
  TENSOR_LOAD_TO_LDS  D#_stage0        // tile 0 -> LDS buffer 0
  TENSOR_LOAD_TO_LDS  D#_stage1        // tile 1 -> LDS buffer 1

loop:
  S_WAIT_TENSORCNT 1                   // tile n has landed; n+1 still moving
  ...  DS_LOAD_TR16_B128 / WMMA ...    // compute on buffer (n % 2)
  s_add_u32   global_addr_lo, ...      // advance the descriptor: 2 SGPRs
  s_addc_u32  global_addr_hi, ...
  TENSOR_LOAD_TO_LDS  D#               // issue tile n+2 into the buffer
  s_cbranch   loop                     // just freed
```

![A timeline with a TDM lane showing four back-to-back tile loads and a matrix core lane showing three compute blocks, each compute block starting one slot after its load and separated by an S_WAIT_TENSORCNT 1 marker.](/blog/cdna5-tdm/tdm_09_pipeline.svg "The steady state. Each compute block runs against a tile that landed an iteration ago while the engine is already two tiles ahead, and the only synchronization is a counter threshold.")

`S_WAIT_TENSORCNT 1` is the crux: it waits for the older transfer while explicitly permitting the newer one to keep running. The matrix core works on tile *n* while the DMA engine fetches *n+2*, the wave's per-iteration overhead is two scalar adds and one instruction issue, and the register file holds accumulators rather than in-flight data. Deeper pipelines are a matter of more LDS buffers and a larger wait threshold; the 6-bit counter is not going to be what stops you.

The comparison worth making is against the same loop written with per-lane async copies: the address arithmetic, the VGPRs holding those addresses, and the instruction count all scale with tile size there and are constant here. That is the structural argument for TDM, and it holds regardless of what the hardware's achieved bandwidth turns out to be.

### The sharp edges

Collected from across §10.11, in roughly the order they will surprise you:

**Faults are imprecise and non-fatal.** `MEMVIOL` is reported if a tile address falls outside the wave's LDS allocation (and only when `LDS_CONFIG.ADDR_OUT_OF_RANGE_REPORTING == 1`), or if the global address lands outside the global aperture — scratch, LDS, or the hole. But:

> `MEMVIOL` does not cause the TDM to abort — it continues processing the entire descriptor and return `MEMVIOL` to the wave when it's done. Memviol and XNACK-error is reported once for the entire TDM operation, not per data transfer.

So a faulting descriptor runs to completion, and you learn that *something* went wrong, once, with no indication of where. Debugging a malformed descriptor will be an exercise in bisection.

**XNACK is handled internally, except when it isn't.** TDM absorbs XNACK-retry itself — page faults on the DMA path are resolved without the wave's involvement. XNACK-*error* can still surface (a write to a read-only page, for instance), and it is explicitly imprecise: "the wave cannot rewind the PC to the faulting instructions."

**Reserved fields are not decorative.** Group 0 carries `count`, `is_restore`, `is_store`, `nv` and `User_null`, all of which the manual marks "User: must set to zero" — they exist because the same descriptor format is reused by the context save/restore machinery. `is_restore = 1` lets the restore path *override the instruction's own load/store direction*. Garbage in those bits does not produce a bad tile, it produces a differently-typed operation. Zero them explicitly rather than trusting whatever was in the SGPRs.

Related: `type` in bits [127:126] of group 0 must be set to `2` ("image"). A descriptor built from zeroed registers is not a valid descriptor.

**`tile_dim0 == 0` is a NOP**, and `count == 0` means "NULL tensor, no memory is copied, no atomic_barrier sent." Two different ways to accidentally build a descriptor that silently does nothing — and in the `count == 0` case, one that also silently never signals the barrier your consumers are asleep on.

**Unused dimensions are zero, used ones start at one.** `tile_dim1` through `tile_dim4` use `0` to mean "dimension unused" and `1..Max` for a real extent. There is no way to express a legitimately empty higher dimension.

**Context switching knows about you.** There is a message, `RTN_SAVE_WAVE_HAS_TDM`, whose job is to "inform the CWSR machine this wave needs to be saved and has outstanding TDM ops," and the group-0 `count` field is documented with distinct user and context-restore meanings. In-flight DMA is part of the saved wave state. Similarly, VMID or pipeline kill stops all descriptors for affected waves, and killed ops still return "done" so that `TENSORcnt` drains — a wave being torn down will not deadlock on a wait.

---

## What I take away

Three things.

**The descriptor is the API, and it is denser than it looks.** Four bits of mode — `iterate_enable`, `Gather Mode`, `pad_enable`, a non-zero `workgroup_mask` — reconfigure what the other fields mean, sometimes overloading them outright. That is a lot of expressive power in 20 SGPRs, and it is also a lot of ways to build a descriptor that is syntactically fine and semantically something else. I would want a builder abstraction with the mutually-exclusive combinations encoded in the types, not a struct with twenty public fields.

**The out-of-bounds semantics are the quiet win.** Zero-fill on read, drop on write, checked against real tensor bounds, is the correct behavior for ragged GEMM tiles, and it deletes an entire category of epilogue code. I did not expect the bounds handling to be the feature I'd most want to use, and the one place it is conditional — unsorted gather indices — is flagged clearly enough that you can design around it.

**The point is the counter, not the engine.** The DMA hardware is the visible part, but `TENSORcnt` plus `S_WAIT_TENSORCNT N` plus the barrier-arrive-on-completion is what makes the copy genuinely overlap the math. A copy engine that you had to fully drain before computing would be a modest instruction-count win. One that lets you keep *n* transfers in flight, order them precisely against each other, and hand the completion signal to a hardware barrier that wakes sleeping consumers — that changes the shape of the loop.

Whether it delivers is a question for hardware. When I can get time on a CDNA5 part, the first thing I want to measure is the descriptor setup cost against tile size — the crossover point where a single TDM instruction beats a hand-rolled async-copy loop — and the second is what multicast is actually worth on a GEMM whose A-panel is shared across a full 16-workgroup cluster. Until then, this is a reading of the manual, and the manual is unusually clear about what it promises.

---

**References.** All quotations and field tables are from AMD's *"CDNA5" Instruction Set Architecture Reference Guide* — principally §10.11 (Tensor Data Mover Instructions), with supporting material from §1.2.2 (Cache System Hierarchy), §2.3 (Workgroup Clusters), §3.2.2 (EXECute Mask), §5.7 (Data Dependency Resolution), §10.7 (Multicast Load), §10.8 (Asynchronous Memory Load and Store), §10.9 (WMMA Matrix Load Ops with Transpose), and §11.1–11.2 (Local Data Share Operations).
