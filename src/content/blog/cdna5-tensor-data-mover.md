---
title: "Inside the CDNA5 Tensor Data Mover"
description: "CDNA5 adds a dedicated DMA engine that walks up-to-5D tensors and lands tiles straight in LDS, ignoring the EXEC mask and addressing nothing per lane. What the D# descriptor holds, how tile addressing and out-of-bounds clamping work, LDS padding for bank conflicts and transpose, cluster multicast, iteration and gather, and how TENSORcnt lets the copy overlap the math."
date: 2026-09-02
tags: ["GPU", "AMD", "CDNA5", "kernels", "TDM", "LDS", "GEMM"]
draft: false
---

Most of a GEMM kernel isn't multiplication. You compute a per-lane global address, issue a load, wait on the load counter, write the value into LDS, hit a barrier, and only then does the matrix core get to do its job. Count the instructions sometime. The arithmetic is a small fraction of them.

CDNA5 moves that work into hardware. The Tensor Data Mover is a DMA engine sitting next to each pair of SIMDs. You hand it a description of a tensor and a tile within it, and it walks the tensor in global memory on its own and writes the tile into LDS. It uses no VGPRs, ignores the EXEC mask, and runs in parallel with whatever the wave does next.

> **TL;DR.** TDM is a per-SIMD-pair DMA engine driven by a Tensor DMA Descriptor (D#) built out of 12 or 20 SGPRs. Two instructions, TENSOR_LOAD_TO_LDS and TENSOR_STORE_FROM_LDS, hand it that descriptor and return immediately. Completion is tracked with a new counter, TENSORcnt, drained by S_WAIT_TENSORCNT. The descriptor covers a tensor of up to 5 dimensions and a tile within it, and the engine handles the address walk, out-of-bounds clamping (reads return zero, writes are dropped), optional LDS padding for bank conflicts or transposes, multicast of one tile into the LDS of up to 16 workgroups in a cluster, iteration over strided rows, and a 2D gather/scatter mode carrying up to 16 row indices per instruction. It can fire an LDS barrier arrive when it finishes, so consumer waves get woken without anyone polling. All of which exists so the tile for iteration *n+1* can move while the matrix core works on tile *n*.

> **This is a documentation read, not a benchmark.** Everything here comes from AMD's published CDNA5 ISA manual. I haven't run any of it on hardware; CDNA5 parts weren't in my hands when I wrote this. Where I say something should be fast, I mean the architecture is built that way, not that I measured it. Nothing here is endorsed or reviewed by AMD, and anywhere I call something a spec ambiguity, that's my reading rather than an official erratum.

---

## Part 1 — Why a DMA engine

CDNA5 gives you three ways to get global memory into LDS, and they sit at genuinely different points on a cost curve.

The oldest is the VGPR round trip: GLOBAL_LOAD_B128 into registers, S_WAIT_LOADCNT, then DS_STORE_B128 out to LDS. Every lane computes its own address, the data lands in the register file, and you pay a second instruction and a second trip through the LDS pipe to put it where you wanted it in the first place. The register pressure is the real cost. A deep software pipeline needs registers for tiles in flight, and in a matrix-core kernel those are the registers your accumulator wants.

Then there's the async path. §10.8 documents GLOBAL_LOAD_ASYNC_TO_LDS_B{8,32,64,128} and its matching store. These skip the register file: one VGPR supplies the global address, another supplies the LDS address, and data goes from memory into LDS without ever becoming architecturally visible. ASYNCcnt tracks completion, S_WAIT_ASYNCCNT drains it. It's a real improvement, but it's still per-lane. You're computing 32 addresses in VGPRs, and one instruction moves at most 32 lanes × 16 bytes.

TDM is the third. One instruction, no VGPRs, and the volume it moves is bounded by the tile you described rather than by the wave width. The manual is blunt about how far outside the vector model it sits:

> Tensor instructions are encoded as VIMAGE or VGLOBAL instructions but use no VGPRs and are not performed per-lane. The EXEC mask is ignored: tensor instructions are issued no matter if EXEC==0 or not, and it makes no difference which lanes are enabled or disabled.

A TDM instruction behaves more like a scalar instruction than a vector one. It's a command to a coprocessor, parameterized by a descriptor in SGPRs, and it decouples the size of a transfer from the shape of the wave that asked for it.

That inverts the cost structure. The per-lane paths get more expensive as tiles grow, because you issue more instructions. TDM has a fixed setup cost of 12 or 20 SGPRs, amortized over the whole tile, and when the tile shape is loop-invariant you populate them once outside the loop and re-issue with a new base address each iteration.

![Three columns comparing the VGPR round trip, async global-to-LDS copies and the Tensor Data Mover, each showing the path from global memory down to LDS and the instruction, register and counter cost.](/blog/cdna5-tdm/tdm_01_three_paths.svg "The three paths differ in where the data stops on its way to LDS. The VGPR round trip parks it in the register file and pays a second instruction to push it out again. The async copy skips the register file but still computes one address per lane. TDM takes neither route: the shape of the transfer lives in a descriptor, so the instruction count stops scaling with the tile.")

The distinction that matters is where the addressing happens. TDM isn't faster at moving a given 128 bits. It's that the first two columns make the wave responsible for addressing and the third one doesn't.

| | VGPR round trip | Async to LDS | TDM |
|---|---|---|---|
| Instructions per tile | 2 per chunk + waits | 1 per chunk | 1 total |
| VGPRs consumed | data + addresses | addresses only | none |
| Address computation | per lane, in VALU | per lane, in VALU | in the DMA engine |
| Per-instruction volume | 32 lanes × ≤16 B | 32 lanes × ≤16 B | the whole tile |
| Honors EXEC | yes | yes | no |
| Counter | LOADcnt + DScnt | ASYNCcnt | TENSORcnt |
| Multi-dimensional | you write the loop | you write the loop | up to 5D in hardware |

If you've used Hopper's Tensor Memory Accelerator this will feel familiar: a descriptor-driven copy engine that understands multi-dimensional tiles and signals completion through a barrier in shared memory. Plenty differs in the details. CDNA5 keeps the descriptor in SGPRs rather than in memory, the dimensionality and padding models aren't the same, and TDM's gather mode has no direct TMA equivalent. But both vendors made the same bet: data movement for matrix kernels is regular enough to describe declaratively, so stop making the shader core do it a lane at a time.

---

## Part 2 — The programming model

### Where it lives

§1.2.2.1 places the engine precisely:

> Each pair of SIMDs connects to the WGP$ and a Tensor Data Mover (TDM) that can move large blocks of structured data between external memory and LDS.

![A workgroup processor containing two SIMD pairs, each with its own Tensor Data Mover beneath it, both writing into a shared LDS and WGP cache block, with a bidirectional path to global memory.](/blog/cdna5-tdm/tdm_02_where_it_lives.svg "Two SIMD pairs, two TDMs, one LDS. The engine sits between the SIMDs that issue descriptors and the LDS that receives tiles, and it reaches global memory through the ordinary cache hierarchy.")

The granularity is a SIMD pair, so not a whole WGP and not a single SIMD. That matters for thinking about contention: waves on the two SIMDs behind one TDM share its issue bandwidth. There are config registers (TDM_CONTROL) governing what the manual calls "arbitration and TDM policies," which tells you the arbitration exists because multiple waves will be queueing descriptors at one engine. The policy knobs aren't documented, so I won't speculate about them.

Tiles land in the LDS half of the WGP's local memory unit. §11.1 gives the budget: 320 KB of LDS and 64 KB of WGP$ sharing one physical array, split into 64 banks of 4 bytes each. The bank count is what makes LDS padding worth having.

### The two instructions

There are exactly two:

```asm
TENSOR_LOAD_TO_LDS      ; global memory -> LDS
TENSOR_STORE_FROM_LDS   ; LDS -> global memory
```

Both use the VIMAGE encoding, though almost every VIMAGE field is either repurposed or forced to a constant. What carries meaning are the four VADDR slots, which hold SGPR addresses despite the name:

| Field | Meaning |
|---|---|
| VADDR0 | SGPR base of a 4-SGPR block: D# group 0 |
| VADDR1 | SGPR base of an 8-SGPR block: D# group 1 |
| VADDR2 | SGPR base of a 4-SGPR block: D# group 2, or NULL |
| VADDR3 | SGPR base of a 4-SGPR block: D# group 3, or NULL |
| VADDR4 | unused, set to NULL (0x7C) |
| SCOPE, TH, NV | memory scope, temporal hint, non-volatile |

![The four descriptor groups laid out as columns, group 0 with four SGPRs, group 1 with eight, groups 2 and 3 with four each, listing their fields, plus callouts showing how iterate_enable and gather_mode reinterpret groups 2 and 3.](/blog/cdna5-tdm/tdm_03_descriptor.svg "The D# in full. The two callouts on the right are what makes this a tagged union rather than a struct: turning on iteration or gather quietly changes what the group 2 and 3 registers mean.")

A 2D transfer needs groups 0 and 1, so 12 SGPRs. Anything up to 5D needs all four groups, so 20. The last two slots have to both be NULL or both be set; you can't supply group 2 without group 3. Passing NULL behaves as though you'd pointed at zeroed SGPRs, and the manual explicitly allows aiming both at the same block if you want group 3 to duplicate group 2.

Twelve to twenty SGPRs isn't free out of a budget of 106, but it's a one-time cost for a descriptor you build once and re-issue many times, and it comes out of the scalar file rather than the vector file your accumulator is competing for.

### The counter, and why it's the point

CDNA5 already tracks outstanding memory work with a family of counters: LOADcnt, STOREcnt, DScnt, KMcnt, ASYNCcnt and XCNT. TDM adds one more:

> **TENSORcnt** — 6 bits — Tensor (matrix) DMA operation count. Incremented by 1 for each TDM transfer issued. Decremented by 1 for each operation completed.

Six bits gets you 63 outstanding transfers per wave, far deeper than any sane kernel will pipeline. S_WAIT_TENSORCNT N blocks until TENSORcnt <= N. The <= is the important part: S_WAIT_TENSORCNT 1 with two transfers in flight means "wait for the older one, let the newer one keep going."

The ordering rules in §10.11.1 define what you're allowed to assume:

> Tensor-Done is returned once per instruction, not per memory transfer. Tensor instructions complete in-order with other Tensor instructions from the same wave (both loads and stores stay in-order), but are unordered with instructions from other waves. Tensor instructions are unordered with respect to other types of memory instructions.

Completion is per-instruction, so the engine may split your tile into a hundred memory transactions and you'll still see exactly one decrement once all of them have landed. There's no partial completion to observe.

Same-wave TDM ops are FIFO, loads and stores together. Issue a load then a store and S_WAIT_TENSORCNT 1 retires the load. That's a stronger guarantee than ASYNCcnt offers, where loads and stores complete out of order relative to each other.

The third rule is the one that will catch people. TDM is unordered against every other memory type. A GLOBAL_STORE issued before a TENSOR_LOAD_TO_LDS from the same wave has no defined ordering against it. If a TDM load has to observe a prior store, wait on the store's own counter first, because S_WAIT_TENSORCNT orders TDM against TDM and nothing else.

One more restriction, easy to miss: tensor instructions may not occur in a clause. If your assembler or compiler groups memory instructions into clauses for issue efficiency, TDM ops have to sit outside them.

The counter also turns up somewhere unrelated. §3.2.2.1 says the hardware may skip vector memory instructions when EXEC == 0, but only when every one of those counters, TENSORcnt included, is zero. An outstanding TDM transfer suppresses that optimization for the whole wave. Not something you'd design around, but it shows how far into the wave state machine the counter reaches.

---

## Part 3 — Addressing

### Tensor versus tile

The descriptor names two objects, and keeping them straight is most of the work.

The tensor is the logical array in global memory, described by tensor_dim[0..4] for the extent along each dimension in elements, and tensor_dim[0..3]_stride for the distance from one line to the next, also in elements. Extents and strides are separate fields so the stride can be larger, which is how you describe a sub-region of a bigger array, or an array with padding between rows.

The tile is the sub-region you want moved, described by tile_dim[0..4] and a destination lds_addr.

Then there's the field that catches everyone:

> global_addr — Global memory address of the start of **the tile within the tensor** (not the start of the tensor).

![A grid representing a 2D tensor with two shaded padding columns on the right, a blue tile highlighted inside it, a red dot marking where global_addr points, and brackets labelling tensor_dim0, tensor_dim0_stride, tile_dim0 and tile_dim1.](/blog/cdna5-tdm/tdm_04_tensor_tile.svg "tensor_dim0 is how much real data a row holds; tensor_dim0_stride is how far apart rows actually are, which is how padded and sub-region layouts get described. The red dot is the field that moves as you step the tile.")

global_addr points at the tile's first element, not the tensor's origin. tensor_dim exists so the engine knows where the tensor ends for bounds checking. Which means stepping a tile across a matrix comes down to bumping global_addr, two SGPRs of arithmetic per iteration, while every other field stays put. The loop-varying part of the descriptor is one 57-bit field, and that's clearly deliberate.

Element size lives in data_size, log2-encoded in two bits: 0 means 1 byte, 1 means 2, 2 means 4, 3 means 8. The manual's pseudocode decodes it explicitly with dataSize = (1 << D#.data_size), while the address formulas a page earlier write D#.data_size * as though the field held a byte count. Take the pseudocode as authoritative and read the formulas as using the decoded size.

### The address walk

For a 2D tensor:

```c
global_addr[2d] = D#.global_addr + dataSize * (x + y * D#.tensor_dim0_stride)
```

Each additional dimension adds a term with the next stride:

```c
global_addr[3d] = D#.global_addr + dataSize * (x
                + y * D#.tensor_dim0_stride
                + z * D#.tensor_dim1_stride)

global_addr[4d] = D#.global_addr + dataSize * (x
                + y * D#.tensor_dim0_stride
                + z * D#.tensor_dim1_stride
                + zz * D#.tensor_dim2_stride)
```

tensor_dimN_stride is documented as "the total stride across the first N+1 dimensions." These are cumulative, not per-dimension increments you multiply together. tensor_dim1_stride already contains whatever tensor_dim0_stride contributed. If your instinct is to write z * dim0 * dim1, that multiplication isn't happening in hardware; you supply the product.

The stride fields are 48 bits wide in elements. At 8-byte elements that's a 51-bit byte stride, more than the 57-bit global_addr can reach anyway. These won't be your limit.

The manual's pseudocode for a 3D load says what the engine does more clearly than prose can:

```c
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

Notice that the LDS side is dense while the global side is strided. Laddr only ever advances by dataSize, plus padding. However scattered the tile was in memory, it arrives in LDS packed and row-major. There's no LDS-side stride field at all, so padding is your only lever if you want a non-contiguous destination layout.

Dimension 0 is contiguous in both spaces: the engine reads tile_dim0 consecutive elements from consecutive addresses. That's where your coalescing lives, so dimension 0 wants to be the fast-varying dimension of the memory layout. Same rule as always, now expressed as a descriptor field instead of as indexing code.

The loop bounds bother me slightly. for X = 0..D#.tile_dim0 reads as tile_dim0 + 1 iterations, which contradicts tile_dim0 being "tile dimension 0 in elements." I take it as loose pseudocode for a half-open range, but it's the kind of off-by-one I'd want to confirm against silicon before trusting a hand-built descriptor.

### Out of bounds is defined, not undefined

This is easy to skim past and shouldn't be:

> Reads from portions of a tile that extend beyond the right (positive address) end of a tensor dimension in global memory return zero, and writes to those portions of the tile are dropped.

![Two panels, each showing a tile straddling the right and bottom edge of a tensor. On the load side the cells outside the tensor are filled with zeros; on the store side they are crossed out to show the writes being dropped.](/blog/cdna5-tdm/tdm_05_out_of_bounds.svg "The same edge tile under a load and under a store. Neither faults, neither needs a predicated slow path, and neither can touch memory belonging to a neighbouring tile.")

Ragged edges get handled in hardware with the right semantics. A GEMM whose M or N isn't a multiple of the tile size normally needs a separate epilogue kernel, a predicated slow path, or pre-padded buffers. Here you issue the same descriptor for the edge tile, the engine zero-fills the overhang, and the zeros contribute nothing to the accumulation. Stores get the mirror treatment: out-of-range writes are dropped, so a full tile stored over a partial region can't corrupt its neighbour.

Two caveats. The protection is one-sided, covering "the right (positive address) end," so it saves you from running off the end but not from a negative or underflowing global_addr. And it's checked against tensor_dim, which has to be honest for any of it to work. Set tensor_dim equal to the tile size out of convenience and you've turned the whole mechanism off.

### LDS padding: bank conflicts and transposes

![Two panels of five LDS rows with bank numbers in each cell. Without padding every row starts at bank 0, highlighted as a red column. With one DWORD of padding per row the starting banks step 0, 1, 2, 3, 4 down a green diagonal.](/blog/cdna5-tdm/tdm_06_lds_padding.svg "Why padding exists. Densely packed rows put every row start in the same bank, so a column access serializes 64 ways. One DWORD of skew per row spreads those starts across banks, and the copy engine inserts it mid-transfer at no cost.")

Those last two lines of the pseudocode are the padding mechanism, and they solve a problem anyone who has hand-tuned an LDS layout will recognize.

LDS has 64 banks of 4 bytes. A column access into a densely packed tile whose row length is a multiple of 64 DWORDs hits the same bank on every row, which is a 64-way conflict and serializes. The usual fix is to pad each row by a DWORD or two so successive rows start in different banks. Doing that with per-lane stores costs you either address arithmetic on every store or a swizzle. TDM does it inside the copy engine:

- pad_enable turns it on.
- pad_interval sets how many DWORDs to write before inserting padding. 3 bits, power-of-two encoded: 0 means 2 DWORDs, 1 means 4, up through 7 meaning 256.
- pad_amount sets how many DWORDs to skip. 7 bits, offset by one: 0 means 1 DWORD, 1 means 2, up through 127 meaning 128.

Both fields are encoded rather than literal, so writing the number you want gets you something else. pad_amount = 1 skips two DWORDs. Padding is also expressed in DWORDs while the rest of the descriptor works in data_size elements, so for an FP8 tile a single unit of padding is four elements.

Three constraints the manual is firm about:

Padding applies to memory→LDS transfers only. There's no de-padding on the way out, and pad_enable is ignored for TENSOR_STORE_FROM_LDS, so a padded round trip means unpadding by hand.

The amount and the interval must both be zero or both be non-zero. Anything else "produce[s] unpredictable results," which is not zero padding and not an error either. Since zero in each field means one DWORD and two DWORDs respectively, the configuration you'd naturally reach for as "smallest possible padding" is exactly the illegal one unless padding is switched off entirely. Set pad_enable on purpose.

tile_dim0 must be a multiple of 4 bytes for padding to work properly, so multiples of 4 at 1-byte elements, multiples of 2 at 2-byte.

Padding that would run past the workgroup's LDS allocation is ignored rather than faulting. Worth knowing about, because it means an over-padded descriptor degrades into a differently laid out tile instead of an error you'd notice.

The manual also mentions padding is there "to lay out the data in a pattern that may be easier for matrix transpose." That's the second use. The WMMA path wants A and B fragments in a particular orientation, and CDNA5 has DS_LOAD_TR16_B128 and friends (§11.2.4) to transpose on the way from LDS into VGPRs. Picking a padding stride that makes the transposing load conflict-free is a genuine tuning knob that costs nothing at copy time.

---

## Part 4 — The optional modes

Three modes sit on top of the basic copy, each turned on by a descriptor bit.

### Multicast: one memory read, sixteen LDS destinations

§2.3 introduces workgroup clusters: up to 16 workgroups scheduled onto the same shader engine, each on its own WGP, able to synchronize through a cluster-wide barrier. §10.7 explains the motivation in one sentence.

> In GEMM applications, it is common to have multiple workgroups request the same data from memory.

![A single global memory read entering the TDM, which fans the same tile out to the LDS of four workgroups inside a dashed cluster boundary, annotated as scaling to sixteen.](/blog/cdna5-tdm/tdm_07_multicast.svg "One read of the shared operand, N copies delivered. The mask in the descriptor picks which workgroups of the cluster receive it, and the engine switches from GLOBAL_LOAD_ASYNC to CLUSTER_LOAD_ASYNC to do it.")

Sixteen workgroups computing different output tiles in the same row band all need the same A-panel, and reading those bytes sixteen times wastes the scarcest resource you have. So TDM can broadcast:

> If the D#.workgroup_mask is non-zero and the tensor instruction is TENSOR_LOAD_TO_LDS, the TDM moves the data between global memory and LDS using CLUSTER_LOAD_ASYNC ops instead of the usual GLOBAL_LOAD_ASYNC.

workgroup_mask is 16 bits, one per workgroup-in-cluster. Set it and the tile lands in every masked workgroup's LDS from a single trip to memory, up to a 16× reduction in bytes moved for the shared operand.

The constraints are worth reading before you count on that number. It's load only; for TENSOR_STORE_FROM_LDS the field is ignored, since broadcasting a store would mean sixteen writers to one address. workgroup_mask must be zero when the wave isn't in a cluster, which the manual states twice, once in §10.11 and again in the VADDR1 field description. Stating it twice suggests a real correctness hazard rather than a benign no-op.

The underlying mechanism in §10.7 is a rendezvous with a timeout. Every masked workgroup is expected to make the same request, and data returns once they all have or the timeout fires. Late arrivals get their own separate broadcast. So a badly skewed cluster degrades into no win rather than into a hang, but the benefit does depend on the cluster staying roughly in lockstep. early_timeout in the descriptor forwards a flag to the GL1 telling it to return data as soon as GL2 supplies it, to whoever has shown up, which is the knob for not waiting on stragglers.

One more thing: multicast loads force-miss the WGP$ regardless of scope and temporal hints. The broadcast path bypasses the local cache by construction.

### Iteration: strided row extraction

iterate_enable replays one descriptor several times with automatic base-address bumps:

> Iteration allows pulling out every Nth row from the tensor and storing them compacted into LDS as multiple smaller consecutive tiles.

The extra fields are global_addr_increment and lds_addr_increment, both in elements, plus iterate_count, where 0 means iterate once and 255 means 256 times. It works on 2D and 3D tensors only.

Think of it as a strided-slice primitive: sub-sampling a feature map, taking every k-th row of a matrix, deinterleaving a packed layout. The sort of thing that otherwise costs an outer loop and a descriptor rebuild per step.

What it costs is register real estate. Turning iteration on overloads three group-2 fields: tensor_dim3 becomes the LDS increment, tensor_dim2_stride becomes the global increment, and tile_dim3 becomes the iteration count. Those are exactly the fields a 4D or 5D tensor needs, so iteration and high dimensionality are effectively mutually exclusive, which lines up with iteration being restricted to 2D and 3D in the first place. iterate_enable is also ignored outright when gather mode is on.

### Gather and scatter: row indices in the descriptor

This is the mode I didn't expect to find. Set Gather Mode in group 0 and descriptor groups 2 and 3 stop being tensor dimensions and become a list of row indices: 16 of them at 16 bits each, or 8 at 32 bits.

tile_dim1 is redefined to say how many of those indices are valid, and each row comes back as height 1 by width tile_dim0. The second tile dimension and its stride are ignored. Bounds checking against tensor_dim1 still happens, so an out-of-range index gives you a zero-filled row rather than a fault.

![A 2D tensor with four non-adjacent rows highlighted, an index list held in descriptor groups 2 and 3, and those four rows landing compacted and adjacent in LDS.](/blog/cdna5-tdm/tdm_08_gather.svg "Gather mode spends the group 2 and 3 registers on row indices instead of tensor dimensions. Scattered rows in memory arrive contiguous in LDS, in the order the index list names them.")

That's an embedding-table lookup, or an MoE expert-row gather, as a single instruction. The same index list applies to TENSOR_STORE_FROM_LDS, which gives you scatter for free, with the indices generating destination addresses instead of source ones.

There's a caveat, and it's a sharp one:

> Random index selection is supported; indices are not required to be strictly increasing and may repeat. However, correct handling of out-of-bounds (OOB) accesses only occurs when each index is greater than or equal to the previous index.

Arbitrary index orders work. Out-of-bounds *handling* is only correct for non-decreasing lists. So the zero-fill guarantee from Part 3, the one I just called the best thing about the addressing model, is conditional here. With unsorted indices you have to guarantee every index is in range yourself. Since gather mode is aimed at workloads where the indices come from data (token IDs, routing decisions), that's something to design around rather than assume away: sort them, or clamp them before they reach the descriptor.

Gather is 2D only.

---

## Part 5 — Synchronization and the sharp edges

### Waking the consumers without polling

S_WAIT_TENSORCNT is per-wave and blocking, which suits the case where the wave that issued the copy is also the wave that consumes it. Real producer/consumer kernels usually aren't shaped that way: one wave issues the loads and many waves consume the tile.

So the descriptor can fire a barrier arrive when it finishes. Set atomic_barrier_enable, give it atomic_barrier_address (a 64-bit-aligned LDS location, stored as bits [18:3] with the low three implied), and:

> A tensor op may optionally request that an LDS atomic occur after the tensor completes by setting D#.atomic_barrier_enable. When set, after a tensor operation completes it sends DS_ATOMIC_ASYNC_BARRIER_ARRIVE to signal its completion.

CDNA5's LDS barriers (§11.2.2) are split-phase counting barriers living in LDS, with a pending count and a phase, decremented by arrivals. When the count rolls under, the hardware sends a wake-up signal to sleeping waves in the workgroup and reloads the count. Consumers can S_SLEEP on the barrier rather than spinning, and the DMA engine performs the arrive itself.

That takes the producer wave out of the loop entirely. It issues a descriptor and moves on, the engine does the copy and signals the barrier, the consumers wake up. Nobody polls, nobody spins on a flag in LDS, nobody burns issue slots waiting.

The manual adds an ordering note that's easy to misread: "The TDM sends DS_ATOMIC_ASYNC_BARRIER_ARRIVE to keep ordering of operations between descriptors of the same wave." The arrive isn't only a consumer-facing signal. It's also how the engine keeps successive descriptors from one wave ordered against each other at the LDS end.

### The shape it's all for

Put the pieces together and the intended inner loop falls out:

```asm
  ; outside the loop: build the D# once
  s_mov_b32   ...                      ; tensor dims, strides, tile dims,
  ...                                  ; data_size, padding, barrier address

  ; prologue: get the pipeline started
  TENSOR_LOAD_TO_LDS  D#_stage0        ; tile 0 -> LDS buffer 0
  TENSOR_LOAD_TO_LDS  D#_stage1        ; tile 1 -> LDS buffer 1

loop:
  S_WAIT_TENSORCNT 1                   ; tile n has landed; n+1 still moving
  ...  DS_LOAD_TR16_B128 / WMMA ...    ; compute on buffer (n % 2)
  s_add_u32   global_addr_lo, ...      ; advance the descriptor: 2 SGPRs
  s_addc_u32  global_addr_hi, ...
  TENSOR_LOAD_TO_LDS  D#               ; issue tile n+2 into the buffer
  s_cbranch   loop                     ; into the buffer just freed
```

![A timeline with a TDM lane showing four back-to-back tile loads and a matrix core lane showing three compute blocks, each compute block starting one slot after its load and separated by an S_WAIT_TENSORCNT 1 marker.](/blog/cdna5-tdm/tdm_09_pipeline.svg "The steady state. Each compute block runs against a tile that landed an iteration ago while the engine is already two tiles ahead, and the only synchronization is a counter threshold.")

S_WAIT_TENSORCNT 1 does the real work here, waiting for the older transfer while letting the newer one run. The matrix core chews on tile *n* while the engine fetches *n+2*. Per-iteration overhead in the wave is two scalar adds and one instruction issue, and the register file holds accumulators instead of data in flight. Deeper pipelines are a question of more LDS buffers and a larger wait threshold; the 6-bit counter won't be what stops you.

Compare that against the same loop written with per-lane async copies, where the address arithmetic, the VGPRs holding those addresses, and the instruction count all scale with tile size. Here they're constant. That's the structural argument for TDM, and it holds whatever the achieved bandwidth turns out to be.

### The sharp edges

A collection from across §10.11, roughly in the order they'll surprise you.

Faults are imprecise and non-fatal. MEMVIOL gets reported if a tile address falls outside the wave's LDS allocation (and only when LDS_CONFIG.ADDR_OUT_OF_RANGE_REPORTING == 1), or if the global address lands outside the global aperture, meaning scratch, LDS or the hole. But:

> MEMVIOL does not cause the TDM to abort — it continues processing the entire descriptor and return MEMVIOL to the wave when it's done. Memviol and XNACK-error is reported once for the entire TDM operation, not per data transfer.

A faulting descriptor runs to completion and you learn that something went wrong, once, with no indication of where. Debugging a malformed descriptor is going to be an exercise in bisection.

XNACK is handled internally, mostly. TDM absorbs XNACK-retry itself, so page faults on the DMA path get resolved without the wave's involvement. XNACK-error can still surface, for a write to a read-only page for instance, and it's explicitly imprecise: "the wave cannot rewind the PC to the faulting instructions."

Reserved fields aren't decorative. Group 0 carries five fields (count, is_restore, is_store, nv and User_null) all marked "User: must set to zero," and they exist because the same descriptor format is reused by the context save/restore machinery. is_restore = 1 lets the restore path override the instruction's own load/store direction. Garbage in those bits doesn't give you a bad tile, it gives you a differently-typed operation. Zero them explicitly instead of trusting whatever the SGPRs held. Relatedly, the type field in bits [127:126] must be 2 ("image"), so a descriptor built from zeroed registers isn't a valid descriptor at all.

Two fields silently disable the whole transfer. tile_dim0 == 0 makes the tensor a NOP, and count == 0 means "NULL tensor, no memory is copied, no atomic_barrier sent." The second one is worse, because it also never signals the barrier your consumers are asleep on.

Unused dimensions are zero while used ones start at one. tile_dim1 through tile_dim4 use 0 for "dimension unused" and 1..Max for a real extent, so there's no way to express a legitimately empty higher dimension.

Context switching knows about in-flight DMA. There's a message, RTN_SAVE_WAVE_HAS_TDM, whose job is to "inform the CWSR machine this wave needs to be saved and has outstanding TDM ops," and the group-0 count field is documented with separate user and context-restore meanings. VMID or pipeline kill stops all descriptors for the affected waves, and killed ops still return "done" so TENSORcnt drains, so a wave being torn down won't deadlock on a wait.

---

## What I'd want to know next

The descriptor is the API, and it's denser than it looks. Four mode bits (iterate_enable, Gather Mode, pad_enable, and a non-zero workgroup_mask) reconfigure what the other fields mean, sometimes overloading them outright. That's a lot of expressive power packed into 20 SGPRs, and also a lot of ways to build something syntactically fine and semantically wrong. If I were wrapping this, I'd want the mutually exclusive combinations encoded in the type system rather than a struct with twenty public fields and a comment.

The out-of-bounds handling surprised me most. Zero-fill on read, drop on write, checked against real tensor bounds is exactly right for ragged GEMM tiles, and it deletes a whole category of epilogue code. I went in expecting the multi-dimensional walk to be the headline feature and came out thinking the bounds semantics were more useful. The one place it's conditional, unsorted gather indices, is at least flagged clearly enough to design around.

And the counter matters as much as the engine. A copy engine you had to fully drain before computing would be a modest instruction-count win. TENSORcnt with a <= threshold, plus barrier-arrive-on-completion, is what lets you keep several transfers in flight, order them against each other, and hand the completion signal to hardware that wakes sleeping consumers.

Whether any of it delivers is a hardware question. Given time on a CDNA5 part I'd measure two things first: descriptor setup cost against tile size, to find where a single TDM instruction starts beating a hand-rolled async-copy loop, and what multicast is really worth on a GEMM whose A-panel is shared across a full 16-workgroup cluster.

---

**References.** All quotations and field tables come from AMD's *"CDNA5" Instruction Set Architecture Reference Guide*, principally §10.11 (Tensor Data Mover Instructions), with supporting material from §1.2.2 (Cache System Hierarchy), §2.3 (Workgroup Clusters), §3.2.2 (EXECute Mask), §5.7 (Data Dependency Resolution), §10.7 (Multicast Load), §10.8 (Asynchronous Memory Load and Store), §10.9 (WMMA Matrix Load Ops with Transpose), and §11.1–11.2 (Local Data Share Operations).
