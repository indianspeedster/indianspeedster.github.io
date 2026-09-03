---
title: "Inside the CDNA5 Tensor Data Mover"
description: "CDNA5 adds a small DMA engine that copies tiles of a tensor from global memory straight into LDS, without the shader core computing a single address. What it is, how you drive it, and why it lets the copy overlap the math."
date: 2026-09-02
tags: ["GPU", "AMD", "CDNA5", "kernels", "TDM", "LDS", "GEMM"]
draft: false
---

Most of a matrix-multiply kernel isn't multiplication. Before the hardware can multiply anything, someone has to fetch the numbers: compute an address for every thread, issue a load, wait for it to come back, copy it into fast local memory, wait at a barrier so everyone agrees it's there. Only then does the math happen. Count the instructions in a real GEMM kernel sometime. The arithmetic is a small fraction of them.

CDNA5 moves that fetching into dedicated hardware called the Tensor Data Mover. You describe a chunk of memory you want, hand the description to the engine, and it goes and gets it while your kernel keeps running.

Some vocabulary first, since the rest of this leans on it:

| Term | What it means |
|---|---|
| Wave | A group of 32 threads that execute together in lockstep. The unit of work on the GPU. |
| VGPR | A vector register. Each of the 32 threads gets its own copy. These are scarce. |
| SGPR | A scalar register. One copy shared by the whole wave. Much cheaper than a VGPR. |
| LDS | Local Data Share, a small fast scratchpad that all threads in a workgroup can read. |
| WGP | Workgroup Processor, the hardware block holding 4 SIMDs and one LDS. |
| Tile | The rectangular chunk of a big matrix that a kernel works on at a time. |

> **TL;DR.** The Tensor Data Mover is a DMA engine that copies a tile of an array from global memory into LDS. You set up a descriptor in about a dozen SGPRs, issue one instruction, and it does the whole copy on its own while your wave gets on with other work. It handles arrays of up to 5 dimensions, clamps reads and writes that run off the edge, can insert padding to avoid LDS bank conflicts, and can broadcast one tile to as many as 16 workgroups at once. A counter called TENSORcnt tells you when it has finished. The point of all of it: the copy for the next tile happens while the matrix core is still working on this one.

> **This is a documentation read, not a benchmark.** Everything here comes from AMD's published CDNA5 ISA manual. I haven't run any of it on hardware. Where I say something should be fast, I mean the architecture is built that way, not that I measured it. Nothing here is endorsed or reviewed by AMD.

---

## Why bother with a DMA engine

There are three ways to get data from global memory into LDS on CDNA5, and they're worth seeing side by side.

![Three columns comparing the VGPR round trip, async global-to-LDS copies and the Tensor Data Mover, each showing the path from global memory down to LDS and the instruction, register and counter cost.](/blog/cdna5-tdm/tdm_01_three_paths.svg "The three paths differ in where the data stops on its way to LDS. The VGPR round trip parks it in the register file and pays a second instruction to push it out again. The async copy skips the register file but still computes one address per lane. TDM takes neither route: the shape of the transfer lives in a descriptor, so the instruction count stops scaling with the tile.")

The old way is a round trip through registers: load from memory into VGPRs, wait, then store from VGPRs into LDS. It works, but the data takes a detour through the register file, and those are the registers your accumulator wants.

The async loads added in recent generations skip the register file, sending data straight from memory into LDS. Better, but every thread still computes its own address, so one instruction only moves 32 lanes' worth of data.

The Tensor Data Mover does neither. From the manual:

> Tensor instructions are encoded as VIMAGE or VGLOBAL instructions but use no VGPRs and are not performed per-lane.

No VGPRs at all, and nothing computed per thread. One instruction moves the entire tile, however big you said it was. Think of it less as a load and more as handing a work order to a helper that runs alongside you.

| | Register round trip | Async to LDS | Tensor Data Mover |
|---|---|---|---|
| Instructions per tile | 2 per chunk, plus waits | 1 per chunk | 1 total |
| VGPRs used | data and addresses | addresses only | none |
| Who computes addresses | every thread | every thread | the engine |
| Multi-dimensional | you write the loop | you write the loop | up to 5D in hardware |

If you know NVIDIA's Tensor Memory Accelerator on Hopper, this is the same idea with different details.

---

## Where it sits

![A workgroup processor containing two SIMD pairs, each with its own Tensor Data Mover beneath it, both writing into a shared LDS and WGP cache block, with a bidirectional path to global memory.](/blog/cdna5-tdm/tdm_02_where_it_lives.svg "Two SIMD pairs, two TDMs, one LDS. The engine sits between the SIMDs that issue descriptors and the LDS that receives tiles, and it reaches global memory through the ordinary cache hierarchy.")

There's one engine for every pair of SIMDs, so two per Workgroup Processor. They write into the LDS, which on CDNA5 is 320 KB split into 64 banks of 4 bytes each. Hold on to that bank number, it explains a feature further down.

---

## Driving it

Two instructions, one in each direction:

```asm
TENSOR_LOAD_TO_LDS      ; global memory -> LDS
TENSOR_STORE_FROM_LDS   ; LDS -> global memory
```

Neither takes the data as an operand. Instead they point at a descriptor, called the D#, which is just a block of SGPRs you fill in beforehand. A 2D copy needs 12 of them, the full 5D version needs 20.

![The four descriptor groups laid out as columns, group 0 with four SGPRs, group 1 with eight, groups 2 and 3 with four each, listing their fields, plus callouts showing how iterate_enable and gather_mode reinterpret groups 2 and 3.](/blog/cdna5-tdm/tdm_03_descriptor.svg "The D# in full. The two callouts on the right are what makes this a tagged union rather than a struct: turning on iteration or gather quietly changes what the group 2 and 3 registers mean.")

Don't try to memorize that. What's worth noticing are the two callouts on the right: a couple of mode bits change what the other registers mean. Turn on iteration or gather and the registers you were using for dimensions now hold something else entirely. It behaves more like a tagged union than a plain struct, which is easy to get wrong when you're filling it in by hand.

The instruction returns immediately, so you need some way to ask whether the copy has finished. That's a counter called TENSORcnt. It goes up by one when you issue a transfer and down by one when a transfer lands, and S_WAIT_TENSORCNT N blocks until at most N are still outstanding.

That N is the whole trick. Waiting for 0 means "wait for everything." Waiting for 1 with two copies in flight means "wait for the older one, let the newer one keep running." That's what lets you build a pipeline instead of a stall.

One rule to remember: TENSORcnt only orders these transfers against each other. If an ordinary store has to land before a tensor load reads that memory, wait on the store's own counter first.

---

## Describing what to copy

This is the part that trips people up, so it's worth going slowly.

The descriptor names two different things. The tensor is the whole array as it sits in memory. The tile is the smaller rectangle you actually want copied.

![A grid representing a 2D tensor with two shaded padding columns on the right, a blue tile highlighted inside it, a red dot marking where global_addr points, and brackets labelling tensor_dim0, tensor_dim0_stride, tile_dim0 and tile_dim1.](/blog/cdna5-tdm/tdm_04_tensor_tile.svg "tensor_dim0 is how much real data a row holds; tensor_dim0_stride is how far apart rows actually are, which is how padded and sub-region layouts get described. The red dot is the field that moves as you step the tile.")

Two ideas in that picture do most of the work.

First, a row's length and a row's stride are separate numbers. Length is how much real data a row holds. Stride is how far apart consecutive rows start. Letting the stride be larger is how you describe a tile carved out of a bigger matrix, or an array with gaps between its rows.

Second, the starting address points at the tile, not at the array. That catches everyone, and the manual calls it out explicitly:

> global_addr — Global memory address of the start of the tile within the tensor (not the start of the tensor).

The nice consequence is that walking a tile across a matrix means updating exactly one field. Everything else stays as it is, which is why you can set the descriptor up once outside your loop.

For a 2D array the engine computes each address like this:

```c
address = global_addr + elementSize * (x + y * row_stride)
```

Rows get read left to right and stacked into LDS back to back. However scattered the data was in memory, it arrives packed.

---

## Running off the edge

Real matrices aren't neat multiples of your tile size. The last tile in a row hangs off the end of the array, and normally you deal with that using a separate cleanup kernel or a slower guarded path.

![Two panels, each showing a tile straddling the right and bottom edge of a tensor. On the load side the cells outside the tensor are filled with zeros; on the store side they are crossed out to show the writes being dropped.](/blog/cdna5-tdm/tdm_05_out_of_bounds.svg "The same edge tile under a load and under a store. Neither faults, neither needs a predicated slow path, and neither can touch memory belonging to a neighbouring tile.")

The engine just handles it. Reads past the end of the array return zero, and writes past the end are thrown away. So you issue the same descriptor for the edge tile as for every other tile, the overhang comes back as zeros, and zeros contribute nothing to a sum. Going the other way, a full tile stored over a partial region can't scribble on its neighbour.

Two things to know. It only protects the far end, not a negative address. And the check uses the array dimensions you put in the descriptor, so if you set those equal to the tile size for convenience, you've switched the whole thing off.

---

## Padding, and why LDS has banks

LDS is split into 64 banks. Roughly: two threads reading different banks at the same time are free, two threads reading the same bank have to take turns.

That causes a classic problem. If a tile's rows are packed tightly and the row length lines up badly with 64, every row starts in the same bank. Reading down a column then hits that one bank over and over, and the reads serialize.

![Two panels of five LDS rows with bank numbers in each cell. Without padding every row starts at bank 0, highlighted as a red column. With one DWORD of padding per row the starting banks step 0, 1, 2, 3, 4 down a green diagonal.](/blog/cdna5-tdm/tdm_06_lds_padding.svg "Why padding exists. Densely packed rows put every row start in the same bank, so a column access serializes 64 ways. One DWORD of skew per row spreads those starts across banks, and the copy engine inserts it mid-transfer at no cost.")

The standard fix is to waste a few bytes at the end of each row so the next one starts somewhere else. By hand that means extra address arithmetic on every store. The Tensor Data Mover will do it for you mid-copy: tell it how often to insert a gap and how big the gap should be, and it skips those slots as it writes.

Two warnings. The settings are encoded rather than literal, so writing 1 doesn't get you 1, and they must both be zero or both be non-zero. And padding only works on the way in, since there's no un-padding on the way back out.

---

## Two extra tricks

**Broadcast.** In a large matrix multiply, many workgroups need the same slice of data, and reading it separately for each one wastes bandwidth.

![A single global memory read entering the TDM, which fans the same tile out to the LDS of four workgroups inside a dashed cluster boundary, annotated as scaling to sixteen.](/blog/cdna5-tdm/tdm_07_multicast.svg "One read of the shared operand, N copies delivered. The mask in the descriptor picks which workgroups of the cluster receive it, and the engine switches from GLOBAL_LOAD_ASYNC to CLUSTER_LOAD_ASYNC to do it.")

Set a 16-bit mask in the descriptor and one trip to memory delivers the tile into the LDS of up to 16 workgroups at once. They have to be part of the same cluster and they have to ask at roughly the same time; the hardware waits briefly for stragglers and then gives up on them.

**Gather.** Flip a mode bit and part of the descriptor stops holding dimensions and starts holding a list of row numbers.

![A 2D tensor with four non-adjacent rows highlighted, an index list held in descriptor groups 2 and 3, and those four rows landing compacted and adjacent in LDS.](/blog/cdna5-tdm/tdm_08_gather.svg "Gather mode spends the group 2 and 3 registers on row indices instead of tensor dimensions. Scattered rows in memory arrive contiguous in LDS, in the order the index list names them.")

Up to 16 scattered rows get pulled together into LDS in one instruction, which is exactly the shape of an embedding lookup or a mixture-of-experts gather. One catch: the out-of-bounds protection from earlier is only guaranteed when the indices are in non-decreasing order. Since gather exists for cases where indices come from data, either sort them or check them yourself.

There's a third mode, iteration, which replays one descriptor several times to pull out every Nth row. Useful, though it borrows registers from the higher dimensions, so it can't be combined with a 4D or 5D copy.

---

## The payoff

Here is what all of it is for.

![A timeline with a TDM lane showing four back-to-back tile loads and a matrix core lane showing three compute blocks, each compute block starting one slot after its load and separated by an S_WAIT_TENSORCNT 1 marker.](/blog/cdna5-tdm/tdm_09_pipeline.svg "The steady state. Each compute block runs against a tile that landed an iteration ago while the engine is already two tiles ahead, and the only synchronization is a counter threshold.")

Start two copies before the loop. Inside the loop, wait for the older one, do the math on the tile that just arrived, bump the address, and fire off another copy. The engine stays a tile or two ahead of the matrix core.

```asm
loop:
  S_WAIT_TENSORCNT 1                   ; tile n has landed; n+1 still moving
  ...  compute on the tile that arrived ...
  s_add_u32   global_addr_lo, ...      ; point the descriptor at the next tile
  TENSOR_LOAD_TO_LDS  D#               ; start fetching tile n+2
  s_cbranch   loop
```

Per iteration the wave does one address update and issues one instruction. No per-thread address math, and no registers tied up holding data in transit. The copy is effectively free as long as the math takes longer than the fetch.

There's one more piece for the common case where one wave fetches and many waves compute. The descriptor can ask the engine to signal an LDS barrier when the copy finishes. Consumers sleep on that barrier and get woken by hardware, so nobody spins on a flag.

---

## Things that will bite you

- A descriptor field left at zero can silently turn the whole transfer into a no-op, including the completion signal other waves are waiting on.
- Several fields are encoded rather than literal, so the number you write isn't the number you get.
- Errors are reported once, at the end, with no indication of which part of the transfer went wrong. Debugging a bad descriptor means bisecting it.
- Some bits exist for the hardware's own context-switching machinery and must be zeroed explicitly. Leaving junk there doesn't corrupt the data, it changes which operation you asked for.

---

## Worth knowing

Two things stand out after reading all this.

The out-of-bounds behaviour is more useful than it first looks. Zero on read, drop on write, checked against the real array bounds is exactly the right answer for ragged matrix edges, and it deletes a whole category of cleanup code that GEMM kernels otherwise carry.

And the counter matters as much as the engine does. A copy engine you had to fully drain before computing would only save you some instructions. Being able to say "wait until only one is still outstanding" is what turns it into a pipeline.

Whether it delivers in practice is a hardware question, and I haven't had a CDNA5 part to try it on. The first thing I'd measure is how big a tile has to be before one of these instructions beats a hand-written copy loop.

---

## Appendix: the details this skipped

Everything above is the working mental model. This part is the reference material you need if you're actually filling in a descriptor by hand, pulled from §10.11 of the manual and its neighbours.

### Instruction encoding

Both tensor instructions use the VIMAGE encoding, with most of its fields either repurposed or forced to a constant. The ones that matter are the four VADDR slots, which hold **SGPR** addresses despite the name.

| Field | Meaning |
|---|---|
| VADDR0 | SGPR base of a 4-SGPR block: D# group 0 |
| VADDR1 | SGPR base of an 8-SGPR block: D# group 1 |
| VADDR2 | SGPR base of a 4-SGPR block: D# group 2, or NULL |
| VADDR3 | SGPR base of a 4-SGPR block: D# group 3, or NULL |
| VADDR4 | unused, set to NULL (0x7C) |
| SCOPE, TH, NV | memory scope, temporal hint, non-volatile |

That's 4 + 8 = 12 SGPRs for a 2D copy and 20 for the full 5D form. VADDR2 and VADDR3 must both be NULL or both be set; you cannot supply group 2 without group 3. Passing NULL behaves as though you had pointed at zeroed SGPRs, and pointing both at the same block is legal if you want group 3 to duplicate group 2.

Tensor instructions may not appear in a clause. If your assembler groups memory instructions into clauses for issue efficiency, these have to sit outside them.

### Address generation

Element size lives in data_size, log2-encoded in two bits: 0 is 1 byte, 1 is 2, 2 is 4, 3 is 8. The manual's pseudocode decodes it as `dataSize = (1 << D#.data_size)`, while the address formulas a page earlier write `D#.data_size *` as though the field held a byte count. Take the pseudocode as authoritative.

```c
global_addr[2d] = D#.global_addr + dataSize * (x + y * D#.tensor_dim0_stride)

global_addr[3d] = D#.global_addr + dataSize * (x
                + y * D#.tensor_dim0_stride
                + z * D#.tensor_dim1_stride)

global_addr[4d] = D#.global_addr + dataSize * (x
                + y * D#.tensor_dim0_stride
                + z * D#.tensor_dim1_stride
                + zz * D#.tensor_dim2_stride)
```

Strides are **cumulative**, not per-dimension increments you multiply together. tensor_dim1_stride already contains whatever tensor_dim0_stride contributed, so if your instinct is to write `z * dim0 * dim1`, that multiplication isn't happening in hardware. You supply the product.

The stride fields are 48 bits wide in elements, against a 57-bit global_addr, so neither will be your limit.

The manual's own pseudocode for a 3D load, including the padding step:

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

Two things to read out of it. There is no LDS-side stride field at all, so padding is the only way to get a non-contiguous destination layout. And dimension 0 is contiguous in both spaces, which is where your coalescing lives, so it wants to be the fast-varying dimension of the memory layout.

Those loop bounds bother me slightly: `for X = 0..D#.tile_dim0` reads as tile_dim0 + 1 iterations, which contradicts tile_dim0 being "tile dimension 0 in elements." I take it as loose pseudocode for a half-open range, but it's the kind of off-by-one worth confirming against silicon.

### Padding encodings

| Field | Bits | Encoding |
|---|---|---|
| pad_enable | 1 | off / on |
| pad_interval | 3 | power of two: 0 is 2 DWORDs, 1 is 4, up to 7 meaning 256 |
| pad_amount | 7 | offset by one: 0 is 1 DWORD, 1 is 2, up to 127 meaning 128 |

Constraints:

- pad_amount and pad_interval must **both** be zero or **both** be non-zero. Anything else "produce[s] unpredictable results," which is neither zero padding nor an error. Since zero in each field means one and two DWORDs respectively, the setting you would naturally reach for as "smallest possible padding" is exactly the illegal one unless padding is switched off entirely.
- Padding is expressed in DWORDs while the rest of the descriptor works in data_size elements. For an FP8 tile, one unit of padding is four elements.
- tile_dim0 must be a multiple of 4 bytes for padding to work properly.
- Padding that would run past the workgroup's LDS allocation is ignored rather than faulting, so an over-padded descriptor degrades into a differently laid out tile instead of an error you would notice.

The second use for padding, beyond bank conflicts, is transposes: the manual describes it as laying data out "in a pattern that may be easier for matrix transpose." CDNA5 has DS_LOAD_TR16_B128 and friends (§11.2.4) to transpose on the way from LDS into VGPRs, and choosing a padding stride that makes those loads conflict-free costs nothing at copy time.

### Completion and ordering

TENSORcnt is 6 bits, so up to 63 outstanding transfers per wave. The manual's three ordering guarantees:

> Tensor-Done is returned once per instruction, not per memory transfer. Tensor instructions complete in-order with other Tensor instructions from the same wave (both loads and stores stay in-order), but are unordered with instructions from other waves. Tensor instructions are unordered with respect to other types of memory instructions.

Completion is per-instruction, so the engine may split a tile into a hundred memory transactions and you still see exactly one decrement once all of them land. There is no partial completion to observe.

Same-wave transfers are FIFO, loads and stores together, which is stronger than ASYNCcnt gives you (there, loads and stores complete out of order relative to each other).

The third is the trap: TDM is unordered against every other memory type. A GLOBAL_STORE issued before a TENSOR_LOAD_TO_LDS from the same wave has no defined ordering against it.

One side effect worth knowing: §3.2.2.1 lets the hardware skip vector memory instructions when EXEC == 0, but only when LOADcnt, STOREcnt, ASYNCcnt and TENSORcnt are all zero. An outstanding transfer suppresses that optimization for the whole wave.

### The barrier arrive

Set atomic_barrier_enable and supply atomic_barrier_address, a 64-bit-aligned LDS location stored as bits [18:3] with the low three implied.

> A tensor op may optionally request that an LDS atomic occur after the tensor completes by setting D#.atomic_barrier_enable. When set, after a tensor operation completes it sends DS_ATOMIC_ASYNC_BARRIER_ARRIVE to signal its completion.

CDNA5's LDS barriers (§11.2.2) are split-phase counting barriers with a pending count and a phase. When the count rolls under, hardware wakes sleeping waves in the workgroup and reloads the count. The arrive is also how the engine keeps successive descriptors from one wave ordered against each other at the LDS end.

### Mode details

**Multicast.** workgroup_mask is 16 bits, one per workgroup-in-cluster. It applies to loads only; for TENSOR_STORE_FROM_LDS the field is ignored. It must be zero when the wave is not in a cluster, which the manual states twice, suggesting a real correctness hazard rather than a benign no-op. The underlying mechanism is a rendezvous with a timeout: every masked workgroup is expected to make the same request, and data returns once they all have or the timeout fires, with late arrivals getting their own separate broadcast. early_timeout tells the GL1 to return data as soon as GL2 supplies it, to whoever has shown up. Multicast loads force-miss the WGP$ regardless of scope and temporal hints.

**Iteration.** The extra fields are global_addr_increment and lds_addr_increment, both in elements, plus iterate_count where 0 means iterate once and 255 means 256 times. 2D and 3D only. Turning it on overloads three group-2 fields: tensor_dim3 becomes lds_addr_increment, tensor_dim2_stride becomes global_addr_increment, and tile_dim3 becomes iterate_count. Those are exactly the fields a 4D or 5D tensor needs, which is why the two can't be combined. iterate_enable is ignored outright when gather mode is on.

**Gather.** Groups 2 and 3 become a list of row indices: 16 at 16 bits each, or 8 at 32 bits. tile_dim1 is redefined as how many indices are valid, and each row comes back as height 1 by width tile_dim0. tile_dim2 and tile_dim1_stride are ignored. Bounds checking against tensor_dim1 still happens, so an out-of-range index gives a zero-filled row rather than a fault. The same index list applied to a store gives you scatter. 2D only. And the caveat from the body, in the manual's own words:

> Random index selection is supported; indices are not required to be strictly increasing and may repeat. However, correct handling of out-of-bounds (OOB) accesses only occurs when each index is greater than or equal to the previous index.

### Faults and edge cases

MEMVIOL is reported if a tile address falls outside the wave's LDS allocation (and only when LDS_CONFIG.ADDR_OUT_OF_RANGE_REPORTING == 1), or if the global address lands outside the global aperture, meaning scratch, LDS or the hole.

> MEMVIOL does not cause the TDM to abort — it continues processing the entire descriptor and return MEMVIOL to the wave when it's done. Memviol and XNACK-error is reported once for the entire TDM operation, not per data transfer.

XNACK-retry is absorbed by the engine, so page faults on the DMA path resolve without the wave's involvement. XNACK-error can still surface, for a write to a read-only page for instance, and it is explicitly imprecise: "the wave cannot rewind the PC to the faulting instructions."

Group 0 carries five fields (count, is_restore, is_store, nv and User_null) all marked "User: must set to zero." They exist because the same descriptor format is reused by the context save and restore machinery, and is_restore = 1 lets the restore path override the instruction's own load/store direction. Junk in those bits doesn't give you a bad tile, it gives you a differently-typed operation. The type field in bits [127:126] must be 2 ("image"), so a descriptor built from zeroed registers isn't valid at all.

Two ways to silently do nothing: tile_dim0 == 0 makes the tensor a NOP, and count == 0 means "NULL tensor, no memory is copied, no atomic_barrier sent." The second is worse, since it also never signals the barrier your consumers are asleep on.

tile_dim1 through tile_dim4 use 0 for "dimension unused" and 1..Max for a real extent, so there's no way to express a legitimately empty higher dimension.

In-flight DMA is part of saved wave state. There's a message, RTN_SAVE_WAVE_HAS_TDM, to "inform the CWSR machine this wave needs to be saved and has outstanding TDM ops," and the group-0 count field carries separate user and context-restore meanings. VMID or pipeline kill stops all descriptors for the affected waves, and killed ops still return done so TENSORcnt drains, so a wave being torn down won't deadlock on a wait.


---

**References.** Quotations and field details come from AMD's *"CDNA5" Instruction Set Architecture Reference Guide*, mainly §10.11 (Tensor Data Mover Instructions), with supporting material from §2.3 (Workgroup Clusters), §5.7 (Data Dependency Resolution), §10.7 (Multicast Load), §10.8 (Asynchronous Memory Load and Store) and §11.1–11.2 (Local Data Share Operations).
