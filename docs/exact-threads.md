# Exact parallel launches (`VGPU_EXACT_THREADS`)

`VGPU_THREADS=1` runs the blocks of a launch one after another, in order, on one host thread. It is the setting whose results are a
function of the program alone: the reference runs of `tools/verify`, the counter checks of `tools/bench` and the result cache
(`VGPU_MEMO`, for kernels with atomics) all ask for it. Several host threads (`VGPU_THREADS=N`, the default) are faster and give the
same answer for a kernel whose blocks are independent, but nothing checks that they are: blocks that share an atomic, or race, give
one of several answers.

`VGPU_EXACT_THREADS=N`, set beside `VGPU_THREADS=1`, keeps the promise of the first and takes the speed of the second where it can
prove the two are the same. It is opt-in and off by default; without it nothing changes. It applies to the SASS engine (the default
for NVIDIA kernels). Launches of the PTX interpreter and of the AMD engine run in order as before.

## What is promised

With `VGPU_THREADS=1 VGPU_EXACT_THREADS=N`, for every launch:

* device memory after the launch,
* the launch's counters (`LaunchStats`: every counter, the per-class and per-opcode ones too; `VGPU_COUNTERS`, `VGPU_COUNTERS_FILE`,
  `vgpu ncu`),
* the error of a launch that fails, and the memory it leaves behind,

are those of `VGPU_THREADS=1` alone. The exceptions, both invisible to a kernel and to a copy back to the host, are listed under
"What is not the same".

## How

A launch that qualifies (below) is run on trial: `N` host threads take its blocks from the same hand-out the other launches use, and
while the trial lasts the memory manager keeps a shadow of the device's address range (`src/core/memory_spec.cpp`). One 32-bit word
stands for four bytes of device memory and holds one of four states:

| state | meaning |
|---|---|
| 0 | no block has touched the bytes |
| `b + 1` | block `b` has read them; no other block has touched them |
| all ones | more than one block has read them; none has written them |
| top bit and `b + 1` | block `b` has written them (and may have read them); no other block has touched them |

A word only moves down that table (untouched, one reader, several readers; or untouched or its own reader mark to its writer), by
compare-and-swap, and a block makes an access only after the word allows it:

* a read of a word another block wrote, and a write of a word another block read or wrote, stop the trial before the access is
  made (`MemoryManager::SpecStop`, thrown from the access);
* the first write of a block to a word copies the bytes that are there into the block's undo log, then writes.

Every method of the memory manager that reads or writes device bytes makes this note first (`spec_read`, `spec_write` and their
batch forms in `src/core/memory.cpp`; with no trial on, each is one null test). The methods that hand out a pointer for the caller
to use later (`batch_window`, `atomic_word_ptrs`) decline during a trial, and the caller takes the path that is noted.

When the workers are done:

* **no stop:** the trial is kept. The shadow is cleared, the undo logs are dropped, the workers' counters are added up.
* **a stop** (a conflict, or any exception in any block: a fault, a step limit, a trap): every undo log is written back, the shadow
  is cleared, and the launch runs again from block 0 in order on one thread, exactly as without the variable. What it reports is
  what the run in order reports.

### Why a kept trial is the run in order

Take two blocks and one byte. If both only read it, neither can tell. If one writes it and the other reads or writes it, both go
through the byte's word, and whichever comes second finds a state it may not enter (a writer finds another block's mark or the
several-readers mark; a reader finds another block's writer mark) and stops the trial before touching the byte. So in a trial that
was not stopped, no byte was written by one block and touched by another. Each block therefore read, for every byte, either what
the launch started with or what the block itself stored, and that is also what it reads when the blocks run in order: a byte it
reads was written by no other block, before it or after it. A block's execution is a function of what it reads (its registers,
shared and local memory are its own, and its warps are scheduled inside the block as always), so every block ran as it does in
order, stored the same values to bytes no other block stored to, and counted the same counts. The sums of the counters are sums of
integers, and do not depend on the order of adding.

The other ways a block could see the order of blocks are closed one by one:

| channel | what the trial does |
|---|---|
| atomics on device memory | noted as a write of the word: two blocks on one word stop the trial; a block's atomics on words only it touches have one answer |
| `%smid` (SR_VIRTUALSMID) on a grid larger than the device: the number is the host thread's | stops the trial (`Runner::sreg`) |
| device `printf`, `assert`, `malloc`/`free`, any device runtime call | not tried: a module with any extern runs in order |
| a kernel that launches kernels | not tried |
| a kernel that polls memory (lockstep), cooperative launches, clusters (and `clusterlaunchcontrol.try_cancel`, which takes the next block from the hand-out) | not tried: these keep their own schedules |
| textures and surfaces | not tried |
| pinned, managed or registered host memory | reads are allowed (no block can write it: a store to host memory stops the trial); the host itself writing it during a launch is as visible in order |
| virtual-memory reservations, mappings and handles, memory shared with another process, file-backed allocations | not tried: the same bytes may be mapped at two addresses, which a shadow by address cannot see |
| an armed fault (`vgpu fault arm`), the host-GPU card gate, the result cache, a memory trace | not tried |
| `VGPU_SASS_TRACE`, `VGPU_TRACE`, `VGPU_RACE`, `VGPU_TIER_STATS` | not tried: their output is in the order things ran |
| an allocation, a free or a device-to-device copy during the launch | stops the trial |
| the warp clocks (`SR_CLOCK*`) | a count of the warp's own instructions: the same in any order |

### Why a stopped trial leaves nothing behind

A block writes a byte only after its word is the block's own writer mark, and it copies the old bytes of the word between taking
the mark and the first store. No other block writes a word that carries the mark, so what was copied is what the launch started
with, and no byte the trial wrote is outside a word some block marked and copied. Writing the copies back, in any order, gives
the memory of before the launch. Bytes of a marked word that lie outside every allocation (the tail of an allocation whose size is
no multiple of four) are not copied and cannot be written.

Every block of a trial sees a state of memory that could be real (the launch's start plus its own stores), so a block of a trial
that will be stopped still runs on consistent data. It may run longer than it does in order, or wait for a store that never comes;
each worker therefore looks at the stop flag after every round of a block's warps (at most 4096 instructions a warp) and at every
access. A block that waits on an earlier block's store makes the conflict itself: its read and the other's write meet at one word.

### A kernel that keeps being stopped

A stopped trial costs the work done before the stop. A kernel whose trial was stopped is run in order for the next launch, then
tried again; after each further stop it waits twice as long (2, 4, ... 4096 launches), and a kept trial puts it back. Which
launches are tried moves time only.

The width rule of ordinary launches (`LaunchWidth`, `src/exec/worker_pool.cpp`) applies too: a launch too small to pay for a
hand-off to the pool stays on one thread and is not tried.

## What is not the same

* **How memory is held, not what it holds.** A stopped trial may leave a 64 KiB chunk materialized that the run in order would have
  left untouched or uniform (a block of the trial stored to it and the store was undone). Every byte reads the same; the host
  memory the simulator uses can be higher.
* **Time**, and what is derived from it (the pool's width rule, the telemetry's utilisation).
* **The order of the lines of `VGPU_EXACT_LOG`** relative to other output.

## Cost

While a trial runs, each access of a warp to global memory makes one shadow lookup per lane on top of the access (a load of the
word and a compare when the block has been there; a compare-and-swap, and for a store a copy of the old bytes, the first time).
The shadow takes 4 KiB of host memory per 4 KiB of device memory the launch touches and is cleared after each launch; the undo
log holds the bytes the launch wrote for the first time, at most 1 GiB per worker (more stops the trial). Measurements: Round sp47
in [the performance log](perf-design/rounds/README.md).

## Seeing what happened

`VGPU_EXACT_LOG=<file>` appends a line per launch the variable applied to: kernel, blocks, threads, then `committed`, `stopped`
or `in order`, the reason, the bytes kept for undoing and the 4 KiB pages touched. `sass::exact_counts()` gives the process's
totals to a test.

## Tests

`tests/unit/test_exact_threads.cpp` (also run with `VGPU_TIER=0` and `VGPU_FASTPATH=0`, which send the accesses through other
methods of the memory manager): kernels with independent blocks, blocks that store to the same words, a chain in which each block
reads what the one before stored, a fault in block 1, and `%smid` on a grid larger than the device, each compared word for word and
counter for counter with the run in order at 2, 3, 4 and 8 threads; every reading and writing method of `MemoryManager` against the
shadow; a stopped trial undone to the byte across chunk ends and at an allocation's tail.
