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
while the trial lasts the memory manager keeps a shadow of the device's address range (`src/core/memory_spec.cpp`). The shadow has
two grains.

**By word.** One 32-bit word stands for four bytes of device memory and holds one of four states:

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

**By line.** Blocks mostly keep to bytes of their own: a warp's 32 lanes one after another are 128 bytes, and nobody else comes
there. So the sixteen words of 64 bytes (a *line*) are not kept as sixteen marks while one block alone has touched them, but as
one 64-bit record:

| line record | meaning |
|---|---|
| 0 | no block has touched the line's sixteen words |
| `b + 1`, read bits, written bits | block `b` alone has touched them: a bit for each word it read, a bit for each word it wrote |
| by word | the sixteen words (above) say what was done to their bytes; final for the trial |
| busy | a block is writing the sixteen marks out from the record (for the length of sixteen stores) |

A record of block `b` is exactly sixteen word marks, each of them 0, `b + 1` or "written by `b`", written down in 64 bits. The
block moves it by one compare-and-swap that sets the bits of the words it is about to read or write, which is the word moves
"untouched to its reader" and "untouched or its own reader mark to its writer" made for several of its words at the same instant.
At its first write in the line the block copies the line's 64 bytes to its undo log, with the record "busy" for the length of
the copy, and only then sets its written bits; its later writes in the line keep nothing more. A second block that comes to
the line reads from the record which words the first read and which it wrote:

* if its access may not be made by the word rules (it reads a word whose written bit is set; it writes a word whose read or
  written bit is set), it stops the trial there, before the access;
* otherwise it takes the record to "busy" by compare-and-swap (which fails, and is tried again from a fresh look, if the first
  block noted more in between), stores the sixteen marks the record stood for into the line's words, and sets the record to
  "by word". From then on both blocks, and any other, go by the words of that line (and keep, word by word, the bytes of the
  words they take there). A block that finds "busy" waits: for the sixteen stores, or for the other block's copy of 64 bytes.

Nothing is noted that the words would not note, and nothing the words would note is left out, so a line never stops a trial by
itself and there is no "coarse" case for it. What it saves is the compare-and-swaps (one for sixteen words instead of sixteen, or
eight with the pairs of round sp55, which lines replace), host memory that is written (8 bytes of record per 64 bytes of device
memory instead of 64, the words being touched only in lines that two blocks came to), and pieces of the undo log (one of 64 bytes
for a line a block fills a word at a time, where there were sixteen of four).

**By page.** Most of what a kernel touches it only reads, and a compare-and-swap per word for bytes nobody writes is
the larger part of what a trial costs. So each 4 KiB page of the range has a state of its own, and a page is noted by word only
once a block writes in it:

| page state | meaning |
|---|---|
| 0 | no block has touched the page |
| `b + 1` | block `b` has read in it; no other block has touched it; nothing is noted by word |
| all ones | more than one block has read in it; none has written in it; nothing is noted by word |
| by word | the page's lines and words (above) say what was done to its bytes |
| by word, coarse | the same, after the page's only reader went on to write in it: its reads had not been noted by word, so every word of the page started as "read by that block" (every line a record of that block with all sixteen read bits) |

A page moves untouched, one reader, several readers by compare-and-swap, as a word does. It becomes "by word" in two ways only:
from untouched, by the first block that writes in it (or by the first block that touches it at all, where the trial's plan says so,
below); and from one reader, by that reader itself when it writes in the page, after it has put a page of words in place in which
every line is that block's record with every read bit set. "Several readers", "by word" and "by word, coarse" are final for the trial. A write is only
ever made in a page that is noted by word, after the line's record or the word allows it; so:

* a write in a page that another block has read by page state, or that several have, stops the trial before the write is made;
* a read in a page noted by word goes by the words.

Every method of the memory manager that reads or writes device bytes makes this note first (`spec_read`, `spec_write` and their
batch forms in `src/core/memory.cpp`; with no trial on, each is one null test). The methods that hand out a pointer for the caller
to use later (`batch_window`, `atomic_word_ptrs`) decline during a trial, and the caller takes the path that is noted.

Three memories a worker keeps for itself make the note cheaper without changing what is noted; none adds a state:

* **Pages a worker has seen.** Each worker remembers up to 256 pages its block may read with no further note: pages whose state
  was the block's own reader mark, valid for that block only, and pages with several readers, valid for the rest of the trial
  (that state is final). A read of such a page is one compare in the worker's own memory, with no atomic at all. The memory is
  asked the page and the block, so the next block on the same worker starts with nothing but the pages of several readers. (The
  place of a page in the memory is a hash of its number: a warp's lanes are often a fixed number of pages apart, the rows of a
  matrix, and the low bits of the page number sent them all to a few places.)
* **Runs of words a block has noted.** In a page noted by word, a worker remembers the last four runs of sixteen words or
  more (a warp's lanes one after another) that its block took as their writer (every word holds the block's writer mark, which nothing changes), and the last four it
  noted for reading (every word held the block's reader mark, its writer mark or the several-readers mark, and from those a
  word only moves to another of them). An access of the same block, of sixteen words or more, inside a remembered run has
  nothing to note: the store of `tmp[i] += ...` in a loop is one range compare instead of 32 loads and compares. Two runs of
  one block that touch are remembered as one; a run is asked the block. (One lane's few words neither enter the memory nor ask
  it: the runs of 32 lanes would push one another out, and the asking cost each of them.)
* **Pages of lines a worker has read in.** The worker remembers where the lines and words of 64 such pages are; a read whose
  record or words allow it already (looked at every time) needs no call and no compare-and-swap.

(Round sp55 also took two neighbouring words by one 64-bit compare-and-swap; the line record does that for sixteen, and the
words of a line that several blocks share are taken one at a time again.)

The undo log of a worker keeps its memory from one trial to the next (up to 64 MiB; a launch that writes much writes as much the
next time).

When the workers are done:

* **no stop:** the trial is kept. The shadow is cleared, the undo logs are dropped, the workers' counters are added up.
* **a stop** (a conflict, or any exception in any block: a fault, a step limit, a trap): every undo log is written back, the shadow
  is cleared, and the launch runs again from block 0 in order on one thread, exactly as without the variable. What it reports is
  what the run in order reports.

### Why a kept trial is the run in order

Take two blocks and one byte. If both only read it, neither can tell. If one writes it and the other reads or writes it, the
writer wrote in a page that was noted by word at that moment (no write is made in any other), and it held the byte's word as its
writer. The other block made its note either

* by the word, in which case both went through the byte's word, and whichever came second found a state it may not enter (a
  writer finds another block's mark or the several-readers mark; a reader finds another block's writer mark) and stopped the trial
  before touching the byte; or
* by the page's state (it can only have been reading). Then the page was in the state "read by that block" or "several readers"
  when it looked. From "several readers" a page never becomes "by word", so the writer can never have written in it. From "read by
  that block" only that block itself makes the page "by word", so the writer, a different block, can only have found the page
  already noted by word, which means the reader made it so after its read; but then the reader filled every word of the page with
  its own reader mark before the page's state changed, the writer found that mark on the byte's word, and stopped. And a block
  that looks at the worker's memory instead of the page's state relies on one of these two states having been seen earlier for
  the same block (or, for "several readers", for any block): both stay true of the page for as long as the memory is used, the
  first because a page read by `b` can only be changed to "several readers" (also fine for `b`) or to "by word" by `b` itself
  with `b`'s mark on every word.

The same holds for a block that goes by a remembered run of words instead of the words: a word in a run the block wrote holds the
block's writer mark for the rest of the trial; a word in a run it noted for reading held a mark that allows the block's read, and
every state such a word can move to still stops another block's write of it (another block's reader mark on it makes it "several
readers"; the block's own write makes it the block's).

And it holds for words kept as a line's record. Say "the mark of word `i`" for what the record says of it while the line is one
block's (0, `b + 1`, or written by `b`, from its two bits) and for the word itself once the line is by word. Then:

* *One block's moves are word moves.* The owner's compare-and-swap adds read bits (untouched to its reader, or nothing for a
  word it already holds) and written bits (untouched or its own reader mark to its writer) and takes none away; it succeeds
  only on the record it looked at, so each word moves from the mark that was judged.
* *Writing out changes no mark.* The block that writes a record out first takes it to "busy" by compare-and-swap from the very
  value it then writes out, so the owner cannot add a bit that is lost (its compare-and-swap fails on "busy", it waits, and
  goes by the words, where its earlier marks are). The words of a line are zero until then and are looked at by nobody before
  the record says "by word"; while it says "busy" every other block waits, and the block that holds it does sixteen stores and
  nothing that can throw or wait. So every access sees each word's mark either from the record or from the word, never half
  of one and half of the other.
* *The second block decides from the marks.* It stops where a mark forbids its access by the word rules and writes the record
  out where none does, then makes its own note in the words as any block does.

So a line is the note by word, written down differently while one block is alone in it, and the argument above (whichever of two
blocks came second to a word found a state it may not enter) goes through with "the mark of the word" in place of "the word".

So in a trial that was not stopped, no byte was written by one block and touched by another. Each block therefore read, for every
byte, either what the launch started with or what the block itself stored, and that is also what it reads when the blocks run in
order: a byte it reads was written by no other block, before it or after it. A block's execution is a function of what it reads
(its registers, shared and local memory are its own, and its warps are scheduled inside the block as always), so every block ran
as it does in order, stored the same values to bytes no other block stored to, and counted the same counts. The sums of the
counters are sums of integers, and do not depend on the order of adding.

The note by page says more was read than was (a whole page for a read of four bytes; every word of a page for a block that read
some of it and then wrote in it). More reading noted can only stop more trials, never keep one that the words would have stopped.
What it costs is dealt with under "A stop the page grain made".

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

In a line a block alone holds, the copy is of the whole line and is made once, at the block's first write in it. The record is
"busy" from before the copy to after it, and was untouched or the block's own until then: no other block had a mark in the line, so
none had written in it, and none can get a mark there (and so write there) while the copy is made. The copy is therefore what
the trial started with in all sixteen words, whichever of them the block goes on to write. A block that comes to the line later
and writes other words of it keeps those words by the word rule, which are again the bytes the trial started with; so every
piece that covers a byte holds the same value for it, and the order of putting back stays free. (More is copied than is written,
which is harmless: putting back bytes nobody changed changes nothing. Bytes of the line outside every allocation are not
copied.)

When a word in the middle of a store's
range is another block's, the trial stops with the words before it marked and not yet copied, and no byte of the store has been
written (the note comes before the store), so there is nothing of theirs to put back. A store inside a remembered run of the
block's own words keeps nothing: those words were kept when the block took them. The page states hold no bytes and are cleared
with the words.

Every block of a trial sees a state of memory that could be real (the launch's start plus its own stores), so a block of a trial
that will be stopped still runs on consistent data. It may run longer than it does in order, or wait for a store that never comes;
each worker therefore looks at the stop flag after every round of a block's warps (at most 4096 instructions a warp) and at every
access. A block that waits on an earlier block's store makes the conflict itself: its read and the other's write meet at one word,
or at the page the reader took.

### A stop the page grain made

A kernel that updates an array in place (`a[i] = f(a[i])`) reads and writes words of its own, and blocks share pages: by word
nothing conflicts, by page the first block's read takes the page and the next block's write in it is stopped. Such a stop is
marked *coarse*: a write in a page other blocks read by page state, or a write that met a reader's mark (one block's, or several
readers') on a word of a page that is "by word, coarse". A stop at a writer's mark is never coarse (writers' marks are exact), nor
is any stop of a read.

A trial with a coarse stop is undone like any other. The launch is then tried again, with the allocation the stop was in noted by
word from its first touch (`MemoryManager::SpecPlan`; pages elsewhere stay by page), and what was learned is kept for the kernel's
later launches (`Module::exact_plan`, by address range). After the third coarse stop in one launch, after one with nothing new to
learn, or with more than 32 ranges learned, the kernel is noted by word everywhere, which has no coarse stops. Only a trial that
stops for another reason, or commits, ends the launch's trials; what is run in order is run in order exactly as before. The
second trial changes time only: a trial that commits is the run in order whatever its plan, by the argument above.

What a coarse stop throws away is the work up to a block's first store in a page it read. With many blocks, or a store early in
a block, that is little. With a few blocks and a store at a block's end it is most of the launch, so a launch of fewer than 8
blocks is noted by word from the start and has no coarse stop.

`VGPU_EXACT_PAGES=0` notes every page by word from its first touch (the note the mode began with), for measurements and as the
fourth form of the test.

### The scout: the first block alone

A coarse stop finds one allocation out and throws the trial away. The first trial of a kernel does better where the grid has
blocks enough: it runs block 0 alone first, on the launching thread, under the shadow like any block of the trial, and only then
starts the others (`Runner::run_exact`, `MemoryManager::spec_scouted`). Between the two:

* the allocations in which block 0 read by page state and also wrote are added to what the trial, and the kernel's later
  trials, note by word from the first touch (all of them at once, where coarse stops find them one at a time);
* **block 0's notes are forgotten**: its page states, its lines, and what its worker remembered of them. Its undo log is kept.

Forgetting them is sound because of where block 0 stands in the order. The run in order runs block 0 first, alone, on the memory
the launch started with; the scout did exactly that. So after it the memory is the memory every later block finds in order, and
whether a later block reads or overwrites what block 0 read or wrote is no conflict at all: in order it does the same, after
block 0. The proof for the rest is the proof above with "the memory after block 0" in place of "what the launch started with":
the blocks 1 to n - 1 are noted against one another, and if none touched what another wrote, each ran as it does in order after
block 0. Block 0 itself ran in order by construction (and counted its counters, which are added to the launch's as any
worker's are).

A stopped trial must put back what the launch started with, and a later block may have written a word block 0 wrote. That block
kept what it found (block 0's store) and block 0 kept what the launch started with, so the pieces block 0 kept as the scout go back
last, after every other piece (among which the order stays free, as before). If block 0 itself fails or stops the trial, nothing
is learned or forgotten and the launch runs in order as after any stop.

The scout costs the launch one block's time on one thread instead of on all of them, once per kernel. That is left to launches of
at least 32 blocks per thread of the trial (about 3 % of such a launch at most); smaller grids learn by coarse stops as before.
`VGPU_EXACT_SCOUT=0` turns it off (for measurements).

### A kernel that keeps being stopped

A stopped trial costs the work done before the stop. A kernel whose trial was stopped is run in order for the next launch, then
tried again; after each further stop it waits twice as long (2, 4, ... 4096 launches), and a kept trial puts it back. Which
launches are tried moves time only.

The width rule of ordinary launches (`LaunchWidth`, `src/exec/worker_pool.cpp`) applies too: a launch too small to pay for a
hand-off to the pool stays on one thread and is not tried. For a trial the rule is stricter in two ways, because what a trial
competes with is the run in order without the shadow, and because a trial's launches scatter more (each sets the shadow up and
takes it down on every thread):

* **What a block costs is measured on one thread.** The rule's figure for a kernel comes from the time its threads spent in
  blocks, and under a trial that time holds the proof's cost: a small launch looked large enough for the pool, went to the pool,
  and was measured there again, for as long as the program ran. A figure that no launch on one thread has given now counts for
  half, a launch it does not carry to the full width runs in order (which measures it), and a figure one thread gave is not
  replaced by a pool's, unless the pool's is under three quarters of it (a block under the proof costs no less than a block in
  order, so the figure was too high: the kernel's first launch in order, a thread the host took away).
* **A small launch goes wide on trial only while most launches win.** The ordinary rule widens a small kernel for eight launches
  and keeps the width if the best of them took under 70 % of one thread's time. Under a trial the width is kept only if at least
  four of the seven counted launches took under 85 % of one thread's time and the seven together took no longer than one thread
  would have (a single launch counted as four times that at most); otherwise the kernel runs in order for 256 launches (twice
  as many after each further loss) and is tried again.

Both move time only: which launches are tried never changes a result.

## What is not the same

* **How memory is held, not what it holds.** A stopped trial may leave a 64 KiB chunk materialized that the run in order would have
  left untouched or uniform (a block of the trial stored to it and the store was undone). Every byte reads the same; the host
  memory the simulator uses can be higher.
* **Time**, and what is derived from it (the pool's width rule, the telemetry's utilisation).
* **The order of the lines of `VGPU_EXACT_LOG`** relative to other output.

## Cost

While a trial runs, each access of a warp to global memory makes one call of the note on top of the access. For a page the block
may read without a note (read by it, or by several blocks, and written by none) that is a compare in the worker's memory per page
the warp touches; for a page noted by word, a range compare when the block is remembered to have noted the words, else a load
and a compare per line (64 bytes) when the block has been there and a compare-and-swap per line the first time (per word in a
line two blocks share), and for a block's first store in a line a copy of the line's bytes. The page states take four bytes of host memory per 4 KiB
of device memory the launch touches; a page a block writes in (or that the plan notes by word) takes 4.5 KiB that is set aside
and 512 bytes of it that are written, plus 64 bytes for each line that two blocks came to; all are cleared after each trial. The undo log holds the bytes the launch wrote for the first time, at most 1 GiB per worker (more stops
the trial). A kernel with a coarse stop pays for the part of the trial before the stop once per learned allocation. Measurements:
Round sp47 and Round sp55 in [the performance log](perf-design/rounds/README.md).

## Seeing what happened

`VGPU_EXACT_LOG=<file>` appends a line per launch the variable applied to: kernel, blocks, threads, then `committed`, `stopped`
or `in order`, the reason, the bytes kept for undoing, the 4 KiB pages touched and the microseconds the trial took; a trial that was
stopped coarsely and tried again has a line of its own before it, with `again`. `VGPU_EXACT_STATS=<file>` appends a line per trial with what its notes cost
(`MemoryManager::SpecStats`): calls of the hooks, pages taken by one reader, by several, noted by word, made coarse, words looked at
and taken by compare-and-swap for reads and for writes, pieces and bytes kept, accesses answered by a remembered run, lines that a
second block came to (`lines_by_word`) and lines one block alone had touched at the end (`lines_one_block`).
`sass::exact_counts()` gives the process's totals to a test (`again`: the trials run a second time; `scouted`: the trials whose
first block ran alone first).

## Tests

`tests/unit/test_exact_threads.cpp` (also run with `VGPU_TIER=0` and `VGPU_FASTPATH=0`, which send the accesses through other
methods of the memory manager, and with `VGPU_EXACT_PAGES=0`): kernels with independent blocks, blocks that store to the same words,
a chain in which each block reads what the one before stored, a fault in block 1, a step budget that fails every block before its
store, and `%smid` on a grid larger than the device, each compared word for word and counter for counter with the run in order at 2,
3, 4 and 8 threads; every reading and writing method of `MemoryManager` against the shadow; a stopped trial undone to the byte across
chunk ends and at an allocation's tail.

For the note by page: a second block, on the same worker and on another, in a page that a first block claimed whole (as its reader;
as its reader and then a writer; with a second reader), at every byte of the first, a middle and the last word of the page, with
lengths of 1 to 8 bytes and accesses that end in the next page: stopped before the access wherever a note by word stops it, undone
to the byte, said to be coarse exactly where only the grain stopped it, and kept when run again with the allocation by word; the
worker's memory of a page not answering for the next block or for another page that shares its place (nor its memory of a page
of words); a warp's lanes each noted (a word in the middle taken by another block, a lane whose first or last byte is in another
page than the rest); remembered runs of words answering for their block only, joined only when they touch; readable words
looked at every time; two words taken at once only when both are free, at even and odd places, and kept like two; 600 random
interleavings of four blocks on two workers against a model of the four word states (by word: stopped at exactly
the access the model stops; by page: never later, earlier only as a coarse stop, and the model's again with the allocation by
word); a kernel that updates its buffer in place, tried again by word and kept, with the bits and counters of the run in order.

For lines (round sp58): a second block, on the same worker and on another, reading and writing at every byte of a line another
block holds as its record (some words read, some written, some untouched) and of the words on either side, with six lengths
from 1 to 17 bytes: stopped exactly where the words stop it, never coarsely, and both blocks going on by word where it is not;
what the counters say of a line one block touched and of one two did, and the words written out being zero for the next trial;
a line kept whole once at its block's first write in it, and words kept one by one in a line another block came to first;
1500 random interleavings of four blocks inside three lines across a page boundary against the model of the word states (noted by
word, by page, and with the allocation by word); and two races on real threads, 800 rounds each: a line written out to words
while its block goes on writing in it (no mark lost: a third block is stopped at every word, and every byte goes back), and a
writer going up a line while a reader comes down it (one of the two is always stopped). For the scout: what block 0 read and
wrote alone is the other blocks' to read and write, a stopped trial puts back what the launch started with though a later block
wrote over block 0's store, a later piece of the log is not joined to the scout's, only what was read by page and written is
learned, a stopped scout learns nothing, and the worker forgets what it remembered of the scout's marks; the kernel that updates
its buffer in place is scouted at its first launch (no trial thrown away) and not on a small grid or with `VGPU_EXACT_SCOUT=0`.
The width rule of trials is tested in `tests/unit/test_worker_pool.cpp` (the verdict by wins and sum; a kernel measured on one
thread before its small launches go wide; a figure too high dropped).
