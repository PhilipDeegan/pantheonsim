Data of round sp53 (the section "Round sp53" of `../../rounds/11-rounds-sp36-sp41.md`). Scripts: `scripts/` (its README says what each does).

SYR2K with linking:
* `aslr-syr2k-off.json`: unlinked and linked with address-space randomisation off and the stack moved by 0 to 3,072 bytes (round sp52's build).
* `bisect-syr2k.txt`: what the objects hold, linking feature by feature, for the two forms of the row copy.
* `ab-syr2k-bisect.json`, `ab-syr2k-inline.json`: the timings of the two fixes (seven sides: the row copy and the linking features; five sides: the helpers inlined or not), one line a program (`scripts/compact.py` of `sp33.py`'s file: medians, spreads, every run's wall).
* `stepcount-syr2k.txt`: the instructions executed over whole warps, unlinked and linked, by function, and the generated code's by kind.
* `ab-rule.json`: SYR2K and SYRK on the final build: JIT, JIT again, linked (left unlinked by the loop rule), linked with `VGPU_JIT_LINK_RULE=0`.

BSYNC:
* `exits-linked-nobsync.json`, `exits-linked.json`: the exit counts of eleven programs without and with BSYNC a region's op (the build before the loop rule); `exits-linked-18.json`: all 18 programs on the final build with `VGPU_JIT_LINK_BSYNC=1` (with the rule's count of sections left unlinked).
* `ab-18.json` (eight programs, five sides: JIT, JIT again, linked with BSYNC a region's op, linked without, linked with BSYNC and three more compiler flags; the build before the loop rule), `ab-18-b.json` (ten programs, the first four sides; nine on the build with the loop rule, LU on the one where BSYNC is an opt-in), `ab-tables.md` (the tables).
* `same-bits-full-form.log`: linking in the full form (residence and specialisation), with BSYNC too, and with the bug of the stop op put back, on five programs.
* `same-bits.log`: output, counters file, kernel digest and exit status of the JIT and the linked JIT, cold and warm, at 1 and 4 threads, against the tier.

The compile:
* `compile-cost.txt`: the compile of three programs' objects, variant by variant; `flag-sweep-black-scholes.txt`: the same source with more flags.

Tests:
* `tests.log`: the JIT's unit tests and the run-loop tests in all their legs on the final build (one line a leg).
* `tests-final.log`: the JIT's unit tests and the linking legs of `test_sass_chain_cont` on the build where BSYNC is an opt-in; `matrix.log`: the test of all sixteen forms on the final build (with the fix of the stop op) and under its mutant `res_stop_keep`.
* `mutants.log` (the BSYNC mutants and `row_loop`, on the build before the loop rule; the columns of a line are cut to a fixed width by the script), `mutants-escape.log` (`escape_outline`), `mutants-rule.log` (the rule's mutants, on the final build).
