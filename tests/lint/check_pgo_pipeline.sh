#!/usr/bin/env bash
# The pieces of scripts/pgo-build.sh work, on a tiny manifest and a tiny C program, in seconds, without a simulator build:
#   1. the checked-in manifests parse, and no program is both trained on and held out;
#   2. the training runner repeats entries, caps them, and tells failed, capped and missing programs apart;
#   3. the profile fingerprint (compiler, source digest, build path) rejects a stale profile with exit status 3;
#   4. gcc's own profile flow as the CMake option uses it: generate, merge two profile directories with gcov-tool, use,
#      and a source that changed after training is a COMPILE ERROR under -Werror=coverage-mismatch (VGPU_PGO_STRICT),
#      where the old -Wno-coverage-mismatch built it without a word.
# Needs bash, python3, gcc, gcov-tool. Exit 77 (ctest: skipped) when gcc or gcov-tool is missing.
set -uo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
pgo="$root/scripts/pgo-build.sh"; train="$root/scripts/pgo/train.py"
fail=0
ok() { echo "ok    $*"; }
bad() { echo "FAIL  $*"; fail=1; }
t="$(mktemp -d)"; trap 'rm -rf "$t"' EXIT

# 1. manifests -----------------------------------------------------------------------------------------------------
python3 -I - "$root" <<'PY' && ok "train and test manifests parse; no program is in both; every test program is held out" || bad "manifests"
import sys
sys.path.insert(0, sys.argv[1] + "/scripts/pgo")
import manifest as m
tr = m.parse(sys.argv[1] + "/scripts/pgo/train-manifest.txt")
te = m.parse(sys.argv[1] + "/scripts/pgo/test-manifest.txt")
def prog(e):  # the program a line runs, without the engine / thread suffixes of the id
    return (e.command[0].split("/")[-1], tuple(e.command[1:2]) if e.command[0].endswith("sweep") else ())
def base(e):
    i = e.id
    for suf in ("-ptx", "-t4"):
        if i.endswith(suf):
            i = i[: -len(suf)]
    return i.replace("tiny/", "nv/")
trb = {base(e) for e in tr}
clash = [e.id for e in te if base(e) in trb]
assert not clash, "trained and held out: %s" % clash
assert len(te) >= 20 and len(tr) >= 50, (len(tr), len(te))
groups = {e.id.split("/")[0] for e in tr}
assert {"nv", "amd", "pan", "torch"} <= groups, groups
nv_test = [e for e in te if e.group == "nv"]
amd_test = [e for e in te if e.group == "amd"]
assert len(nv_test) >= 10 and len(amd_test) >= 4, (len(nv_test), len(amd_test))
PY

# 2. the training runner ---------------------------------------------------------------------------------------------
mkdir -p "$t/b/shim" "$t/c"
cat >"$t/c/ok.sh" <<E
#!/bin/sh
echo "\$1" >>"$t/count"
E
cat >"$t/c/fail.sh" <<'E'
#!/bin/sh
exit 3
E
cat >"$t/c/slow.sh" <<'E'
#!/bin/sh
sleep 30
E
chmod +x "$t"/c/*.sh
cat >"$t/m.txt" <<E
# id | kind | threads | reps | cap_s | cwd | env | command
tiny/a/ok     | raw | 1 | 3 | 20 | tmp | - | @CORP@/ok.sh a
tiny/a/ok2    | raw | 2 | 1 | 20 | tmp | X=1 | @CORP@/ok.sh b
tiny/a/fail   | raw | 1 | 1 | 20 | tmp | - | @CORP@/fail.sh
tiny/a/slow   | raw | 1 | 1 | 1  | tmp | - | @CORP@/slow.sh
tiny/a/absent | raw | 1 | 1 | 20 | tmp | - | @CORP@/nothere.sh
E
out=$(python3 -I "$train" --manifest "$t/m.txt" --build "$t/b" --corp "$t/c" --min-ok 0.3 --no-nice 2>&1); rc=$?
[[ $rc -eq 0 ]] && ok "runner: 2 of 5 entries finished, --min-ok 0.3 accepts" || { bad "runner exit $rc with --min-ok 0.3"; echo "$out"; }
[[ $(wc -l <"$t/count") -eq 4 && $(grep -c '^a$' "$t/count") -eq 3 ]] && ok "runner: reps honoured (3 runs of a, 1 of b)" || bad "runner: reps ($(cat "$t/count" 2>/dev/null | tr '\n' ' '))"
grep -q 'tiny/a/fail .* failed' <<<"$out" && grep -q 'tiny/a/slow .* capped' <<<"$out" && grep -q 'tiny/a/absent .* missing' <<<"$out" \
  && ok "runner: failed, capped and missing are told apart" || { bad "runner statuses"; echo "$out"; }
python3 -I "$train" --manifest "$t/m.txt" --build "$t/b" --corp "$t/c" --min-ok 0.9 --no-nice >/dev/null 2>&1; rc=$?
[[ $rc -eq 1 ]] && ok "runner: --min-ok 0.9 fails the training (exit 1)" || bad "runner exit $rc with --min-ok 0.9"
printf 'tiny/a/x | bogus | 1 | 1 | 5 | tmp | - | true\n' >"$t/bad.txt"
python3 -I "$train" --manifest "$t/bad.txt" --build "$t/b" --corp "$t/c" >/dev/null 2>&1; rc=$?
[[ $rc -eq 2 ]] && ok "runner: a bad manifest line is exit 2" || bad "runner: bad manifest gave exit $rc"
python3 -I "$train" --manifest "$t/m.txt" --build "$t/nobuild" --corp "$t/c" >/dev/null 2>&1; rc=$?
[[ $rc -eq 2 ]] && ok "runner: a missing instrumented build is exit 2" || bad "runner: missing build gave exit $rc"

# 3. the profile fingerprint ------------------------------------------------------------------------------------------
mkdir -p "$t/src/nvidia/src" "$t/src/b/pgo"
echo 'int f(){return 1;}' >"$t/src/nvidia/src/a.cpp"
echo 'project(x)' >"$t/src/CMakeLists.txt"
cxx="$(command -v c++ || command -v g++ || true)"
if [[ -n $cxx ]]; then
  echo "CMAKE_CXX_COMPILER:FILEPATH=$cxx" >"$t/src/b/CMakeCache.txt"
  # a fingerprint as the train stage writes it
  digest=$(cd "$t/src" && find . -type f \( -name '*.cpp' -o -name 'CMakeLists.txt' \) -not -path './b/*' -print0 | LC_ALL=C sort -z | xargs -0 sha256sum | sha256sum | cut -d' ' -f1)
  printf 'compiler=%s %s\nsource_digest=%s\nbuild_dir=%s\n' "$("$cxx" -dumpfullversion)" "$("$cxx" -dumpmachine)" "$digest" "$t/src/b" >"$t/src/b/pgo/PROFILE-INFO"
  "$pgo" --src "$t/src" --build b verify >/dev/null 2>&1; rc=$?
  [[ $rc -eq 0 ]] && ok "fingerprint: a matching profile is accepted" || bad "fingerprint: matching profile gave exit $rc"
  echo 'int f(){return 2;}' >"$t/src/nvidia/src/a.cpp"
  out=$("$pgo" --src "$t/src" --build b verify 2>&1); rc=$?
  [[ $rc -eq 3 ]] && grep -q 'sources changed' <<<"$out" && ok "fingerprint: a changed source is exit 3 (STALE PROFILE)" || { bad "fingerprint: changed source gave exit $rc"; echo "$out"; }
  echo 'int f(){return 1;}' >"$t/src/nvidia/src/a.cpp"
  sed -i 's/^compiler=.*/compiler=9.9.9 x86_64-none/' "$t/src/b/pgo/PROFILE-INFO"
  out=$("$pgo" --src "$t/src" --build b verify 2>&1); rc=$?
  [[ $rc -eq 3 ]] && grep -q 'compiler was' <<<"$out" && ok "fingerprint: another compiler version is exit 3" || { bad "fingerprint: other compiler gave exit $rc"; echo "$out"; }
  "$pgo" --src "$t/src" --build b --allow-stale verify >/dev/null 2>&1; rc=$?
  [[ $rc -eq 0 ]] && ok "fingerprint: --allow-stale lets it through" || bad "fingerprint: --allow-stale gave exit $rc"
else
  bad "fingerprint: no C++ compiler"
fi

# 4. gcc's profile flow, with the flags VGPU_PGO uses ----------------------------------------------------------------------
if ! command -v gcc >/dev/null || ! { command -v gcov-tool >/dev/null || command -v "gcov-tool-$(gcc -dumpversion | cut -d. -f1)" >/dev/null; }; then
  echo "skip  gcc or gcov-tool not installed: the compiler-flow checks are skipped"
  [[ $fail -eq 0 ]] && exit 77 || exit 1
fi
w="$t/gcc"; mkdir -p "$w"
cat >"$w/p.c" <<'E'
#include <stdio.h>
#include <stdlib.h>
static int hot(int x) { int s = 0; for (int i = 0; i < 1000; i++) s += (x + i) % 7 == 0 ? 3 : 1; return s; }
static int cold(int x) { return x * 2; }
int main(int argc, char **argv) { int n = argc > 1 ? atoi(argv[1]) : 0; int s = 0;
  for (int i = 0; i < 2000; i++) s += n ? hot(i) : cold(i); printf("%d\n", s); return 0; }
E
G=(-fprofile-generate="$w/prof1" -fprofile-update=prefer-atomic)
( cd "$w" && gcc -O2 "${G[@]}" p.c -o p && ./p 0 >/dev/null && cp -r prof1 prof-a && rm -rf prof1 && ./p 1 >/dev/null && cp -r prof1 prof-b ) \
  && ok "gcc: instrumented build trained in two shards" || bad "gcc: generate/train"
"$pgo" merge "$w/merged" "$w/prof-a" "$w/prof-b" >/dev/null 2>&1 && [[ -n $(find "$w/merged" -name '*.gcda') ]] \
  && ok "merge: gcov-tool merged two profile directories" || bad "merge stage"
# the profile files are named after the object path: use must see the same absolute path as generate did, so the use build
# goes through the same -o path (as pgo-build.sh keeps one build directory)
U=(-fprofile-use="$w/merged" -fprofile-correction)
# every compile writes to the same output name: the profile file is named after it (a different -o finds no profile)
( cd "$w" && gcc -O2 "${U[@]}" -Werror=coverage-mismatch -Werror=missing-profile p.c -o p ) >"$w/use.log" 2>&1 \
  && ok "use: the matching source builds, and finds its profile, with -Werror=coverage-mismatch" || { bad "use: matching source failed"; cat "$w/use.log"; }
( cd "$w" && gcc -O2 "${U[@]}" -Werror=missing-profile p.c -o elsewhere ) >"$w/moved.log" 2>&1; rc=$?
[[ $rc -ne 0 ]] && grep -q 'missing-profile' "$w/moved.log" && ok "use: another object path finds no profile (why the build directory must not move; -Wno-missing-profile would hide it)" \
  || { bad "use: expected a missing profile for another output name (rc $rc)"; cat "$w/moved.log"; }
# change hot() so that its control-flow checksum differs
cat >"$w/p.c" <<'E'
#include <stdio.h>
#include <stdlib.h>
static int hot(int x) { int s = 0; for (int i = 0; i < 1000; i++) { if ((x + i) % 7 == 0) s += 3; else if (i & 4) s += 5; else s += 1; } return s; }
static int cold(int x) { return x * 2; }
int main(int argc, char **argv) { int n = argc > 1 ? atoi(argv[1]) : 0; int s = 0;
  for (int i = 0; i < 2000; i++) s += n ? hot(i) : cold(i); printf("%d\n", s); return 0; }
E
( cd "$w" && gcc -O2 "${U[@]}" -Werror=coverage-mismatch p.c -o p ) >"$w/stale.log" 2>&1; rc=$?
[[ $rc -ne 0 ]] && grep -q 'coverage-mismatch' "$w/stale.log" && ok "use: a source changed after training is a compile error (strict)" \
  || { bad "use: stale source did not fail (rc $rc)"; cat "$w/stale.log"; }
( cd "$w" && gcc -O2 "${U[@]}" -Wno-missing-profile -Wno-coverage-mismatch p.c -o p ) >"$w/quiet.log" 2>&1; rc=$?
[[ $rc -eq 0 && ! -s "$w/quiet.log" ]] && ok "use: -Wno-coverage-mismatch (the old setting) builds the same stale source without a word" \
  || { bad "use: expected the old flag to be silent (rc $rc)"; cat "$w/quiet.log"; }
# os._exit skips the profile write; libc's exit() (what scripts/pgo/run_sweep.py puts in its place) does not
cat >"$w/lib.c" <<'E'
int f(int x) { int s = 0; for (int i = 0; i < x; i++) s += i & 3; return s; }
E
gcc -O2 -shared -fPIC -fprofile-generate="$w/profx" lib.c -o "$w/libx.so" 2>/dev/null || ( cd "$w" && gcc -O2 -shared -fPIC -fprofile-generate="$w/profx" lib.c -o libx.so )
python3 -I -c "import ctypes,sys; ctypes.CDLL(sys.argv[1]).f(100); import os; os._exit(0)" "$w/libx.so"
n1=$(find "$w/profx" -name '*.gcda' 2>/dev/null | wc -l)
python3 -I -c "import ctypes,sys,os; ctypes.CDLL(sys.argv[1]).f(100); ctypes.CDLL(None).exit(0)" "$w/libx.so"
n2=$(find "$w/profx" -name '*.gcda' 2>/dev/null | wc -l)
[[ $n1 -eq 0 && $n2 -ge 1 ]] && ok "profile: os._exit writes no profile ($n1 files), libc exit() does ($n2): scripts/pgo/run_sweep.py" || bad "profile: os._exit wrote $n1 files, exit() $n2"
python3 -I -m py_compile "$root/scripts/pgo/run_sweep.py" "$root/scripts/pgo/train.py" "$root/scripts/pgo/ab.py" "$root/scripts/pgo/manifest.py" \
  && ok "the scripts/pgo python files compile" || bad "py_compile"
# the CMake option exists and is on by default
grep -q 'option(VGPU_PGO_STRICT .* ON)' "$root/CMakeLists.txt" && grep -q 'Werror=coverage-mismatch' "$root/CMakeLists.txt" \
  && ok "CMakeLists.txt: VGPU_PGO_STRICT is on by default and adds -Werror=coverage-mismatch" || bad "CMakeLists.txt: VGPU_PGO_STRICT"
exit $fail
