#!/usr/bin/env bash
# Profile-guided release build, end to end and reproducibly: instrumented build, training over a manifest of programs,
# profile check, and the optimised build. This is what a release job calls; the default build stays plain and PGO stays
# opt-in (VGPU_PGO in CMakeLists.txt). See docs/pgo.md.
#
#   scripts/pgo-build.sh [options] [stage]
#
# stages (default: all)
#   all        generate, train, use
#   generate   configure -DVGPU_PGO=generate and build the release targets (instrumented; thrown away afterwards)
#   train      run scripts/pgo/train-manifest.txt against the instrumented build, write the profile fingerprint
#   use        check the fingerprint, configure -DVGPU_PGO=use (strict) and build the release targets
#   check      check the manifest and the training programs only; build nothing
#   verify     check that the profile in --build belongs to this compiler, source tree and build path; build nothing
#   merge OUT DIR...   merge profile directories (gcov-tool) trained on several machines into OUT
#
# options
#   --build DIR        build directory (default build-pgo, relative to the source tree). The profile files are named
#                      after the object paths in it: every stage must see the same absolute path.
#   --src DIR          source tree (default: the tree this script is in)
#   --corp DIR         training corpus (default $PGO_CORP): see docs/pgo.md for its layout
#   --py PYTHON        python with PyTorch, for the PyTorch entries (default $PGO_PY, else python3)
#   --manifest FILE    training manifest (default scripts/pgo/train-manifest.txt)
#   --jobs N           build parallelism (default 3)
#   --train-jobs N     training programs run side by side (default 1)
#   --cmake-arg ARG    extra argument for every cmake configure (repeatable), e.g. -DCUDA_ABI_INCLUDE=/path/include
#   --targets "A B"    cmake targets to build (default: every vgpu* shared library and vgpu_cli)
#   --lock FILE        take this flock around every build and every training run (shared build machines)
#   --min-ok F         fraction of manifest entries that must finish in training (default 0.9)
#   --min-profiled F   fraction of the release translation units that must have profile data (default 0.5)
#   --allow-stale      skip the fingerprint check (the compile error on a mismatching source stays on)
#
# Exit status: 0 ok; 1 a build or training step failed; 2 usage; 3 the profile does not belong to this compiler, source or
# build path (nothing is built); 4 the optimised build reported a profile mismatch.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$(cd "$HERE/.." && pwd)"
BUILD=build-pgo
CORP="${PGO_CORP:-}"; PY="${PGO_PY:-python3}"
MANIFEST=""
JOBS=3; TRAIN_JOBS=1; LOCK=""; MIN_OK=0.9; MIN_PROFILED=0.5; STALE=0
CMAKE_ARGS=(); TARGETS=""
stage=all; MERGE_ARGS=()

die() { echo "pgo-build: $*" >&2; exit "${2:-1}"; }
note() { echo "pgo-build: $*"; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --build) BUILD="$2"; shift 2 ;;
    --src) SRC="$(cd "$2" && pwd)"; shift 2 ;;
    --corp) CORP="$2"; shift 2 ;;
    --py) PY="$2"; shift 2 ;;
    --manifest) MANIFEST="$2"; shift 2 ;;
    --jobs) JOBS="$2"; shift 2 ;;
    --train-jobs) TRAIN_JOBS="$2"; shift 2 ;;
    --cmake-arg) CMAKE_ARGS+=("$2"); shift 2 ;;
    --targets) TARGETS="$2"; shift 2 ;;
    --lock) LOCK="$2"; shift 2 ;;
    --min-ok) MIN_OK="$2"; shift 2 ;;
    --min-profiled) MIN_PROFILED="$2"; shift 2 ;;
    --allow-stale) STALE=1; shift ;;
    -h|--help) sed -n '2,/^set -euo/p' "${BASH_SOURCE[0]}" | sed '$d' | sed 's/^# \{0,1\}//'; exit 0 ;;
    all|generate|train|use|check|verify) stage="$1"; shift ;;
    merge) stage=merge; shift; MERGE_ARGS=("$@"); break ;;
    *) die "unknown argument $1 (try --help)" 2 ;;
  esac
done

[[ "$BUILD" = /* ]] || BUILD="$SRC/$BUILD"
PGO_DIR="$BUILD/pgo"
INFO="$PGO_DIR/PROFILE-INFO"
[[ -n "$MANIFEST" ]] || MANIFEST="$SRC/scripts/pgo/train-manifest.txt"
locked() { if [[ -n "$LOCK" ]]; then flock "$LOCK" "$@"; else "$@"; fi; }

compiler() { # the C++ compiler cmake chose for $BUILD
  sed -n 's/^CMAKE_CXX_COMPILER:FILEPATH=//p' "$BUILD/CMakeCache.txt" 2>/dev/null | head -1
}
compiler_id() { local cxx; cxx="$(compiler)"; [[ -n "$cxx" ]] || cxx="${CXX:-c++}"; echo "$("$cxx" -dumpfullversion) $("$cxx" -dumpmachine)"; }

# A digest of what the optimised libraries are compiled from: C and C++ sources, headers, embedded device profiles and
# the CMake files, not tests, docs or scripts. A different digest at "use" than at "train" is a stale profile.
source_digest() {
  ( cd "$SRC" && find . -type f \( -name '*.c' -o -name '*.cc' -o -name '*.cpp' -o -name '*.cxx' -o -name '*.h' -o -name '*.hpp' \
        -o -name '*.inc' -o -name '*.def' -o -name '*.yaml' -o -name 'CMakeLists.txt' -o -name '*.cmake' \) \
      -not -path './.git/*' -not -path './build*' -not -path '*/tests/*' -not -path './docs/*' -not -path './scripts/*' \
      -not -path './ci/*' -not -path './tools/*' -not -path './tests/*' -print0 \
    | LC_ALL=C sort -z | xargs -0 sha256sum | sha256sum | cut -d' ' -f1 )
}

release_targets() {
  if [[ -n "$TARGETS" ]]; then echo "$TARGETS"; return; fi
  cmake --build "$BUILD" --target help | sed -n 's/^\.\.\. //p' | sed 's/ .*//' \
    | grep -E '^vgpu[a-z0-9_]*$' | grep -v -E '^(vgpu|vgpunpp_objs|vgpunvcomp_cpu)$' | tr '\n' ' '
}

configure() { # mode
  locked cmake -S "$SRC" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release -DVGPU_PGO="$1" -DVGPU_PGO_DIR="$PGO_DIR" \
    ${CMAKE_ARGS[@]+"${CMAKE_ARGS[@]}"} "${@:2}" >"$BUILD.configure-$1.log" 2>&1 \
    || { tail -20 "$BUILD.configure-$1.log" >&2; die "configure ($1) failed, log: $BUILD.configure-$1.log"; }
}

build() { # mode
  local t log="$BUILD.build-$1.log"
  t="$(release_targets)"
  note "building $(wc -w <<<"$t") targets ($1), log $log"
  # shellcheck disable=SC2086
  locked nice -n 19 cmake --build "$BUILD" -j"$JOBS" --target $t >"$log" 2>&1 || { tail -30 "$log" >&2; die "build ($1) failed, log: $log"; }
}

train_args() {
  echo --manifest "$MANIFEST" --build "$BUILD" --corp "$CORP" --py "$PY" --src "$SRC"
}

stage_check() {
  [[ -f "$MANIFEST" ]] || die "no manifest $MANIFEST" 2
  [[ -n "$CORP" ]] || die "no corpus: pass --corp or set PGO_CORP" 2
  # shellcheck disable=SC2046
  python3 -I "$HERE/pgo/train.py" $(train_args) --check
}

stage_generate() {
  mkdir -p "$BUILD"
  rm -rf "$PGO_DIR"
  configure generate
  build generate
}

stage_train() {
  [[ -d "$BUILD/shim" ]] || die "no instrumented build in $BUILD: run the generate stage" 2
  grep -q '^VGPU_PGO:STRING=generate' "$BUILD/CMakeCache.txt" || die "$BUILD is not configured with VGPU_PGO=generate" 2
  [[ -n "$CORP" ]] || die "no corpus: pass --corp or set PGO_CORP" 2
  rm -rf "$PGO_DIR"; mkdir -p "$PGO_DIR"
  local lockarg=(); [[ -n "$LOCK" ]] && lockarg=(--lock "$LOCK")
  local t0 rc=0; t0=$(date +%s)
  # shellcheck disable=SC2046
  python3 -I "$HERE/pgo/train.py" $(train_args) --jobs "$TRAIN_JOBS" --min-ok "$MIN_OK" ${lockarg[@]+"${lockarg[@]}"} \
      --log "$BUILD.train-logs" --json "$BUILD.train.json" || rc=$?
  [[ $rc -eq 0 ]] || die "training: too many programs did not finish (see above, logs in $BUILD.train-logs)" 1
  local n; n=$(find "$PGO_DIR" -name '*.gcda' | wc -l)
  [[ "$n" -gt 0 ]] || die "training wrote no profile data under $PGO_DIR: is the build instrumented and the shim loaded?" 1
  {
    echo "compiler=$(compiler_id)"
    echo "source_digest=$(source_digest)"
    echo "build_dir=$BUILD"
    echo "manifest_sha256=$(sha256sum "$MANIFEST" | cut -d' ' -f1)"
    echo "gcda_files=$n"
    echo "trained_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "training_seconds=$(( $(date +%s) - t0 ))"
    command -v git >/dev/null && git -C "$SRC" rev-parse HEAD 2>/dev/null | sed 's/^/commit=/' || true
  } >"$INFO"
  note "profile: $n files, $(du -sk "$PGO_DIR" | cut -f1) KB, trained in $(( $(date +%s) - t0 )) s ($INFO)"
}

info_get() { sed -n "s/^$1=//p" "$INFO" | head -1; }

check_fingerprint() {
  [[ -f "$INFO" ]] || die "no profile fingerprint $INFO: run the train stage" 3
  local ok=1 want have
  want="$(info_get compiler)"; have="$(compiler_id)"
  [[ "$want" == "$have" ]] || { echo "pgo-build: STALE PROFILE: compiler was '$want', is '$have' (a .gcda profile is valid for one compiler version)" >&2; ok=0; }
  want="$(info_get source_digest)"; have="$(source_digest)"
  [[ "$want" == "$have" ]] || { echo "pgo-build: STALE PROFILE: the sources changed since training (digest ${want:0:12} -> ${have:0:12})" >&2; ok=0; }
  want="$(info_get build_dir)"
  [[ "$want" == "$BUILD" ]] || { echo "pgo-build: STALE PROFILE: trained in '$want', building in '$BUILD' (profile files are named after the object path)" >&2; ok=0; }
  if [[ $ok -eq 0 && $STALE -eq 0 ]]; then die "the profile in $PGO_DIR does not belong to this compiler, source or build path: re-run the generate and train stages (or pass --allow-stale; the compile error on a mismatching function stays on)" 3; fi
}

# Translation units of the release targets that have no profile file: their code is compiled without a profile, silently
# (-Wno-missing-profile). Prints "<profiled> <total>".
profile_coverage() {
  local obj stem name total=0 have=0
  while IFS= read -r obj; do
    total=$((total + 1))
    stem="${obj%.o}"; name="${stem//\//#}"
    [[ -f "$PGO_DIR/$name.gcda" ]] && have=$((have + 1))
  done < <(find "$BUILD/CMakeFiles" -name '*.o' \( -path '*/vgpu.dir/*' -o -path '*/vgpu[a-z0-9_]*.dir/*' \) -not -path '*/vgpu_test*' -not -path '*/vgpumetis.dir/*' -not -path '*/vgpunpp*' -not -name 'test_*' 2>/dev/null)
  echo "$have $total"
}

stage_use() {
  [[ -d "$PGO_DIR" ]] || die "no profile directory $PGO_DIR: run the train stage" 3
  check_fingerprint
  configure use -DVGPU_PGO_STRICT=ON
  local log="$BUILD.build-use.log" rc=0 t
  t="$(release_targets)"
  note "building $(wc -w <<<"$t") targets (use, strict), log $log"
  # shellcheck disable=SC2086
  locked nice -n 19 cmake --build "$BUILD" -j"$JOBS" --target $t >"$log" 2>&1 || rc=$?
  if grep -E -i 'coverage-mismatch|profile data may be corrupted|control flow checksum|number of counters' "$log" >/dev/null; then
    grep -E -i -m5 'coverage-mismatch|profile data may be corrupted|control flow checksum|number of counters' "$log" >&2
    die "the optimised build reports a source that does not match its profile (log $log): the profile is stale" 4
  fi
  [[ $rc -eq 0 ]] || { tail -30 "$log" >&2; die "build (use) failed, log: $log" 1; }
  read -r have total < <(profile_coverage)
  note "translation units with profile data: $have of $total"
  awk -v h="$have" -v t="$total" -v m="$MIN_PROFILED" 'BEGIN{exit !(t>0 && h/t >= m)}' \
    || die "only $have of $total translation units have profile data (< $MIN_PROFILED): the training run did not load most of the libraries" 1
  note "done: $BUILD (libraries in $BUILD/shim)"
}

stage_merge() {
  local out="${MERGE_ARGS[0]:-}"; local dirs=("${MERGE_ARGS[@]:1}")
  [[ -n "$out" && ${#dirs[@]} -ge 1 ]] || die "usage: pgo-build.sh merge OUT DIR [DIR...]" 2
  local gt; gt="gcov-tool-$(c++ -dumpversion | cut -d. -f1)"; command -v "$gt" >/dev/null || gt=gcov-tool
  command -v "$gt" >/dev/null || die "gcov-tool not found (it ships with gcc)" 2
  rm -rf "$out"; mkdir -p "$out"
  local first="${dirs[0]}"
  cp -a "$first/." "$out/"
  local d tmp
  for d in "${dirs[@]:1}"; do
    tmp="$(mktemp -d)"; "$gt" merge -o "$tmp" "$out" "$d" || die "gcov-tool merge failed for $d"
    rm -rf "$out"; mv "$tmp" "$out"
  done
  note "merged ${#dirs[@]} profile directories into $out ($(find "$out" -name '*.gcda' | wc -l) files); copy PROFILE-INFO by hand if the shards share one"
}

case "$stage" in
  check) stage_check ;;
  generate) stage_generate ;;
  train) stage_train ;;
  use) stage_use ;;
  verify) check_fingerprint; note "the profile in $PGO_DIR matches this compiler, source tree and build path" ;;
  merge) stage_merge ;;
  all) stage_check; stage_generate; stage_train; stage_use ;;
esac
