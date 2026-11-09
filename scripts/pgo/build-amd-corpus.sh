#!/usr/bin/env bash
# Build the AMD training/test programs of scripts/pgo/train-manifest.txt from the CUDA sources of the Phase 0 corpus:
# hipify-perl, then hipcc for gfx942 (the simulator runs them as amd/mi300x). Needs ROCm (hipify-perl, hipcc) and the
# corpus layout of tools/bench/real/pack_corpora.sh plus the Rodinia and PolyBench source trees:
#
#   scripts/pgo/build-amd-corpus.sh CORP_DIR            # writes CORP_DIR/amd/bin/<name>
#
# A program that does not hipify or compile is left out and listed at the end; the manifest entries that name it fail
# soft (scripts/pgo/train.py reports a missing binary, it does not abort). This is a once-per-corpus step, not a
# training-time step: training only runs the binaries.
set -u
CORP="${1:?usage: build-amd-corpus.sh CORP_DIR}"
CORP="$(cd "$CORP" && pwd)"
ROCM="${ROCM_PATH:-/opt/rocm}"
HIPIFY="$ROCM/bin/hipify-perl"; HIPCC="$ROCM/bin/hipcc"
RODINIA="${RODINIA_SRC:-$HOME/.cache/gpu-rodinia/cuda}"
POLY="${POLYBENCH_SRC:-$HOME/.cache/polybenchGpu}"
OUT="$CORP/amd"; mkdir -p "$OUT/bin" "$OUT/log" "$OUT/work"
rm -f "$OUT/failed.txt"

hipall() { local f; for f in $(find . -type f \( -name '*.cu' -o -name '*.cuh' -o -name '*.h' -o -name '*.c' -o -name '*.cpp' \)); do
  "$HIPIFY" -inplace -quiet-warnings "$f" >/dev/null 2>&1; done; }
hipbuild() { # name, flags and sources...
  local n=$1; shift
  if nice -n 19 "$HIPCC" -O2 -ffp-contract=off --offload-arch=gfx942 -x hip "$@" -o "$OUT/bin/$n" >"$OUT/log/$n.log" 2>&1; then
    echo "OK $n"; else echo "FAIL $n"; echo "$n" >>"$OUT/failed.txt"; fi; }

# HeCBench: <dir>-cuda, every source but the *_gold / *_cpu references
hec() { local d=$1 extra=("${@:2}") w="$OUT/work/hec-$1"
  rm -rf "$w"; mkdir -p "$w"; cp -r "$CORP/hec/src/$d-cuda/." "$w/"; ( cd "$w" || exit 1; hipall
    srcs=$(ls *.cu *.c *.cpp 2>/dev/null | grep -v -E '^(.*_gold|.*_cpu)\.' | tr '\n' ' ')
    hipbuild "hec-$d" -I. "${extra[@]}" $srcs ); rm -rf "$w"; }
for d in adam aop asmooth asyncAllocation atomicPerf axpby bitonic-sort silu sortKV bilateral atan2 \
         aobench radixsort2 shmembench; do hec "$d"; done

# Rodinia
W="$OUT/work/rod"; rm -rf "$W"; mkdir -p "$W"; cp -r "$RODINIA/." "$W/"
( cd "$W" || exit 1; hipall
  ( cd gaussian && hipbuild rod-gaussian gaussian.cu )
  ( cd nw && hipbuild rod-needle needle.cu )
  ( cd pathfinder && hipbuild rod-pathfinder pathfinder.cu )
  ( cd hotspot && hipbuild rod-hotspot hotspot.cu )
  ( cd srad/srad_v2 && hipbuild rod-srad_v2 srad.cu )
  ( cd bfs && hipbuild rod-bfs bfs.cu -DTIMING -I../util ) ); rm -rf "$W"

# PolyBench/GPU
W="$OUT/work/pb"; rm -rf "$W"; mkdir -p "$W"; cp -r "$POLY/CUDA" "$POLY/common" "$W/"
( cd "$W" || exit 1; hipall; cd CUDA || exit 1
  for d in 2DCONV ATAX BICG GEMM GEMVER GESUMMV JACOBI2D MVT 3MM ADI 3DCONV JACOBI1D SYRK; do
    ( cd "$d" && src=$(ls ./*.cu | head -1) && hipbuild "pb-$d" "$src" -I. -I../../common ); done ); rm -rf "$W"
rm -rf "$OUT/work"
echo "built: $(ls "$OUT/bin" | wc -l) programs; failed: $(tr '\n' ' ' <"$OUT/failed.txt" 2>/dev/null || true)"
