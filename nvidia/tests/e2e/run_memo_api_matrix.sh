#!/usr/bin/env bash
# Launch-level result caching (VGPU_MEMO) through the CUDA runtime API (memo_api_matrix.cu): the program's text must be
# the same with the cache off, cold and warm, and with every hit also run for real (VGPU_MEMO_VERIFY=2); the warm run must
# hit; and the cache must say what it did not cache.
set -uo pipefail
root="$(cd "$(dirname "$0")/../../.." && pwd)"
. "$root/tests/shim_guard.sh"
shim="${VGPU_BUILD_DIR:-$root/build}/shim"
out="${TMPDIR:-/tmp}/vgpu-memo-matrix.$$"
shopt -s nullglob
carts=("$shim"/libcudart.so.[0-9]*)
shopt -u nullglob
(( ${#carts[@]} )) || { echo "SKIP: libvgpucudart not built"; exit 0; }

nvcc_bin="$(pick_nvcc_for_shim "$shim")"
[[ -n "$nvcc_bin" ]] || { echo "SKIP: no nvcc matching this shim's toolkit"; exit 0; }
nvcc_host_compiler_fix
mkdir -p "$out.d"
trap 'rm -rf "$out" "$out.d"' EXIT

read -r -a san_flags <<< "$(shim_sanitizer_nvcc_flags "$shim")"
"$nvcc_bin" -std=c++17 -arch=sm_86 -cudart shared -Wno-deprecated-gpu-targets "${san_flags[@]}" \
     "$root/nvidia/tests/e2e/memo_api_matrix.cu" -o "$out" || { echo "FAIL: does not compile"; exit 1; }
if ! require_shim_libs "$shim" "$out"; then exit 0; fi

run() {   # name, extra env...
  local name="$1"; shift
  env VGPU_QUIET=1 VGPU_GPU=nvidia/rtx3060 VGPU_THREADS=1 LD_LIBRARY_PATH="$shim" \
      VGPU_KERNEL_DIGEST="$out.d/$name.digest" VGPU_COUNTERS_FILE="$out.d/$name.counters" \
      VGPU_MEMO_STATS_FILE="$out.d/$name.stats" "$@" "$out" > "$out.d/$name.txt" 2>&1
  local rc=$?
  sed -E 's/"launch":[0-9]+,"pid":[0-9]+,//' "$out.d/$name.counters" > "$out.d/$name.counters.n" 2>/dev/null || : > "$out.d/$name.counters.n"
  return $rc
}
cache="$out.d/cache"
run off || { cat "$out.d/off.txt"; echo "FAIL: the program fails without the cache"; exit 1; }
[[ "$(tail -n 1 "$out.d/off.txt")" == PASS ]] || { cat "$out.d/off.txt"; echo "FAIL: no PASS"; exit 1; }
run cold VGPU_MEMO=1 VGPU_MEMO_DIR="$cache" VGPU_MEMO_BUDGET=inf || { cat "$out.d/cold.txt"; echo "FAIL: cold run"; exit 1; }
run warm VGPU_MEMO=1 VGPU_MEMO_DIR="$cache" VGPU_MEMO_BUDGET=inf || { cat "$out.d/warm.txt"; echo "FAIL: warm run"; exit 1; }
run verify VGPU_MEMO=1 VGPU_MEMO_DIR="$cache" VGPU_MEMO_BUDGET=inf VGPU_MEMO_VERIFY=2 || { cat "$out.d/verify.txt"; echo "FAIL: verify run"; exit 1; }

ok=1
for r in cold warm verify; do
  for f in txt digest counters.n; do
    if ! cmp -s "$out.d/off.$f" "$out.d/$r.$f"; then
      echo "FAIL: $r differs from off in $f"; diff "$out.d/off.$f" "$out.d/$r.$f" | head -10; ok=0
    fi
  done
done
stat() { sed -E "s/.*[ ]$2=([0-9]+).*/\1/p;d" "$out.d/$1.stats" | head -1; }
hits_warm="$(stat warm hits)"; mism="$(stat verify verify_mismatch)"; checked="$(stat verify verify_checked)"
echo "    warm: hits=$hits_warm of $(stat warm launches) launches; verify: checked=$checked mismatches=$mism"
sed -E 's/.*reasons=//' "$out.d/cold.stats" | sed 's/^/    not cached (cold): /'
(( ${hits_warm:-0} > 0 )) || { echo "FAIL: the warm run did not hit"; ok=0; }
(( ${checked:-0} > 0 )) || { echo "FAIL: nothing was verified"; ok=0; }
[[ "${mism:-1}" == 0 ]] || { echo "FAIL: verify found $mism differences"; ok=0; }
for why in device-runtime-call cooperative textures; do
  grep -q "$why" "$out.d/cold.stats" || { echo "FAIL: '$why' was not counted as uncacheable"; ok=0; }
done
(( ok )) || exit 1
echo PASS
