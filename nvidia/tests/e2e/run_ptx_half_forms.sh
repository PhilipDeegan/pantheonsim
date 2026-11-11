#!/usr/bin/env bash
# Half-precision PTX (f16, f16x2, bf16, bf16x2 with every modifier, comparisons,
# conversions to and from every integer width) over 256 operand triples,
# against the card (ptx_half_forms.cu).
set -uo pipefail
root="$(cd "$(dirname "$0")/../../.." && pwd)"
. "$root/tests/shim_guard.sh"
shim="${VGPU_BUILD_DIR:-$root/build}/shim"
out="${TMPDIR:-/tmp}/vgpu-ptx-half.$$"
shopt -s nullglob
carts=("$shim"/libcudart.so.[0-9]*)
shopt -u nullglob
(( ${#carts[@]} )) || { echo "SKIP: libvgpucudart not built"; exit 0; }

nvcc_bin="$(pick_nvcc_for_shim "$shim")"
[[ -n "$nvcc_bin" ]] || { echo "SKIP: no nvcc matching this shim's toolkit"; exit 0; }
nvcc_host_compiler_fix
trap 'rm -f "$out" "$out.log"' EXIT

read -r -a san_flags <<< "$(shim_sanitizer_nvcc_flags "$shim")"
"$nvcc_bin" -std=c++17 -arch=sm_86 -cudart shared -Wno-deprecated-gpu-targets "${san_flags[@]}" \
     "$root/nvidia/tests/e2e/ptx_half_forms.cu" -o "$out" || { echo "FAIL: does not compile"; exit 1; }
if ! require_shim_libs "$shim" "$out"; then exit 0; fi

log="$out.log"
# The program's cubin carries SASS for sm_86, which runs by default. A module whose SASS holds one instruction the decoder refuses
# runs on the PTX engine instead, which hid every f16 form the SASS engine got wrong (.ftz, .sat, .relu, the directed cvt) until
# HMNMX2 was decoded. So the card's answers are checked on the SASS interpreter (VGPU_TIER=0), on the pre-decoded tier, and on the
# PTX engine (VGPU_SASS=0), and the SASS runs must say they ran SASS.
fail=0
for mode in "VGPU_TIER=0 VGPU_SASS_LOG=1" "VGPU_TIER=1 VGPU_SASS_LOG=1" "VGPU_SASS=0"; do
  # shellcheck disable=SC2086
  env VGPU_QUIET=1 VGPU_GPU=nvidia/rtx3060 LD_LIBRARY_PATH="$shim" $mode "$out" > "$log" 2>&1
  rc=$?
  echo "  [$mode]"
  sed 's/^/    /' "$log"
  if [[ $rc != 0 || "$(tail -n 1 "$log")" != PASS ]]; then echo "FAIL: half-precision PTX forms ($mode)"; fail=1; fi
  if [[ $mode == *VGPU_SASS_LOG* ]] && { ! grep -q "running SASS" "$log" || grep -q "running PTX instead of SASS" "$log"; }; then
    echo "FAIL: the SASS did not run ($mode)"; fail=1
  fi
done
exit $fail
