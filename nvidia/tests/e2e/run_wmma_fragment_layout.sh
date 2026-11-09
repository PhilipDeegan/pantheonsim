#!/usr/bin/env bash
# Which matrix element every WMMA fragment element is (wmma_fragment_layout.cu), for every shape and type, and the accumulator-masked-by-an-
# A-fragment pattern tiny-cuda-nn's fused backward pass uses. The expected layout is an RTX 3060's. Run on both engines: SASS (the program is
# built for sm_86, so its kernels run as SASS) and PTX (VGPU_SASS=0). The PTX engine used a layout of its own, so this failed there (an
# accumulator element met an A-fragment element of another position) and, because a module with a SASS instruction the engine does not decode (HMNMX2) falls back to the PTX
# engine wholesale, in the SASS engine's default mode too.
set -euo pipefail
root="$(cd "$(dirname "$0")/../../.." && pwd)"
. "$root/tests/shim_guard.sh"
build="${VGPU_BUILD_DIR:-$root/build}"
shim="$build/shim"
src="$root/nvidia/tests/e2e/wmma_fragment_layout.cu"
out="${TMPDIR:-/tmp}/vgpu_e2e_wmma_fragment_layout_$$"
trap 'rm -f "$out" "$out.log"' EXIT
if ! command -v nvcc >/dev/null 2>&1; then
  echo "SKIP: nvcc not found (e2e needs the CUDA toolkit to compile the app)"; exit 0
fi
shopt -s nullglob
cudart_libs=("$shim"/libcudart.so.[0-9]*)
shopt -u nullglob
if (( ${#cudart_libs[@]} == 0 )); then
  echo "SKIP: libvgpucudart not built (CUDA ABI headers absent at build time)"; exit 0
fi
if ! nvcc -std=c++17 -O1 -cudart shared -arch=sm_86 -Wno-deprecated-gpu-targets \
       $(shim_sanitizer_nvcc_flags "$shim") "$src" -o "$out" 2>"$out.log"; then
  if grep -q "sm_86" "$out.log"; then
    echo "SKIP: this nvcc cannot target sm_86"; exit 0
  fi
  cat "$out.log"; exit 1
fi
if ! require_shim_libs "$shim" "$out"; then exit 0; fi
fails=0
for sass in 1 0; do
  result="$(VGPU_QUIET=1 VGPU_GPU=nvidia/rtx3060 VGPU_SASS=$sass LD_LIBRARY_PATH="$shim" "$out" 2>&1 || true)"
  if [[ "$(tail -n 1 <<<"$result")" == "PASS" ]] && ! grep -q '^FAIL' <<<"$result"; then
    echo "ok   VGPU_SASS=$sass"
  else
    echo "FAIL VGPU_SASS=$sass"; head -40 <<<"$result"; fails=1
  fi
done
(( fails == 0 )) && echo PASS || { echo FAIL; exit 1; }
