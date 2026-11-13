#!/bin/bash
# builds.sh: the round's three build trees on server3, one after another, in holds of the machine lock.
H=$HOME/.cache/sp62; L=/tmp/pantheonsim-builder.lock; ABI=$HOME/.cache/pantheonsim-sp39-pgo/cuda-abi/include
T="vgpucuda vgpucudart test_exact_threads"
conf() {  # tree build-dir type flags
  flock $L nice -n 19 cmake -G Ninja -S $H/$1/src -B $H/$1/$2 -DCMAKE_BUILD_TYPE=$3 -DVGPU_LTO=OFF -DVGPU_FETCH_CUDNN_HEADERS=OFF \
    -DCUDA_ABI_INCLUDE=$ABI "${@:4}" > $H/$1/$2-cmake.log 2>&1 || { echo "configure $1/$2 failed"; tail -5 $H/$1/$2-cmake.log; exit 1; }
}
for step in "$@"; do
  case $step in
    new) conf new build Release -DCMAKE_CXX_FLAGS=-g1; python3 $H/hold_build.py $H/new/build $T || exit 1 ;;
    base) conf base build Release -DCMAKE_CXX_FLAGS=-g1; python3 $H/hold_build.py $H/base/build vgpucuda vgpucudart || exit 1 ;;
    tsan) conf new tsan Debug "-DCMAKE_CXX_FLAGS=-fsanitize=thread -g -O1" -DCMAKE_EXE_LINKER_FLAGS=-fsanitize=thread -DCMAKE_SHARED_LINKER_FLAGS=-fsanitize=thread
          python3 $H/hold_build.py $H/new/tsan test_exact_threads || exit 1 ;;
  esac
  echo "== $step done $(date +%T)"; df -h / | tail -1
done
