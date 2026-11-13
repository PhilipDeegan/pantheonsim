#!/usr/bin/env bash
# Names the simulator build the GitHub Action needs on this runner, as GITHUB_OUTPUT lines:
#   key=     the actions/cache key of a build from source
#   version= the simulator's version (CMakeLists.txt), whose release (tag v<version>) may hold a prebuilt one
#   asset=   that release asset's file name, or empty where no prebuilt one is published (not x86-64 Linux)
# action.yml computes these before it builds, and .github/workflows/release-binaries.yml publishes
# its build under the action's prebuilt-asset output, so the two always agree on the name. The source
# hash is in the name: a prebuilt simulator is used only for exactly the source it was built from, so
# @main after a release builds as before.
# Usage: action-build-id.sh <source-dir>   (with the job's nvcc, if any, on PATH)
set -euo pipefail
src_dir=$1
src=$(cd "$src_dir" && find CMakeLists.txt src include nvidia amd tools scripts -type f -print0 \
      | sort -z | xargs -0 sha256sum | sha256sum | cut -c1-20)
# No nvcc (an AMD job, or cuda-toolkit: none without one) is a toolkit version of none, not a failure.
nvcc_version=$( { nvcc --version 2>/dev/null || true; } | sed -n 's/.*release \([0-9.]*\).*/\1/p' | head -1)
version=$(sed -n 's/^project(pantheonsim VERSION \([0-9][0-9.]*\).*/\1/p' "$src_dir/CMakeLists.txt" | head -1)
# The OS release too: a build from 24.04 does not load on 22.04.
. /etc/os-release
os="${ID}${VERSION_ID}"
echo "key=pantheonsim-${RUNNER_OS:-Linux}-${RUNNER_ARCH:-X64}-${os}-${src}-cuda${nvcc_version:-none}"
echo "version=$version"
asset=
if [[ "${RUNNER_OS:-Linux}" == Linux && "${RUNNER_ARCH:-X64}" == X64 && -n "$version" ]]; then
  asset="pantheonsim-${version}-linux-x86_64-${os}-cuda${nvcc_version:-none}-${src}.tar.gz"
fi
echo "asset=$asset"
