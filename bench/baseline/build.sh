#!/usr/bin/env bash
# Build the driver that times the original code (tag baseline-2026-09):
#   <outDir>/bin/mospBench_asis  original flags (host code at -O0)
#   <outDir>/bin/mospBench_O3    host code at -O3
# Both from `git archive baseline-2026-09` plus stage-timers.patch (stage
# timers in parallelSOSPUpdate, parallelCombinedGraph and
# sequentialSOSPUpdate; no other change), prof.cuh and mospBench.cu. The
# only build change is the architecture (the original Makefile targets
# sm_70, which CUDA 13 no longer supports) and -lineinfo.
#
# usage: bench/baseline/build.sh [outDir]   (default bench-output/baseline)
# Environment: NVCC (default nvcc), CUDA_ARCH (default sm_86)
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../.." && pwd)
out=$(mkdir -p "${1:-$repo/bench-output/baseline}" && cd "${1:-$repo/bench-output/baseline}" && pwd)
nvcc=${NVCC:-nvcc}; arch=${CUDA_ARCH:-sm_86}
src=$out/src
rm -rf "$src"; mkdir -p "$src" "$out/bin"
git -C "$repo" archive baseline-2026-09 | tar -x -C "$src"
patch -s -d "$src" -p1 < "$here/stage-timers.patch"
cp "$here/prof.cuh" "$src/headers/"
cp "$here/mospBench.cu" "$src/src/"
sources=(src/mospBench.cu src/read.cu src/Dijkstra.cu src/sequentialSOSPUpdate.cu
         src/parallelSOSPUpdate.cu src/parallelCombinedGraph.cu)
for variant in asis O3; do
  host=(); [[ $variant == O3 ]] && host=(-O3)
  (cd "$src" && "$nvcc" -std=c++17 -Iheaders --extended-lambda -arch="$arch" \
      -lineinfo "${host[@]}" -o "$out/bin/mospBench_$variant" "${sources[@]}")
  echo "built $out/bin/mospBench_$variant"
done
