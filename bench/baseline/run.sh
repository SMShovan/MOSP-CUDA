#!/usr/bin/env bash
# Time the original code on one prepared input (see bench/prepare.sh) and
# report the medians of
#   (a) the GPU propagation loops plus the BFS post-passes of the K + 1
#       SOSP calls (without the cudaFree stall inside the BFS stage), and
#   (b) the total wall time (reading the text inputs to writing all
#       outputs),
# as in results/README.md.
#
# usage: bench/baseline/run.sh <mospBench> <graphDir> <changesDir> [reps] [K]
#   <mospBench>  bench-output/baseline/bin/mospBench_{asis,O3} (build.sh)
#   reps         number of runs (default 3); K objectives (default 3)
# If <changesDir>/expected/obj<k>/distancesUpdated.txt exist (`bin/mospPrep
# expected <csr> <changesDir> <changesDir>/expected`), the distances are
# checked against them.
#
# Environment:
#   GPU_LOCK   if set, every run holds `flock $GPU_LOCK` (shared GPUs)
#   OUT_DIR    where logs go (default bench-output/<date>); the outputs of
#              each run are deleted after the run
set -euo pipefail
here=$(cd "$(dirname "$0")/../.." && pwd)
bench=${1:?mospBench}; graph=${2:?graphDir}; changes=${3:?changesDir}
reps=${4:-3}; K=${5:-3}
[[ $reps =~ ^[1-9][0-9]*$ && $K =~ ^[1-9][0-9]*$ ]] ||
  { echo "reps and K must be positive integers" >&2; exit 2; }
out=${OUT_DIR:-$here/bench-output/$(date +%Y%m%d)}
mkdir -p "$out"
name="$(basename "$bench")_$(basename "$graph")_$(basename "$changes")"
expected=()
[[ -d $changes/expected ]] && expected=(--expected "$changes/expected")
lock=()
[[ -n "${GPU_LOCK:-}" ]] && lock=(flock "$GPU_LOCK")

a=(); b=()
for (( r = 1; r <= reps; r++ )); do
  log="$out/${name}_r$r.log"
  run_out="$out/${name}_r${r}_out"
  "${lock[@]}" env CUDA_MODULE_LOADING=EAGER "$bench" "$graph/csr/graphCsr" \
      "$K" 0 "$changes" "$graph/init" "$run_out" "${expected[@]}" > "$log" 2>&1
  rm -rf "$run_out"
  if grep -q '^VALIDATE.*FAIL' "$log"; then
    echo "$name run $r: distances differ from $changes/expected" >&2; exit 1
  fi
  read -r ta tb < <(awk '
    $1 == "STAGE" && $2 ~ /2d_propagate_loop_gpu$/ { a += $3 }
    $1 == "COUNTER" && $2 ~ /bfs_us_(loop|mark)$/  { a += $3 / 1000 }
    $1 == "TOTAL_WALL_MS" { b = $2 }
    END { if (b == "") exit 1; printf "%.3f %.3f\n", a, b }' "$log") ||
    { echo "$name run $r: no TOTAL_WALL_MS line (see $log)" >&2; exit 1; }
  a+=("$ta"); b+=("$tb")
  echo "$name run $r: a_ms=$ta b_ms=$tb"
done
median() { printf '%s\n' "$@" | sort -g | awk '{v[NR]=$1} END {print (NR%2 ? v[(NR+1)/2] : (v[NR/2]+v[NR/2+1])/2)}'; }
echo "MEDIAN $name a_ms=$(median "${a[@]}") b_ms=$(median "${b[@]}") reps=$reps"
