#!/usr/bin/env bash
# Run bin/mosp repeatedly on one prepared input and report the median of
# the GPU-compute (a) and end-to-end (b) times.
#
# usage: bench/run.sh <graphDir> <changesDir> [reps] [-- extra mosp options]
#
#   <graphDir>    directory with csr/graphCsr{RowPtr,ColInd,Values}.txt and
#                 init/obj<k>/{distancesOriginal,SSSPTreeOriginal}.txt
#                 (see bench/prepare.sh)
#   <changesDir>  directory with insert.txt and delete.txt
#   reps          number of runs (default 3)
#
# Environment:
#   GPU_LOCK   if set, every run holds `flock $GPU_LOCK` (shared GPUs)
#   OUT_DIR    where logs and outputs go (default bench-output/<date>);
#              outputs of each run are deleted after the run
#   MOSP_BIN   driver to run (default bin/mosp next to this script)
set -euo pipefail

here=$(cd "$(dirname "$0")/.." && pwd)
graph=${1:?graphDir}; changes=${2:?changesDir}; reps=${3:-3}
shift $(( $# < 3 ? $# : 3 ))
[[ "${1:-}" == "--" ]] && shift
extra=("$@")
bin=${MOSP_BIN:-$here/bin/mosp}
out=${OUT_DIR:-$here/bench-output/$(date +%Y%m%d)}
mkdir -p "$out"
name="$(basename "$graph")_$(basename "$changes")"

lock=()
[[ -n "${GPU_LOCK:-}" ]] && lock=(flock "$GPU_LOCK")

gpu=(); e2e=()
for r in $(seq 1 "$reps"); do
  log="$out/${name}_r$r.log"
  run_out="$out/${name}_r${r}_out"
  "${lock[@]}" env CUDA_MODULE_LOADING=EAGER "$bin" \
      --graph "$graph/csr/graphCsr" --changes "$changes" --init "$graph/init" \
      --out "$run_out" "${extra[@]}" > "$log" 2>&1
  rm -rf "$run_out"
  line=$(grep '^RESULT' "$log")
  g=$(sed -E 's/.*gpu_compute_ms=([0-9.eE+-]+).*/\1/' <<< "$line")
  t=$(sed -E 's/.*end_to_end_ms=([0-9.eE+-]+).*/\1/' <<< "$line")
  gpu+=("$g"); e2e+=("$t")
  echo "$name run $r: gpu_compute_ms=$g end_to_end_ms=$t"
done
median() { printf '%s\n' "$@" | sort -g | awk '{a[NR]=$1} END {print (NR%2 ? a[(NR+1)/2] : (a[NR/2]+a[NR/2+1])/2)}'; }
echo "MEDIAN $name gpu_compute_ms=$(median "${gpu[@]}") end_to_end_ms=$(median "${e2e[@]}") reps=$reps"
