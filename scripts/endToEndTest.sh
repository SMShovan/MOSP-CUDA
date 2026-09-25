#!/usr/bin/env bash
# End-to-end test of bin/mospPrep and bin/mosp on a small generated graph
# (run by `make test`): prepare the inputs, run the driver with --validate
# on connectivity-safe and disconnecting batches, with --canonicalize,
# -k, --pref and --cache (written, then read), and compare its output with
# `mospPrep expected` (host Dijkstra on the updated graph).
#
# usage: scripts/endToEndTest.sh <binDir> <workDir> [seed]
set -euo pipefail
bin=${1:?binDir}; work=${2:?workDir}; seed=${3:-1}
(( seed > 0 )) || seed=$(( (RANDOM << 15 | RANDOM) + 1 ))
rm -rf "$work"; mkdir -p "$work"
log=$work/endToEnd.log
: > "$log"

fail() { echo "FAIL: $*"; echo "(log: $log)"; exit 1; }
run() { # run a command, keep its output in the log, fail on non-zero exit
  echo "\$ $*" >> "$log"
  "$@" >> "$log" 2>&1 || fail "exit code $? of: $*"
}
# mosp with --validate: exit 0 and every VALIDATE line PASS; the output
# is left in $last
last=
validate() {
  echo "\$ $*" >> "$log"
  last=$("$@" --validate 2>&1) || { echo "$last" >> "$log"; fail "exit code of: $*"; }
  echo "$last" >> "$log"
  grep -q '^VALIDATE combined PASS' <<< "$last" || fail "no combined PASS: $*"
  ! grep -q '^VALIDATE .* FAIL' <<< "$last" || fail "VALIDATE FAIL: $*"
}

# A 30 x 30 grid (both directions, 5% of the links missing) plus random
# arcs: dead ends, detours and many equal-length paths.
mtx=$work/graph.mtx
awk -v seed="$seed" 'BEGIN {
  srand(seed); W = 30; H = 30; n = W * H; m = 0
  for (y = 0; y < H; ++y) for (x = 0; x < W; ++x) {
    u = y * W + x + 1
    if (x + 1 < W && rand() >= 0.05) { e[++m] = u " " (u + 1); e[++m] = (u + 1) " " u }
    if (y + 1 < H && rand() >= 0.05) { e[++m] = u " " (u + W); e[++m] = (u + W) " " u }
  }
  for (i = 0; i < 300; ++i) {
    a = int(rand() * n) + 1; b = int(rand() * n) + 1
    if (a != b) e[++m] = a " " b
  }
  print "%%MatrixMarket matrix coordinate pattern general"
  print n, n, m
  for (i = 1; i <= m; ++i) print e[i]
}' > "$mtx"

g=$work/g
run "$bin/mospPrep" mtx2csr "$mtx" "$g/csr/graphCsr" 3 1 20 "$seed"
run "$bin/mospPrep" init "$g/csr/graphCsr" "$g/init"
run "$bin/mospPrep" changes "$g/csr/graphCsr" "$g/safe" --changes 400 \
    --ins 50 --seed "$seed" --safe
run "$bin/mospPrep" changes "$g/csr/graphCsr" "$g/unsafe" --changes 1500 \
    --ins 20 --seed "$seed" # disconnects some vertices
run "$bin/mospPrep" changes "$g/csr/graphCsr" "$g/targeted" --changes 200 \
    --mode targeted --seed "$seed"
n=$(awk 'NR == 2 { print $1 }' "$mtx")

mosp=("$bin/mosp" --graph "$g/csr/graphCsr" --init "$g/init" --quiet)
for batch in safe unsafe targeted; do
  out=$work/out-$batch
  validate "${mosp[@]}" --changes "$g/$batch" --out "$out"
  # The update must equal host Dijkstra on the updated graph (trees too:
  # both use the lowest-id tie rule).
  run "$bin/mospPrep" expected "$g/csr/graphCsr" "$g/$batch" "$work/exp-$batch"
  for k in 0 1 2; do
    for f in distancesUpdated.txt SSSPTreeUpdated.txt; do
      cmp -s "$out/obj$k/$f" "$work/exp-$batch/obj$k/$f" ||
          fail "$batch obj$k/$f differs from mospPrep expected"
    done
  done
  # One line "v c1 c2 c3" per vertex.
  costs=$out/combinedGraph/mospCosts.txt
  [[ -f $costs ]] || fail "$batch: no $costs"
  awk -v n="$n" 'NF != 4 { bad = 1 } END { exit !(NR == n && !bad) }' \
      "$costs" || fail "$batch: $costs does not have $n lines of 4 fields"
done

# --canonicalize: initial trees from another tool may break ties
# differently. On a graph with weights in [1, 2] (many ties), give every
# vertex its highest-id tight parent instead of the lowest; with
# --canonicalize the result must still equal mospPrep expected.
t=$work/ties
run "$bin/mospPrep" mtx2csr "$mtx" "$t/csr/graphCsr" 2 1 2 "$seed"
run "$bin/mospPrep" init "$t/csr/graphCsr" "$t/init"
run "$bin/mospPrep" changes "$t/csr/graphCsr" "$t/changes" --changes 400 \
    --ins 50 --seed "$seed" --wmin 1 --wmax 2
for k in 0 1; do
  mkdir -p "$t/init-high/obj$k"
  cp "$t/init/obj$k/distancesOriginal.txt" "$t/init-high/obj$k/"
  awk -v k="$k" '
    FILENAME == ARGV[1] { rowPtr[FNR - 1] = $1; next }
    FILENAME == ARGV[2] { colInd[FNR - 1] = $1; next }
    FILENAME == ARGV[3] { weight[FNR - 1] = $(k + 1); next }
    FILENAME == ARGV[4] { d[$1] = $2; n = FNR; next }
    END {
      for (v = 0; v < n; ++v) parent[v] = -1
      for (u = 0; u < n; ++u) {
        if (d[u] == "INF") continue
        for (e = rowPtr[u]; e < rowPtr[u + 1]; ++e) {
          v = colInd[e]
          if (d[v] != "INF" && d[u] + weight[e] == d[v] && u > parent[v])
            parent[v] = u
        }
      }
      for (v = 0; v < n; ++v) print v, parent[v]
    }' "$t/csr/graphCsrRowPtr.txt" "$t/csr/graphCsrColInd.txt" \
      "$t/csr/graphCsrValues.txt" "$t/init/obj$k/distancesOriginal.txt" \
      > "$t/init-high/obj$k/SSSPTreeOriginal.txt"
  ! cmp -s "$t/init/obj$k/SSSPTreeOriginal.txt" \
      "$t/init-high/obj$k/SSSPTreeOriginal.txt" ||
      fail "the highest-id trees equal the canonical ones (no ties?)"
done
validate "$bin/mosp" --graph "$t/csr/graphCsr" --changes "$t/changes" \
    --init "$t/init-high" --out "$work/out-ties" --canonicalize --quiet
run "$bin/mospPrep" expected "$t/csr/graphCsr" "$t/changes" "$work/exp-ties"
for k in 0 1; do
  for f in distancesUpdated.txt SSSPTreeUpdated.txt; do
    cmp -s "$work/out-ties/obj$k/$f" "$work/exp-ties/obj$k/$f" ||
        fail "--canonicalize: obj$k/$f differs from mospPrep expected"
  done
done

# -k, --pref, and the binary cache: the first run writes it, the second
# reads it; both must give the output of the text graph.
cache=$work/graph.bin
for pass in write read; do
  validate "${mosp[@]}" --changes "$g/unsafe" -k 2 --pref 2,1 \
      --cache "$cache" --out "$work/out-cache-$pass"
  [[ -f $cache ]] || fail "--cache did not write $cache"
  if [[ $pass == write ]]; then
    stamp=$(stat -c %y "$cache")
  elif [[ $(stat -c %y "$cache") != "$stamp" ]] || grep -q Note <<< "$last"; then
    fail "--cache was rebuilt instead of read"
  fi
done
validate "${mosp[@]}" --changes "$g/unsafe" -k 2 --pref 2,1 \
    --out "$work/out-text"
for pass in write read; do
  diff -r -q "$work/out-text" "$work/out-cache-$pass" > /dev/null ||
      fail "output with --cache ($pass) differs from the text graph"
done
# `mospPrep cache` writes the same format.
run "$bin/mospPrep" cache "$g/csr/graphCsr" "$work/prep.bin"
validate "${mosp[@]}" --changes "$g/unsafe" -k 2 --pref 2,1 \
    --cache "$work/prep.bin" --out "$work/out-prep-cache"
! grep -q Note <<< "$last" || fail "the cache of mospPrep was rebuilt"
diff -r -q "$work/out-text" "$work/out-prep-cache" > /dev/null ||
    fail "output with the cache of mospPrep differs from the text graph"

# Trees of another source must be rejected.
if "${mosp[@]}" --changes "$g/safe" --source 5 --no-output >> "$log" 2>&1; then
  fail "mosp accepted initial trees of another source"
fi

echo "mosp + mospPrep end to end: 3 batches, --canonicalize, -k/--pref, --cache (seed $seed)"
