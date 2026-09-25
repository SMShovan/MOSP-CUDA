# Results

Measurements of the `fix/correctness-perf` branch (`bin/mosp` built at
`12e2c8d`) against the original code (`baseline-2026-09`). The fixes and
optimizations are described in [CHANGES.md](../CHANGES.md).

- **(a) GPU compute**: the GPU work of the K SOSP updates and of Steps 2-3
  (combined graph + its SOSP), the region timed in the DynaMOSP paper (see
  References in ../README.md). For the original
  code: the GPU propagation loops plus the BFS post-passes of the K + 1 SOSP
  calls, without its host Step 1, the `cudaFree` stalls and the host
  `std::map` of the combined graph (listed separately where relevant). The
  new code's (a) includes Step 1 and the combined-graph construction, both
  on the GPU, but not applying the batch and the upload (which builds the
  reverse graph), as the original's graph building is not counted either.
- **(b) end to end**: from reading the text inputs (CSR, batch, initial
  trees) to writing all outputs (K trees and distances, MOSP tree and
  distances, MOSP costs); the same files for both codes.

## Method

- **Hardware:** 2x RTX A5000 (24 GB, sm_86), Xeon Gold 6258R (28 cores, 56
  threads), 124 GB RAM; CUDA 13.1, g++ 12.2. The machine is shared with
  other jobs.
- **GPU runs** hold a shared lock (`flock`) so that no other timed job runs
  at the same time, use GPU 1, and record the processes on that GPU before
  and after each run; runs with another process on the GPU were discarded
  and repeated. A few original-code runs ran on GPU 0 under the same lock
  with no other tenant. `CUDA_MODULE_LOADING=EAGER` is set for both codes;
  this moves the original's lazy kernel loading (0.25 s on roadNet-CA, 4 s
  on road_usa) out of its first timed update, i.e. in the original's favour.
- **CPU runs** (OpenMP): 28 threads pinned to the physical cores
  (`OMP_NUM_THREADS=28 OMP_PROC_BIND=close OMP_PLACES=cores`), not
  concurrent with the other runs of this campaign.
- **Repetitions:** median of 3 runs unless noted.
- **Original code:** `baseline-2026-09` built with only the architecture fix
  (as-is: host code at `-O0`, the original Makefile's behaviour) and with
  host code at `-O3`, driven by a thin harness that calls the original
  functions (K x `parallelSOSPUpdate`, then `parallelCombinedGraph`) on the
  same input files and adds stage timers with device synchronization at the
  stage boundaries: [bench/baseline/](../bench/baseline/) (`build.sh`
  builds `mospBench_asis` and `mospBench_O3` from the tag plus
  `stage-timers.patch`; `run.sh` reports the medians of (a) and (b) as
  defined above). A re-run on roadNet-PA (50K safe, -O3, loaded host) gave
  (a) 117 ms and (b) 20.3 s, all distances PASS, against the 113 ms and
  19.4 s below.
- **Inputs** (SuiteSparse): roadNet-PA (1.09M vertices, 3.08M directed
  edges), roadNet-CA (1.97M, 5.53M), rgg_n_2_20_s0 (1.05M, 13.8M), road_usa
  (23.9M, 57.7M); symmetric matrices get both edge directions; K = 3
  uniform weights in [1, 100] (seed 12345, `bin/mospPrep mtx2csr`); source
  0; initial trees by Dijkstra (`bin/mospPrep init`). Batches (seed 777,
  `bin/mospPrep changes`):
  - *safe*: 50K changes, 50% deletions sampled from the existing edges,
    minus the deletions that would disconnect a vertex from the source
    (`--safe`; kept 23,201 / 23,319 / 25,000 / 21,884 deletions);
  - *unsafe*: the same 50K batch unfiltered (about 7% of the deletions cut
    vertices off on the road graphs; for rgg no deletion disconnects, so
    unsafe = safe and it is not listed);
  - *local*: 10K changes (50% deletions, connectivity-safe) with every
    endpoint within a BFS ball of 110-200 hops around one random vertex
    (`--local`; 104K-121K vertices in the ball);
  - K sweep: roadNet-CA widened to 4 objectives (`mospPrep widen`), 50K
    safe batch generated for it, run with K = 2, 3, 4.
- **Validation:** the original harness compares distances with Dijkstra
  (PASS in every run with a reference). The new code was run with
  `--validate --canonicalize` on every graph and batch kind, and on the
  4-objective graph with K = 2 (Pref 1,3) and K = 4 (Pref 4,1,4,2): all
  trees and the MOSP tree identical to Dijkstra (distances and parents).

## K = 3, 50K changes (50% deletions), connectivity-safe

| graph | original as-is (b) | original -O3 (a) | original -O3 (b) | new (a) | new (b) | (a) speedup | (b) speedup vs as-is | (b) speedup vs -O3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| roadNet-PA | 41.8 s | 113 ms | 19.4 s | 18.5 ms | 794 ms | 6.1x | 52.6x | 24.4x |
| roadNet-CA | 69.9 s | 232 ms | 34.6 s | 33.7 ms | 1.25 s | 6.9x | 55.7x | 27.6x |
| rgg_n_2_20_s0 | 126 s | 511 ms | 58.2 s | 83.0 ms | 1.63 s | 6.2x | 77.5x | 35.7x |
| road_usa | – | 5.07 s | 444 s | 377 ms | 12.2 s | 13.5x | – | 36.5x |

(a) of the as-is build equals that of the -O3 build within 2% (the GPU code
is the same); road_usa was not run as-is (about 2x the -O3 time, over the
10-minute budget of a run).

Per step (original -O3 / new):

| graph | SOSP update per objective: original / new | Steps 2-3 original: host map + GPU SOSP | Steps 2-3 new (GPU) | invalidated per objective (new) |
|---|---:|---:|---:|---:|
| roadNet-PA | 34.1 ms / 5.21 ms | 489 ms + 7.28 ms | 2.84 ms | 1.05M |
| roadNet-CA | 73.4 ms / 9.43 ms | 876 ms + 12.8 ms | 5.32 ms | 1.10M |
| rgg_n_2_20_s0 | 167 ms / 26.5 ms | 691 ms + 16.3 ms | 3.99 ms | 0.77M |
| road_usa | 1.63 s / 105 ms | 14.6 s + 194 ms | 61.5 ms | 16.31M |

## K = 3, 50K changes, unfiltered (deletions disconnect vertices)

The original runs into its `maxIterations = n` cap (roadNet-CA: 1,971,281
iterations per objective). That loop is a sequence of host round trips
whose speed depends on the host load: the three -O3 runs on roadNet-CA took
376, 949 and 1,256 s end to end. The table therefore uses the **best**
original run for each column. Runs: roadNet-PA 3 x -O3 (as-is not run),
roadNet-CA 1 x as-is and 3 x -O3; road_usa was not run with the original
code (about 24M iterations per objective, estimated > 1 h).

| graph | original as-is (b) | original -O3 (a) | original -O3 (b) | new (a) | new (b) | (a) speedup | (b) speedup vs as-is | (b) speedup vs -O3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| roadNet-PA | – | 117 s | 136 s | 17.8 ms | 747 ms | 6,570x | – | 182x |
| roadNet-CA | 353 s | 342 s | 376 s | 33.6 ms | 1.25 s | 8,482x | 283x | 301x |
| road_usa | – | – | – | 378 ms | 12.0 s | – | – | – |

The roadNet-CA (a) speedup is against the as-is run (285 s, the best (a)).

| graph | SOSP update per objective: original / new | Steps 2-3 original: host map + GPU SOSP | Steps 2-3 new (GPU) | invalidated per objective (new) |
|---|---:|---:|---:|---:|
| roadNet-PA | 39.1 s / 5.02 ms | 470 ms + 7.27 ms | 2.73 ms | 1.05M |
| roadNet-CA | 112 s / 9.38 ms | 883 ms + 13.1 ms | 5.33 ms | 1.11M |
| road_usa | – / 105 ms | – | 61.5 ms | 16.31M |

## K = 3, 10K local changes (50% deletions, connectivity-safe)

| graph | original as-is (b) | original -O3 (a) | original -O3 (b) | new (a) | new (b) | (a) speedup | (b) speedup vs as-is | (b) speedup vs -O3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| roadNet-PA | 40.9 s | 130 ms | 20.1 s | 39.7 ms | 842 ms | 3.2x | 48.6x | 23.9x |
| roadNet-CA | 74.5 s | 85.9 ms | 37.1 s | 21.0 ms | 1.18 s | 4.1x | 62.9x | 31.3x |
| rgg_n_2_20_s0 | 125 s | 1.21 s | 58.8 s | 227 ms | 1.80 s | 5.3x | 69.5x | 32.6x |
| road_usa | – | 1.98 s | 439 s | 179 ms | 11.0 s | 11.0x | – | 39.9x |

| graph | SOSP update per objective: original / new | Steps 2-3 original: host map + GPU SOSP | Steps 2-3 new (GPU) | invalidated per objective (new) |
|---|---:|---:|---:|---:|
| roadNet-PA | 34.6 ms / 10.5 ms | 470 ms + 26.0 ms | 8.07 ms | 0.48M |
| roadNet-CA | 16.2 ms / 3.82 ms | 916 ms + 35.4 ms | 9.89 ms | 0.09M |
| rgg_n_2_20_s0 | 365 ms / 68.3 ms | 655 ms + 107 ms | 19.6 ms | 0.27M |
| road_usa | 231 ms / 22.4 ms | 15.1 s + 1.28 s | 112 ms | 0.25M |

## K = 2, 3, 4 (roadNet-CA with 4 objectives, 50K safe batch)

| K | original -O3 (a) / (b) | new CUDA (a) / (b) | new OpenMP (a) / (b) | CUDA (a) speedup vs original | CUDA (b) speedup vs original |
|---|---:|---:|---:|---:|---:|
| 2 | 160 ms / 28.9 s | 23.6 ms / 1.27 s | 87.2 ms / 1.27 s | 6.8x | 22.7x |
| 3 | 234 ms / 38.2 s | 33.6 ms / 1.26 s | 101 ms / 1.13 s | 7.0x | 30.4x |
| 4 | 309 ms / 48.2 s | 43.4 ms / 1.43 s | 126 ms / 1.18 s | 7.1x | 33.7x |

The new (a) grows by about 10 ms per objective; (b) changes little with K
because reading the 4-weight CSR dominates it.

## Ablation (roadNet-CA, K = 3; GPU work per step)

Every row except the first is `bin/mosp` built at the listed commit and run
with `--timing` (the sum of its GPU stages); rows up to MP3 predate the
in-memory pipeline, so their driver still reads and writes the text files
per step, which is not part of these times. The first row is the original
code (`bench/baseline/`). "Combined step" is
the GPU part of Steps 2-3 (for the original and the commits before MP3 only
the SOSP on the combined graph; the host map is excluded).

| step (commit) | 50K safe: update per objective | 50K safe: combined step (GPU) | 50K safe: (a) | 10K local: update per objective | 10K local: combined step (GPU) | 10K local: (a) |
|---|---:|---:|---:|---:|---:|---:|
| original (-O3 host) | 73.4 ms | 12.8 ms | 232 ms | 16.2 ms | 35.4 ms | 85.9 ms |
| M-a, M-b: invalidation + monotone loop (6405a35) | 66.1 ms | 10.5 ms | 204 ms | 8.14 ms | 25.9 ms | 50.3 ms |
| MP1: pull + near-far push, host loop (f9c0324) | 9.91 ms | 5.35 ms | 35.2 ms | 5.18 ms | 12.4 ms | 27.4 ms |
| MP2: persistent kernel, first version (6b199db) | 12.8 ms | 7.45 ms | 46.0 ms | 3.10 ms | 8.21 ms | 17.3 ms |
| MP3: combined graph on the GPU (0d1d3ee) | 12.8 ms | 10.1 ms | 48.6 ms | 3.09 ms | 10.8 ms | 19.8 ms |
| final: in-memory pipeline + aggregated appends (12e2c8d) | 9.43 ms | 5.32 ms | 33.7 ms | 3.82 ms | 9.89 ms | 21.0 ms |

- M-a alone changes little on connectivity-safe batches; its gain is the
  disconnecting case (see above) and correctness.
- The work-efficient push (MP1) is the main gain on large batches.
- The persistent kernel (MP2) helps local batches (no host round trip per
  iteration), but its first version lost on 50K batches because the list
  appends were not warp-aggregated (see CHANGES.md, MP2); the final version
  fixes that. On the local batch the final update (3.82 ms per objective) is
  slower than the first MP2 version (3.10 ms): the explicit aggregation and a
  second barrier per iteration cost about 1 us per iteration (~620
  iterations).

## End-to-end breakdown (K = 3, 50K safe)

| graph | original -O3: text reads / host work / text writes / total | new: context / read / apply batch / upload / (a) / download / write / total |
|---|---|---|
| roadNet-PA | 14.7 s / 2.47 s / 1.41 s / 19.4 s | 201 ms / 271 ms / 56.7 ms / 25.5 ms / 18.5 ms / 12.2 ms / 202 ms / 794 ms |
| roadNet-CA | 26.5 s / 4.20 s / 2.51 s / 34.6 s | 194 ms / 521 ms / 79.8 ms / 43.5 ms / 33.7 ms / 26.1 ms / 353 ms / 1.25 s |
| rgg_n_2_20_s0 | 49.4 s / 5.34 s / 1.83 s / 58.2 s | 177 ms / 864 ms / 125 ms / 56.0 ms / 83.0 ms / 13.5 ms / 263 ms / 1.63 s |
| road_usa | 324 s / 60.6 s / 34.5 s / 444 s | 182 ms / 5.25 s / 633 ms / 500 ms / 377 ms / 278 ms / 4.79 s / 12.2 s |

The original's host work is building adjacency lists, applying the batch,
flattening to CSR and the combined-graph map, repeated per call.

### Binary graph cache (`--cache`, K = 3, 50K safe)

| graph | new, text inputs: (b) | new, binary graph cache: (b) | reading inputs (text / cache) |
|---|---:|---:|---:|
| roadNet-PA | 794 ms | 685 ms | 271 ms / 99.2 ms |
| roadNet-CA | 1.25 s | 926 ms | 521 ms / 124 ms |
| rgg_n_2_20_s0 | 1.63 s | 950 ms | 864 ms / 138 ms |
| road_usa | 12.2 s | 8.37 s | 5.25 s / 1.21 s |

## Near-far bucket width Delta (update per objective, K = 3)

The default is 32 x average weight / average out-degree (about 570-670 on
the road graphs, 122 on rgg, 34-42 on the combined graphs). One run per
cell, final kernel.

| graph | batch | default Delta | Delta = 30 | 60 | 100 | 200 | 250 | 400 | 500 | 1000 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| roadNet-CA | safe | 9.20 ms |  |  | 10.4 ms | 9.30 ms |  | 9.05 ms |  | 11.0 ms |
| roadNet-CA | local | 3.81 ms |  |  | 5.89 ms | 4.77 ms |  | 4.05 ms |  | 3.36 ms |
| rgg_n_2_20_s0 | safe | 26.8 ms | 20.9 ms | 22.4 ms |  |  | 38.2 ms |  | 66.0 ms |  |
| rgg_n_2_20_s0 | local | 62.9 ms | 92.2 ms | 75.0 ms |  |  | 55.6 ms |  | 57.5 ms |  |
| road_usa | safe | 105 ms |  |  |  | 107 ms |  | 104 ms |  | 109 ms |
| road_usa | local | 22.2 ms |  |  |  | 28.6 ms |  | 24.3 ms |  | 21.8 ms |

The default is within ~15% of the best fixed Delta everywhere except rgg
with the 50K batch (26.8 ms vs 20.9 ms at Delta = 30), and the rgg local
batch prefers a larger Delta (250): no single value is best, so the
heuristic stays; `--delta` sets it.

## CUDA vs OpenMP (new code, identical inputs)

OpenMP: MOSP-OpenMP `bin/mosp` of the same branch, 28 pinned threads.
Both produce byte-identical trees.

| graph | batch | OpenMP (a) | CUDA (a) | CUDA speedup (a) | OpenMP (b) | CUDA (b) |
|---|---|---:|---:|---:|---:|---:|
| roadNet-PA | safe | 63.2 ms | 18.5 ms | 3.4x | 645 ms | 794 ms |
| roadNet-PA | unsafe | 63.6 ms | 17.8 ms | 3.6x | 645 ms | 747 ms |
| roadNet-PA | local | 95.9 ms | 39.7 ms | 2.4x | 626 ms | 842 ms |
| roadNet-CA | safe | 110 ms | 33.7 ms | 3.3x | 1.17 s | 1.25 s |
| roadNet-CA | unsafe | 109 ms | 33.6 ms | 3.3x | 1.14 s | 1.25 s |
| roadNet-CA | local | 74.8 ms | 21.0 ms | 3.6x | 1.03 s | 1.18 s |
| rgg_n_2_20_s0 | safe | 166 ms | 83.0 ms | 2.0x | 1.30 s | 1.63 s |
| rgg_n_2_20_s0 | local | 241 ms | 227 ms | 1.1x | 1.65 s | 1.80 s |
| road_usa | safe | 1.32 s | 377 ms | 3.5x | 12.9 s | 12.2 s |
| road_usa | unsafe | 1.31 s | 378 ms | 3.5x | 13.6 s | 12.0 s |
| road_usa | local | 707 ms | 179 ms | 3.9x | 11.2 s | 11.0 s |

The GPU is 2.0-3.9x faster on (a) except for the rgg local batch (1.1x:
about 1,100 near-far iterations per objective and 7,300 for the combined
step, each with little work). End
to end both are dominated by text I/O; the CUDA run also pays ~0.2 s of
context creation plus the host-device copies, so the OpenMP build is
slightly faster end to end on the three smaller graphs.

## Reproducing

```
bench/prepare.sh roadNet-CA.mtx g 3          # CSR, initial trees, 50K batches
bin/mospPrep changes g/csr/graphCsr g/changes_local_10000_50_safe \
    --changes 10000 --ins 50 --local 160 --safe --seed 777
GPU_LOCK=/path/to/lock bench/run.sh g g/changes_50000_50_safe 3
```

Local-batch radii: 110 hops (roadNet-PA, rgg), 160 (roadNet-CA), 200
(road_usa).

Raw logs are not kept in the repository.
