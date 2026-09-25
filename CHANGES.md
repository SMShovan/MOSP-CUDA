# Changes on `fix/correctness-perf`

Branch point: `baseline-2026-09` (`ac29545`, "Ported from SYCL to CUDA").
Every commit builds and passes `make test`. Measurements: see
[results/README.md](results/README.md) (method, hardware, full tables).

Two scopes are reported everywhere:

- **(a) GPU compute**: the region the papers time, i.e. the GPU work of the K
  SOSP updates and of Steps 2-3 of the MOSP update (combined graph + its SOSP).
  For the original code this is the GPU propagation loops plus the BFS
  post-passes of the K + 1 SOSP calls (without the `cudaFree` stalls inside
  the BFS stage); its host Step 1 and the host `std::map` that built the
  combined graph are *not* counted, so the comparison is conservative. For
  the new code it includes Step 1 and the combined-graph construction but
  not applying the batch and uploading the graph (the reverse graph is built
  during the upload), just as the original's graph building is not counted.
- **(b) end to end**: wall time from reading the inputs (text CSR, change
  batch, initial trees) to writing all outputs.

"Original" is `baseline-2026-09` with only the architecture fix (`sm_86`,
host code at `-O0` as in the original Makefile); "original -O3" is the same
code with host code at `-O3`, both timed with the driver and stage timers in
[bench/baseline/](bench/baseline/). Medians of 3 runs on an RTX A5000 under
an exclusive GPU lock; K = 3; 50K changes with 50% deletions.

## Summary

- **Correctness.** The SOSP update could return *too small distances for
  reachable vertices* (count to infinity cut off by an iteration cap; about 1
  in 500 random stress cases, in the CUDA, OpenMP and sequential code alike)
  and needed n iterations whenever a batch disconnects vertices. Fixed by
  subtree invalidation plus a monotone update (M-a). Parent ties are now
  broken by the lowest vertex id everywhere (M-c): every updated tree equals
  the Dijkstra tree of the updated graph exactly, runs are deterministic and
  CUDA and OpenMP outputs are byte-identical. The preference vector of the
  thesis is implemented (M-d), and batches can be generated reproducibly in
  several modes (M-e).
- **Performance** (connectivity-safe 50K batch, K = 3):

| graph | (a) original -O3 | (a) new | (a) speedup | (b) original as-is / -O3 | (b) new | (b) speedup vs as-is / -O3 |
|---|---:|---:|---:|---:|---:|---:|
| roadNet-PA | 113 ms | 18.5 ms | 6.1x | 41.8 s / 19.4 s | 794 ms | 52.6x / 24.4x |
| roadNet-CA | 232 ms | 33.7 ms | 6.9x | 69.9 s / 34.6 s | 1.25 s | 55.7x / 27.6x |
| rgg_n_2_20_s0 | 511 ms | 83.0 ms | 6.2x | 126 s / 58.2 s | 1.63 s | 77.5x / 35.7x |
| road_usa | 5.07 s | 377 ms | 13.5x | not run / 444 s | 12.2 s | – / 36.5x |

  On the unfiltered batch (deletions disconnect vertices) the original needs
  n iterations per objective: roadNet-CA 285-912 s (a) against 33.6 ms
  (8,482x against the best original run). On a local 10K batch (a) improves
  3.2-11.0x. With the optional binary graph cache (b) drops further (road_usa
  12.2 s -> 8.4 s). The combined-graph regression of the first persistent
  kernel (7.5-10 ms against 5.35 ms with MP1 on roadNet-CA) is resolved
  (5.32 ms, see MP2).

## Correctness

### M-a: count-to-infinity on disconnection, and wrong distances (6405a35)

Deleting (or raising the weight of) a tree edge (u,v) made Step 1 pick the
best *current* in-neighbour of v as its new parent. That can be a descendant
of v; the stale cycle then only grows by its own weight per round ("count to
infinity"). The code capped the loop at `maxIterations = n` and repaired
unreachable vertices with a BFS post-pass. Two consequences:

1. **Wrong results.** When a stale cycle needs more than n rounds to exceed
   the real alternative path (small n, large weights), the loop stops with
   distances that are too small for *reachable* vertices, which the BFS does
   not touch. Found by the stock stress tests (about 1 failing case in 500
   random ones, CUDA and OpenMP and the sequential reference alike);
   `mospTest` keeps three such cases as regression tests (e.g. n = 6,
   graphSeed 621705, changeSeed 250813: vertex 1 gets 60 instead of 90).
2. **Pathological run time.** With the repository's own uniform random
   deletions about 7% disconnect vertices on road graphs; the loop then runs
   the full n iterations (roadNet-CA: 1,971,281 iterations, about 95-110 s per
   objective).

Fix: the heads of deleted or weight-increased tree edges are *roots*; the
SOSP subtrees of all roots are invalidated (distance INF, parent -1) and the
update becomes monotone (a vertex only takes a strictly better
(distance, parent) pair). Distances only decrease, so the loop terminates
without an iteration cap and cut-off vertices keep INF: the cap and the BFS
post-pass are removed. The sequential reference (`sequentialSOSPUpdate`) got
the same fix. This answers the thesis' future-work item on disconnected
graphs (Ch. 4, "A Special Case").

Disconnecting batch (the 50K batch without the connectivity filter; the
original's run time depends on the host load because its loop is a sequence
of host round trips, so the table lists its **best** run; the three -O3 runs
on roadNet-CA took 376, 949 and 1,256 s end to end):

| graph | original (a) | original (b) | new (a) | new (b) | (a) speedup | (b) speedup |
|---|---:|---:|---:|---:|---:|---:|
| roadNet-PA | 117 s (-O3) | 136 s (-O3) | 17.8 ms | 747 ms | 6,570x | 182x |
| roadNet-CA | 285 s (as-is), 342 s (-O3) | 353 s (as-is), 376 s (-O3) | 33.6 ms | 1.25 s | 8,482x | 283x / 301x |
| road_usa | not run (est. > 1 h) | not run | 378 ms | 12.0 s | – | – |

Per objective the original update took 39.1 s (roadNet-PA) and 112 s
(roadNet-CA); the new one 5.0 ms and 9.4 ms. The new code's time is the same
as on the connectivity-safe batch.

### M-b: Step 1 on the GPU, grouped by destination (6405a35)

Step 1 was a serial, order-dependent host loop over the change list. It now
runs on the GPU straight from the change list: roots are marked by one
kernel over the deletions/weight increases, the subtrees are invalidated by
pointer jumping (no host synchronization per tree level), and the
invalidated vertices plus the heads of all inserted edges pull their best
(distance, parent) over their in-neighbours: one thread per destination, as
the thesis' Step 0/1 grouping intends.

### M-c: defined tie-break (07fa883)

Among in-neighbours u with equal d[u] + w(u,v), the parent of v is now the
lowest u, in every path: `runDijkstra`, `runDijkstraCSR`, the in-memory
Dijkstra, `findBestParent`, the old update kernel, Step 1 insertions, and the
new engines (the packed atomicMin keeps the lowest parent among equal
distances). The update preserves the rule, so every updated tree equals the
Dijkstra tree of the updated graph *exactly*, runs are deterministic, and
CUDA and OpenMP produce byte-identical outputs (checked on roadNet-CA,
disconnecting batch: all K trees, the MOSP tree and the cost vectors). The
stress tests now also compare the SSSP tree files. Parents of vertices with
tied distances may differ from outputs of the original code; for trees
produced elsewhere, `bin/mosp --canonicalize` normalizes the inputs.

### M-d: preference vector (c3cb626)

The combined graph only supported Pref = (1, ..., 1). `parallelCombinedGraph`,
`combinedGraphSospGpu` and `bin/mosp --pref` now implement the thesis weight
W(e) = K + 1 - sum_{i: e in T_i} 1/Pref_i, kept exact by scaling with
L = lcm(Pref) (combined distances are in units of 1/L; Pref = 1s is the old
behaviour). `mospPathCosts` implements the last line of Step 3 (the K
objective values along the MOSP tree; `combinedGraph/mospCosts.txt`).
`mospTest` reproduces the worked example of Ch. 4 ("Finding a single MOSP"):
the three updated trees of sub-figures (a)-(c) and the MOSP path to u7 with
cost (15, 3, 20) for Pref = {4, 1, 4} and (15, 24, 7) for Pref = {4, 4, 1}.
Two inconsistencies of the figures are documented in the test: the edge
u3 -> u6 of the preliminaries figure contradicts sub-figure (a) and is left
out, and sub-figure (d) labels u4 -> u5 with 3 where the formula gives 2.5
(the MOSP result is the same).

### M-e: seeded change generator (355ab67)

`generateChangeBatch` / `bin/mospPrep changes`: `uniform` (identical to
`generateChangedEdges` for the same seed; checked byte for byte against the
study's roadNet-CA batch), `targeted` (thesis workload: below-average insert
weights, deletions of distinct SOSP-tree edges), `reweight`, `increase`
(weight increases on tree edges), `--local HOPS` (batch confined to a BFS
ball) and `--safe` (drop deletions that disconnect a vertex; reproduces the
study's "safe" sets byte for byte).

### Smaller fixes

- `generateGraph` did not create `data/`; `main` failed silently on a fresh
  checkout (CUDA `main` ignored all return values, now checked) (9823893).
- `runDijkstraCSR` rejected every objective index for a graph whose batch
  deleted all edges (stress test seed 53) (17875e7).
- The stress tests take a seed and print it and the parameters of failing
  runs; `generateTestCases` reports failures through its return value.
- `bin/mosp --validate` requires canonical (lowest-id) parents only together
  with `--canonicalize`; trees from other sources are checked for distances
  and parent consistency (3e03c0b).

## Performance

### Build (7711b5f)

Default `CUDA_ARCH=sm_86` (sm_70 is not supported by CUDA 13), host code at
`-O3` (it was `-O0`: nvcc passes no `-O` to the host compiler unless asked),
`-lineinfo`, header dependencies. `-O3` alone halves the original's end-to-end
time (see the original vs original -O3 columns).

### MP1: invalidation + pull + near-far push (f9c0324)

The original Step 2 re-evaluates every candidate over all its in-edges
(about 14 n candidate evaluations per objective on roadNet-CA for a 50K
batch), with two blocking copies and an O(n) memset per iteration. The new
engine (`sospUpdateGpu.cu`) keeps Step 1 as above and replaces Step 2 by a
push-based near-far worklist (Delta-stepping variant): improved vertices
relax their out-edges with a 64-bit `atomicMin` on the packed word
(distance << b | parent). Only improved vertices are expanded, lists are
deduplicated with generation stamps (nothing is reset per iteration), the
lowest parent id wins ties by construction. Delta defaults to
32 x average weight / average out-degree (a sweep is in results/).

### MP2: one persistent cooperative kernel (6b199db, 12e2c8d)

The whole update (pack, roots, pointer jumping, invalidation, pull,
near-far loop, unpack) runs in one cooperative launch; all loop decisions
are taken on the device (grid barriers), the host copies one control block
at the end. Pointer jumping stops when no vertex is still jumping (bounded
by ceil(log2 n) + 1 rounds; the prototype used a fixed 23, too few beyond
2^23 tree depth). The MP1 host loop (one blocking copy per iteration) was
removed: besides being slower on small/local batches, it needs hundreds of
host round trips per update and is fragile under contention (with the
shared host overloaded and other processes on the same GPU, the study's
host-loop prototype took 720 ms instead of 10 ms per objective on
roadNet-CA, while its persistent version took 8.7 ms). CUDA-graph
conditional nodes were not pursued (same device-side control, split over
several kernels).

The first version of MP2 (6b199db) was about 30-40% *slower* than MP1 on
50K batches (roadNet-CA: 12.8 ms instead of 9.9 ms per objective, 7.5 ms instead
of 5.35 ms for the combined step; up to 10 ms after MP3). Cause (found by
bisecting against the study's prototype on an idle GPU, with Nsight
Compute; it was neither the `__ldcg` loads, the number of grid barriers nor
the occupancy): list appends were `atomicAdd` on a counter whose address
rotated over three slots; nvcc warp-aggregates `atomicAdd` only when it can
prove the address warp-uniform, so every push became a separate atomic on
one hot counter. 12e2c8d reserves list slots with an explicit
warp-aggregated increment (`appendIndex`, one atomic per group of converged
threads), uses fixed near/next counters (thread 0 moves the count between
two grid barriers), and replaces the two 64-bit totals that every thread of
the combined-graph count kernel updated by a warp reduction.

Ablation on roadNet-CA (K = 3; GPU work only; rows up to MP3 are the
file-based driver of the respective commit, the last row `bin/mosp`):

| step (commit) | 50K safe: update per objective | 50K safe: combined step (GPU) | 50K safe: (a) | 10K local: update per objective | 10K local: combined step (GPU) | 10K local: (a) |
|---|---:|---:|---:|---:|---:|---:|
| original (-O3 host) | 73.4 ms | 12.8 ms | 232 ms | 16.2 ms | 35.4 ms | 85.9 ms |
| M-a, M-b: invalidation + monotone loop (6405a35) | 66.1 ms | 10.5 ms | 204 ms | 8.14 ms | 25.9 ms | 50.3 ms |
| MP1: pull + near-far push, host loop (f9c0324) | 9.91 ms | 5.35 ms | 35.2 ms | 5.18 ms | 12.4 ms | 27.4 ms |
| MP2: persistent kernel, first version (6b199db) | 12.8 ms | 7.45 ms | 46.0 ms | 3.10 ms | 8.21 ms | 17.3 ms |
| MP3: combined graph on the GPU (0d1d3ee) | 12.8 ms | 10.1 ms | 48.6 ms | 3.09 ms | 10.8 ms | 19.8 ms |
| final: in-memory pipeline + aggregated appends (12e2c8d) | 9.43 ms | 5.32 ms | 33.7 ms | 3.82 ms | 9.89 ms | 21.0 ms |

For the original and the file-based commits "combined step" is only the SOSP
on the combined graph (their host `std::map` is excluded); from MP3 on it
includes building the combined graph. The final kernel is slightly faster
than MP1 on the 50K batch and 1.3-1.4x faster on the local batch, but on the
local batch it is 0.7 ms per objective slower than the first MP2 version:
the explicit aggregation and the second barrier per iteration cost about 1
us per iteration (the roadNet-CA local batch needs about 620 iterations per
objective with little work each). The 50K batches were preferred as they are
the papers' workload.

### MP3: combined graph on the GPU (0d1d3ee)

The original re-read the whole CSR text only to learn n, counted tree-edge
membership in a host `std::map` (0.9 s on roadNet-CA, 14.6 s on road_usa),
wrote the combined graph as temporary text files and ran the file-based SOSP
update on them. Now the in-edges of v are found by comparing its K parents
(one thread per vertex), a count / scan / fill pass builds the out-edge CSR,
and Step 3 is a near-far SSSP from the source with the same persistent
engine. The file-based `parallelCombinedGraph` keeps its signature (the
`workDir` argument is no longer used). Steps 2-3 together: roadNet-CA
876 ms (host map) + 12.8 ms (GPU) -> 5.32 ms; road_usa 14.6 s + 194 ms ->
61.5 ms.

### MP4: packed-word overflow (9e09482)

The packed word needs b = ceil(log2(n + 1)) bits for the parent (25 for
road_usa) and the remaining 64 - b bits for distances. Instead of a fixed
25-bit parent field (n <= 33.5M), b is chosen at run time; if
(n - 1) x maxWeight does not fit, the search keeps 64-bit distances only and
recovers the lowest-id parent in one pass over the out-edges afterwards
(sm_86 has no 128-bit atomics). Tested with weights up to 2^31 - 1 on a
320 x 320 grid, and with every weight 2 * 10^9 so that most vertices have
two tight parents and the recovery must pick the lower id
(`mospTest --only large-weights`).

### H-M1 / H-M2: in-memory pipeline (70d3105, 0c3dcc8, c0baba8)

`mospUpdate()` + `bin/mosp`: the text inputs are read once (fast parser; the
original read the K-weight CSR K + 1 times, once only for n), the batch is
applied once on the host, only the updated out-edge CSR is uploaded (the
reverse graph and the K weight columns are built on the GPU and shared by
all objectives), trees stay on the GPU between steps, all device buffers are
allocated once per run (no `cudaFree` inside the timed region; the original
had frees stalling up to 3.9 s on road_usa), kernels are loaded eagerly
(`CUDA_MODULE_LOADING=EAGER`, otherwise the first launch paid 0.25-4 s of
lazy loading inside the first update), no temporary text files, and the text
files are parsed and written concurrently. Optional binary graph cache
(`--cache`). `main` computes the initial trees before the update loop.

Where the end-to-end time goes (K = 3, 50K safe batch; original -O3 from its
stage timers):

| graph | original -O3: text reads / host work / text writes / total | new: context / read / apply batch / upload / (a) / download / write / total |
|---|---|---|
| roadNet-PA | 14.7 s / 2.47 s / 1.41 s / 19.4 s | 201 ms / 271 ms / 56.7 ms / 25.5 ms / 18.5 ms / 12.2 ms / 202 ms / 794 ms |
| roadNet-CA | 26.5 s / 4.20 s / 2.51 s / 34.6 s | 194 ms / 521 ms / 79.8 ms / 43.5 ms / 33.7 ms / 26.1 ms / 353 ms / 1.25 s |
| rgg_n_2_20_s0 | 49.4 s / 5.34 s / 1.83 s / 58.2 s | 177 ms / 864 ms / 125 ms / 56.0 ms / 83.0 ms / 13.5 ms / 263 ms / 1.63 s |
| road_usa | 324 s / 60.6 s / 34.5 s / 444 s | 182 ms / 5.25 s / 633 ms / 500 ms / 377 ms / 278 ms / 4.79 s / 12.2 s |

The new end-to-end time is now mostly text I/O. With `--cache` (binary CSR,
written on the first run) reading drops from 0.27-5.25 s to 0.10-1.21 s and
(b) to 685 ms, 926 ms, 950 ms and 8.37 s.

## Tests

- `make test`: stock pipeline + 10 test cases, both stress tests (seeded,
  distances and trees must equal Dijkstra) and `bin/mospTest` (148 cases:
  every batch kind on random graphs and road-like grids, file-based and
  in-memory paths, Dijkstra and combined-graph references with default and
  skewed Pref, determinism, thesis example, regressions, large-weight
  fallback, generator/apply equivalence).
- Stress tests: 40 additional seeds (4,000 random cases per test) pass.
- compute-sanitizer memcheck, initcheck, racecheck and synccheck are clean on
  the thesis example, the regression cases and the large-weight group of
  `mospTest`; memcheck is clean on 20 parallel stress cases.
- Real graphs: `bin/mosp --validate --canonicalize` on roadNet-PA,
  roadNet-CA, rgg_n_2_20_s0 and road_usa with the safe, disconnecting and
  local batches, and on roadNet-CA with 4 objectives for K = 2 (Pref 1,3)
  and K = 4 (Pref 4,1,4,2): every tree and the MOSP tree identical to
  Dijkstra (distances and parents).

## Behaviour changes

- Parents of vertices with tied distances follow the lowest-id rule (M-c).
- Vertices cut off by a batch get INF/-1 directly (before: after up to n
  iterations and a BFS).
- Combined-graph distances are in units of 1/L when a Pref vector is given.
- `parallelCombinedGraph` no longer writes temporary files (`workDir` unused).
- The GPU must support cooperative launches (Pascal or newer).
- Weights must be positive integers (as before; now documented).

## Not done / deviations

- The update-vs-recompute selector and a static near-far recompute baseline
  (point 5) are parked as decided; `sospFromScratchGpu` exists only because
  Step 3 needs it.
- Provenance of the published numbers was not checked (as decided).
- MP2 keeps one engine (the persistent kernel); CUDA-graph conditional nodes
  were not implemented, and the host-loop variant of MP1 was removed rather
  than kept as a tuned alternative.
- MP4 uses a run-time packing width plus a distance-only fallback instead of
  128-bit atomics (not available on sm_86).
- Pointer jumping checks convergence instead of running a fixed number of
  rounds.
- The file-based `parallelSOSPUpdate` (used by `main`, the stress tests and
  the test cases) still builds adjacency lists per call; the fast path is the
  in-memory `mospUpdate` used by `bin/mosp`.

## Known issues and risks

- **Delta.** The default bucket width is within ~15% of the best fixed value
  on every measured set except rgg with the 50K batch (26.8 ms vs 20.9 ms
  per objective at Delta = 30), and the rgg local batch prefers a larger
  Delta (250); no single value is best. `--delta` overrides it.
- **Small batches on rgg.** On the rgg local batch the GPU is only 1.1x
  faster than the 28-thread OpenMP build (long, thin near-far epochs).
- **Local batches** are 0.7 ms per objective slower than with the first
  persistent kernel (see the ablation); the trade favours the 50K batches.
- **Cooperative launch.** The kernel's grid is sized from the occupancy of
  the device; a launch that fails (e.g. under MPS limits) is reported as an
  error, there is no non-cooperative fallback.
- **Distance-only fallback** (packing overflow) is tested on grids up to
  320 x 320 only; it costs one extra pass over the out-edges and is not used
  by any of the measured graphs.
- **Index width.** Vertex ids, edge offsets and weights are 32-bit, as in
  the original (graphs up to 2^31 - 1 edges).
- **Measurements** were taken on a shared machine: GPU runs were serialized
  by a lock and discarded if another process was on the GPU, but the host
  was busy with other jobs; the original's disconnecting-batch runs varied
  by more than 3x with host load (best run reported). road_usa was not run
  with the as-is build (> 10 min) nor with the disconnecting batch
  (est. > 1 h).
- **Outputs differ from the original's** where distances tie (parents) and
  for vertices the original left with a wrong distance (M-a).
