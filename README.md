# MOSPCUDA

CUDA implementation of the Multi-Objective Shortest Path (MOSP) algorithm,
ported from MOSPOpenMP. Uses CUDA kernels in place of OpenMP parallel regions.

## Requirements

- NVIDIA GPU with CUDA support
- CUDA Toolkit (nvcc compiler)
- C++17 compatible host compiler

## Build

From the project root:

```
make
```

To specify a different CUDA architecture (default: sm_70):

```
make CUDA_ARCH=sm_80
```

## Run

From the project root:

```
./bin/main
```

Or build and run in one step:

```
make run
```

The app writes the output graph to:
- `data/graph.mtx`
- `data/originalGraph/graphCsrRowPtr.txt`
- `data/originalGraph/graphCsrColInd.txt`
- `data/originalGraph/graphCsrValues.txt`

Then it applies `output/changedEdges/*.txt` changes and writes:
- `data/updatedGraph/updatedGraphCsrRowPtr.txt`
- `data/updatedGraph/updatedGraphCsrColInd.txt`
- `data/updatedGraph/updatedGraphCsrValues.txt`

Then it runs Dijkstra and writes original-graph results to:
- `output/distancesTrees/distances.txt`
- `output/distancesTrees/SSSPTree.txt`
- `output/distancesTrees/distancesCsr.txt`
- `output/distancesTrees/SSSPTreeCsr.txt`

Then it runs Dijkstra on the updated CSR graph and writes:
- `output/updatedDistancesTrees/updatedDistancesCsr.txt`
- `output/updatedDistancesTrees/updatedSSSPTreeCsr.txt`

Then it runs the Sequential SOSP Update algorithm (incremental update without
recomputing Dijkstra from scratch) and writes:
- `output/sospUpdateDistancesTrees/distancesCsr.txt`
- `output/sospUpdateDistancesTrees/SSSPTreeCsr.txt`

It also generates edge-change files:
- `output/changedEdges/insert.txt`
- `output/changedEdges/delete.txt`

## Test Cases

The app also generates 10 deterministic test cases under `tests/testCaseN/`, each containing:
- `originalGraph/` (CSR files)
- `changedEdges/` (insert.txt, delete.txt)
- `updatedGraph/` (CSR files after applying changes)
- `expected/` (ground truth distances and SSSP trees for original and updated graphs,
  plus SOSP update algorithm output for comparison)

These are seeded for reproducibility and vary across graph size, objective count,
change ratio, and Dijkstra objective index.

## Stress Tests

### Sequential Stress Test

```
make stressTest
./bin/stressTest
```

Runs 100 random graph configurations comparing sequential SOSP update against
Dijkstra ground truth.

### CUDA Parallel Stress Test

```
make parallelStressTest
./bin/parallelStressTest
```

Runs 100 random graph configurations comparing the CUDA parallel SOSP update
against Dijkstra ground truth.

## Doxygen

Generate docs from the project root:

```
doxygen Doxyfile
```

Open the HTML output:

```
open html/index.html
```
