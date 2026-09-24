/**
 * @file combinedGraphGpu.cu
 * @brief Steps 2 and 3 of the MOSP update on the GPU.
 *
 * Step 2 (combined graph): the in-edges of v in the combined graph are the
 * distinct values among Parent_1[v] .. Parent_K[v], so one thread per
 * vertex compares the K parents of its vertex ("a single thread per vertex
 * compares its parents"). Two passes build a CSR of the out-edges (the
 * children lists the push-based search needs): count the out-degree of
 * every parent, exclusive scan, fill. Nothing depends on the order of the
 * atomic fills, because Step 3 breaks distance ties by the lowest parent id.
 *
 * Step 3 (SOSP on the combined graph): a near-far SSSP from the source with
 * the same engine as the SOSP update (sospFromScratchGpu).
 */

#include "combinedGraphGpu.cuh"

#include "csrGraph.cuh"
#include "parallelCombinedGraph.cuh"

#include <cub/device/device_scan.cuh>
#include <cuda_runtime.h>

#include <algorithm>
#include <iostream>
#include <numeric>
#include <vector>

using namespace std;

namespace {

constexpr int BLOCK_SIZE = 256;

#define GPU_CHECK(call)                                                        \
  do {                                                                         \
    cudaError_t err_ = (call);                                                 \
    if (err_ != cudaSuccess) {                                                 \
      cerr << "CUDA error: " << cudaGetErrorString(err_) << " at "            \
           << __FILE__ << ":" << __LINE__ << "\n";                             \
      return false;                                                            \
    }                                                                          \
  } while (0)

int blocks(long long work) {
  return static_cast<int>((max(work, 1LL) + BLOCK_SIZE - 1) / BLOCK_SIZE);
}

/**
 * If parent k of v is the first occurrence of its value among the K
 * parents, return true and its combined-graph weight
 *   base - sum_{j : Parent_j[v] == p} prefTerm[j].
 */
__device__ __forceinline__ bool combinedEdge(const int *parents, int n, int K,
                                             int v, int k,
                                             const int *prefTerms, int base,
                                             int &p, int &weight) {
  p = parents[static_cast<size_t>(k) * n + v];
  if (p < 0) {
    return false;
  }
  for (int j = 0; j < k; ++j) {
    if (parents[static_cast<size_t>(j) * n + v] == p) {
      return false; // counted with its first occurrence
    }
  }
  weight = base - prefTerms[k];
  for (int j = k + 1; j < K; ++j) {
    if (parents[static_cast<size_t>(j) * n + v] == p) {
      weight -= prefTerms[j];
    }
  }
  return true;
}

/// Pass 1: out-degree of every parent, number of edges and weight sum.
/// The two totals are reduced per warp first: one atomic per thread on the
/// same two counters serializes (tens of ms at n = 24M).
__global__ void countEdgesKernel(const int *parents, int n, int K, int source,
                                 const int *prefTerms, int base, int *degree,
                                 unsigned long long *sums) {
  int v = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned long long edges = 0, weightSum = 0;
  if (v < n && v != source) {
    for (int k = 0; k < K; ++k) {
      int p, weight;
      if (combinedEdge(parents, n, K, v, k, prefTerms, base, p, weight)) {
        atomicAdd(&degree[p], 1);
        ++edges;
        weightSum += static_cast<unsigned long long>(weight);
      }
    }
  }
  for (int offset = 16; offset > 0; offset >>= 1) {
    edges += __shfl_down_sync(0xffffffffu, edges, offset);
    weightSum += __shfl_down_sync(0xffffffffu, weightSum, offset);
  }
  if ((threadIdx.x & 31) == 0 && edges > 0) {
    atomicAdd(&sums[0], edges);
    atomicAdd(&sums[1], weightSum);
  }
}

/// Pass 2: write every edge (p, v) into row p of the children CSR.
__global__ void fillEdgesKernel(const int *parents, int n, int K, int source,
                                const int *prefTerms, int base, int *cursor,
                                int *colInd, int *weights) {
  int v = blockIdx.x * blockDim.x + threadIdx.x;
  if (v >= n || v == source) {
    return;
  }
  for (int k = 0; k < K; ++k) {
    int p, weight;
    if (combinedEdge(parents, n, K, v, k, prefTerms, base, p, weight)) {
      int position = atomicAdd(&cursor[p], 1);
      colInd[position] = v;
      weights[position] = weight;
    }
  }
}

} // namespace

// ============================================================================
// Host helpers (declared in parallelCombinedGraph.cuh)
// ============================================================================

long long preferenceScale(const vector<int> &preferences, int K) {
  if (preferences.empty()) {
    return 1;
  }
  if (static_cast<int>(preferences.size()) != K) {
    return 0;
  }
  long long scale = 1;
  for (int pref : preferences) {
    if (pref < 1) {
      return 0;
    }
    scale = scale / gcd(scale, static_cast<long long>(pref)) * pref;
    if (scale > (1LL << 20)) {
      return 0;
    }
  }
  return scale;
}

long long combinedEdgeWeight(unsigned int treeMask,
                             const vector<int> &preferences, int K,
                             long long scale) {
  long long weight = scale * (K + 1);
  for (int i = 0; i < K; ++i) {
    if (treeMask & (1u << i)) {
      weight -= preferences.empty() ? scale : scale / preferences[i];
    }
  }
  return weight;
}

bool mospPathCosts(const CsrGraph &graph, const vector<int> &parent,
                   int source, vector<long long> &costs) {
  const int n = graph.numberOfNodes;
  const int K = graph.numberOfObjectives;
  costs.assign(static_cast<size_t>(n) * K, DISTANCE_INF);
  // Children lists of the tree, then a traversal from the source.
  vector<int> childStart(n + 1, 0), children(n);
  for (int v = 0; v < n; ++v) {
    if (v != source && parent[v] >= 0) {
      ++childStart[parent[v] + 1];
    }
  }
  for (int v = 0; v < n; ++v) {
    childStart[v + 1] += childStart[v];
  }
  vector<int> cursor(childStart.begin(), childStart.end() - 1);
  for (int v = 0; v < n; ++v) {
    if (v != source && parent[v] >= 0) {
      children[cursor[parent[v]]++] = v;
    }
  }
  for (int k = 0; k < K; ++k) {
    costs[static_cast<size_t>(source) * K + k] = 0;
  }
  vector<int> queue{source};
  for (size_t i = 0; i < queue.size(); ++i) {
    int p = queue[i];
    for (int c = childStart[p]; c < childStart[p + 1]; ++c) {
      int v = children[c];
      int edge = -1;
      for (int e = graph.rowPtr[p]; e < graph.rowPtr[p + 1]; ++e) {
        if (graph.colInd[e] == v) {
          edge = e;
          break;
        }
      }
      if (edge < 0) {
        return false;
      }
      for (int k = 0; k < K; ++k) {
        costs[static_cast<size_t>(v) * K + k] =
            costs[static_cast<size_t>(p) * K + k] + graph.weight(edge, k);
      }
      queue.push_back(v);
    }
  }
  return true;
}

CombineWorkspace::~CombineWorkspace() { release(); }

void CombineWorkspace::release() {
  cudaFree(rowPtr);
  cudaFree(cursor);
  cudaFree(colInd);
  cudaFree(weights);
  cudaFree(preferences);
  cudaFree(sums);
  cudaFree(scanStorage);
  cudaFreeHost(hostSums);
  rowPtr = cursor = colInd = weights = preferences = nullptr;
  sums = nullptr;
  hostSums = nullptr;
  scanStorage = nullptr;
  scanBytes = 0;
  capacity = 0;
  trees = 0;
}

bool CombineWorkspace::reserve(int numberOfNodes, int K) {
  if (numberOfNodes <= capacity && K <= trees) {
    return true;
  }
  release();
  const size_t n = static_cast<size_t>(max(numberOfNodes, 1));
  GPU_CHECK(cudaMalloc(&rowPtr, (n + 1) * sizeof(int)));
  GPU_CHECK(cudaMalloc(&cursor, (n + 1) * sizeof(int)));
  GPU_CHECK(cudaMalloc(&colInd, n * K * sizeof(int)));
  GPU_CHECK(cudaMalloc(&weights, n * K * sizeof(int)));
  GPU_CHECK(cudaMalloc(&preferences, K * sizeof(int)));
  GPU_CHECK(cudaMalloc(&sums, 2 * sizeof(unsigned long long)));
  GPU_CHECK(cudaMallocHost(&hostSums, 2 * sizeof(unsigned long long)));
  GPU_CHECK(cub::DeviceScan::ExclusiveSum(nullptr, scanBytes, cursor, rowPtr,
                                          static_cast<int>(n + 1)));
  GPU_CHECK(cudaMalloc(&scanStorage, max<size_t>(scanBytes, 1)));
  capacity = numberOfNodes;
  trees = K;
  return true;
}

bool combinedGraphSospGpu(const int *d_parents, int n, int K,
                          const vector<int> &preferences, int source,
                          long long delta, CombineWorkspace &ws,
                          SospWorkspace &sospWorkspace, long long *d_distances,
                          int *d_parent, CombineStats *stats) {
  CombineStats local;
  CombineStats &s = stats != nullptr ? *stats : local;
  s = CombineStats();
  if (K <= 0 || K > 32 || n <= 0 || source < 0 || source >= n) {
    cerr << "Error: invalid combined-graph parameters.\n";
    return false;
  }
  const long long scale = preferenceScale(preferences, K);
  if (scale == 0) {
    cerr << "Error: invalid preference vector (need K values >= 1).\n";
    return false;
  }
  if (!ws.reserve(n, K)) {
    return false;
  }
  // Scaled preference terms L / Pref_i and the base weight L * (K + 1).
  vector<int> terms(K);
  for (int k = 0; k < K; ++k) {
    terms[k] = static_cast<int>(preferences.empty() ? scale
                                                    : scale / preferences[k]);
  }
  const int base = static_cast<int>(scale * (K + 1));
  GPU_CHECK(cudaMemcpyAsync(ws.preferences, terms.data(), K * sizeof(int),
                            cudaMemcpyHostToDevice));

  // Step 2: count, scan, fill.
  GPU_CHECK(cudaMemsetAsync(ws.cursor, 0, (n + 1) * sizeof(int)));
  GPU_CHECK(cudaMemsetAsync(ws.sums, 0, 2 * sizeof(unsigned long long)));
  countEdgesKernel<<<blocks(n), BLOCK_SIZE>>>(d_parents, n, K, source,
                                              ws.preferences, base, ws.cursor,
                                              ws.sums);
  GPU_CHECK(cudaGetLastError());
  size_t bytes = ws.scanBytes;
  GPU_CHECK(cub::DeviceScan::ExclusiveSum(ws.scanStorage, bytes, ws.cursor,
                                          ws.rowPtr, n + 1));
  GPU_CHECK(cudaMemcpyAsync(ws.cursor, ws.rowPtr, n * sizeof(int),
                            cudaMemcpyDeviceToDevice));
  fillEdgesKernel<<<blocks(n), BLOCK_SIZE>>>(d_parents, n, K, source,
                                             ws.preferences, base, ws.cursor,
                                             ws.colInd, ws.weights);
  GPU_CHECK(cudaGetLastError());
  GPU_CHECK(cudaMemcpyAsync(ws.hostSums, ws.sums,
                            2 * sizeof(unsigned long long),
                            cudaMemcpyDeviceToHost));
  GPU_CHECK(cudaStreamSynchronize(0));

  DeviceCsr combined;
  combined.numberOfNodes = n;
  combined.numberOfEdges = static_cast<int>(ws.hostSums[0]);
  combined.rowPtr = ws.rowPtr;
  combined.colInd = ws.colInd;
  combined.weights = ws.weights;
  s.numberOfEdges = combined.numberOfEdges;
  s.scale = scale;
  s.delta = delta > 0 ? delta
                      : defaultDelta(combined.numberOfEdges, n,
                                     static_cast<long long>(ws.hostSums[1]));

  // Step 3: SSSP on the combined graph.
  return sospFromScratchGpu(combined, source, s.delta, base, sospWorkspace,
                            d_distances, d_parent, &s.search);
}
