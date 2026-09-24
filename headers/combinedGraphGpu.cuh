#ifndef COMBINED_GRAPH_GPU_CUH
#define COMBINED_GRAPH_GPU_CUH

/**
 * @file combinedGraphGpu.cuh
 * @brief Steps 2 and 3 of the MOSP update on the GPU: build the combined
 *        graph from the K SOSP trees and find its SOSP tree.
 */

#include "sospUpdateGpu.cuh"

#include <vector>

/// Counters reported by combinedGraphSospGpu().
struct CombineStats {
  int numberOfEdges = 0;   ///< edges of the combined graph
  long long scale = 1;     ///< L: combined distances are in units of 1/L
  long long delta = 0;     ///< near-far bucket width used for Step 3
  SospStats search;        ///< Step 3 statistics
};

/**
 * @brief Device buffers of the combined graph; reserve once per run.
 */
struct CombineWorkspace {
  CombineWorkspace() = default;
  ~CombineWorkspace();
  CombineWorkspace(const CombineWorkspace &) = delete;
  CombineWorkspace &operator=(const CombineWorkspace &) = delete;

  /// Allocate for @p numberOfNodes vertices and @p K trees.
  bool reserve(int numberOfNodes, int K);
  void release();

  int capacity = 0;
  int trees = 0;
  int *rowPtr = nullptr;       ///< n + 1 (children CSR of the combined graph)
  int *cursor = nullptr;       ///< n (fill positions)
  int *colInd = nullptr;       ///< K * n
  int *weights = nullptr;      ///< K * n
  int *preferences = nullptr;  ///< K scaled preference terms L / Pref_i
  unsigned long long *sums = nullptr;      ///< [edge count, weight sum]
  unsigned long long *hostSums = nullptr;  ///< pinned copy of sums
  void *scanStorage = nullptr; ///< CUB scan temporary storage
  size_t scanBytes = 0;
};

/**
 * @brief Build the combined graph of the K trees and run SSSP on it.
 *
 * @details
 * Edge (p, v) belongs to the combined graph iff p is the parent of v in
 * some tree T_i; its weight is L * (K + 1) - sum_{i : Parent_i[v] == p}
 * L / Pref_i (L = lcm(Pref), so the weights stay integral; Pref = all 1s
 * gives K + 1 - m). The edges into v are found by comparing the K parents
 * of v (one thread per vertex), so the graph is built without a map or an
 * edge list sort. Step 3 is a near-far SSSP from @p source on it.
 *
 * @param d_parents   K parent arrays, objective-major: d_parents[k * n + v].
 * @param preferences Pref vector (K values >= 1); empty = all 1s.
 * @param delta       Near-far bucket width; <= 0 selects the default.
 * @param d_distances Output: combined-graph distances (units of 1/L).
 * @param d_parent    Output: parent array of the MOSP tree.
 */
bool combinedGraphSospGpu(const int *d_parents, int numberOfNodes, int K,
                          const std::vector<int> &preferences, int source,
                          long long delta, CombineWorkspace &workspace,
                          SospWorkspace &sospWorkspace, long long *d_distances,
                          int *d_parent, CombineStats *stats = nullptr);

#endif // COMBINED_GRAPH_GPU_CUH
