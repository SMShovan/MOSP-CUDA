/**
 * @file parallelCombinedGraph.cu
 * @brief Build a combined graph from K SSSP trees and find its SSSP
 *        (Steps 2 and 3 of the MOSP update, thesis Ch. 4).
 *
 * ============================================================================
 * ALGORITHM OVERVIEW
 * ============================================================================
 *
 * Given K SSSP trees (one per objective), produced by K independent runs of
 * parallelSOSPUpdate:
 *
 *   Step 2 — Combined graph
 *     An edge (u → v) is in tree k if parent_k[v] == u, so the in-edges of
 *     v in the combined graph are the distinct values among the K parents
 *     of v. Each edge gets the preference weight (scaled by L = lcm(Pref)):
 *       W'(e) = L * (K + 1) - sum_{i : e in T_i} L / Pref_i
 *     With the default Pref = (1,...,1) this is w = K + 1 - m:
 *       m == K  →  w = 1  (appears in ALL trees — highest preference)
 *       ...
 *       m == 1  →  w = K  (appears in ONE tree — lowest preference)
 *
 *   Step 3 — SOSP on the combined graph, from the source.
 *
 * Both steps run on the GPU (combinedGraphGpu.cu): one thread per vertex
 * compares the K parents, a count/scan/fill pass builds the out-edge CSR
 * and a near-far SSSP finds the tree. The host only reads the K trees and
 * writes the result (no edge map, no temporary files).
 *
 * ============================================================================
 */

#include "parallelCombinedGraph.cuh"

#include "combinedGraphGpu.cuh"
#include "csrGraph.cuh"
#include "deviceArray.cuh"
#include "stageTimer.cuh"

#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

using namespace std;

namespace {

/// Number of vertices of a CSR graph: lines of <prefix>RowPtr.txt minus 1.
int countCsrNodes(const string &prefix) {
  FILE *file = fopen((prefix + "RowPtr.txt").c_str(), "rb");
  if (file == nullptr) {
    return -1;
  }
  static char buffer[1 << 20];
  long long lines = 0;
  size_t got;
  char last = '\n';
  while ((got = fread(buffer, 1, sizeof(buffer), file)) > 0) {
    for (size_t i = 0; i < got; ++i) {
      lines += buffer[i] == '\n' ? 1 : 0;
    }
    last = buffer[got - 1];
  }
  fclose(file);
  if (last != '\n') {
    ++lines; // no newline after the last value
  }
  return static_cast<int>(lines - 1);
}

} // namespace

/**
 * @brief Build a combined graph from K SSSP trees and find its SSSP.
 *
 * @see parallelCombinedGraph.cuh for full parameter documentation.
 */
bool parallelCombinedGraph(const string &originalCsrPrefix,
                           const vector<string> &treeInputPaths, int K,
                           int source, const string & /*workDir*/,
                           const string &distancesOutputPath,
                           const string &treeOutputPath,
                           const vector<int> &preferences) {
  if (K <= 0 || static_cast<int>(treeInputPaths.size()) < K) {
    cout << "Error: K=" << K << " but only " << treeInputPaths.size()
         << " tree paths supplied.\n";
    return false;
  }
  if (K > 32) {
    cout << "Error: at most 32 objectives are supported.\n";
    return false;
  }
  if (preferenceScale(preferences, K) == 0) {
    cout << "Error: invalid preference vector (need K values >= 1).\n";
    return false;
  }

  // --- Read the K parent arrays ---------------------------------------------
  ScopedStage readStage("comb/0_read_trees_text");
  const int numberOfNodes = countCsrNodes(originalCsrPrefix);
  if (numberOfNodes <= 0) {
    cout << "Error: Could not read the vertex count of " << originalCsrPrefix
         << "\n";
    return false;
  }
  if (source < 0 || source >= numberOfNodes) {
    cout << "Error: source vertex " << source << " out of range.\n";
    return false;
  }
  vector<int> parents(static_cast<size_t>(numberOfNodes) * K);
  for (int k = 0; k < K; ++k) {
    vector<int> parent;
    if (!readParents(treeInputPaths[k], numberOfNodes, parent)) {
      return false;
    }
    copy(parent.begin(), parent.end(),
         parents.begin() + static_cast<size_t>(k) * numberOfNodes);
  }
  readStage.stop();

  // --- Steps 2 and 3 on the GPU ---------------------------------------------
  ScopedStage uploadStage("comb/1_alloc_h2d", true);
  DeviceArray<int> d_parents, d_parent(numberOfNodes);
  DeviceArray<long long> d_distances(numberOfNodes);
  CombineWorkspace combineWorkspace;
  SospWorkspace sospWorkspace;
  if (!d_parents.upload(parents) || !combineWorkspace.reserve(numberOfNodes, K) ||
      !sospWorkspace.reserve(numberOfNodes)) {
    cout << "Error: could not allocate the combined graph on the GPU.\n";
    return false;
  }
  uploadStage.stop();

  CombineStats stats;
  {
    ScopedStage gpuStage("comb/steps2_3_gpu", true);
    if (!combinedGraphSospGpu(d_parents.data(), numberOfNodes, K, preferences,
                              source, 0, combineWorkspace, sospWorkspace,
                              d_distances.data(), d_parent.data(), &stats)) {
      cout << "Error: combined graph failed on the GPU.\n";
      return false;
    }
  }
  recordCounter("comb/edges", stats.numberOfEdges);
  recordCounter("comb/iterations", stats.search.iterations);
  recordCounter("comb/pushes", stats.search.pushes);

  // --- Write the result -------------------------------------------------------
  ScopedStage writeStage("comb/2_d2h_write_text");
  vector<long long> distances;
  vector<int> parent;
  if (!d_distances.download(distances) || !d_parent.download(parent) ||
      !writeDistances(distancesOutputPath, distances) ||
      !writeParents(treeOutputPath, parent)) {
    cout << "Error: could not write the combined graph result.\n";
    return false;
  }
  cout << "parallelCombinedGraph: " << stats.numberOfEdges << " edges, "
       << numberOfNodes << " vertices; output written to:\n"
       << "  Distances: " << distancesOutputPath << "\n"
       << "  SSSP Tree: " << treeOutputPath << "\n";
  return true;
}
