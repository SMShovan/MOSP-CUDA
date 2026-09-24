/**
 * @file mospUpdate.cu
 * @brief In-memory MOSP update: shared topology, K SOSP updates and the
 *        combined graph on the GPU with buffers allocated once per run.
 */

#include "mospUpdate.cuh"

#include "csrGraph.cuh"
#include "deviceGraph.cuh"
#include "stageTimer.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <string>
#include <vector>

using namespace std;

namespace {

double msSince(chrono::steady_clock::time_point start) {
  return chrono::duration<double, milli>(chrono::steady_clock::now() - start)
      .count();
}

} // namespace

double MospTimings::gpuCompute() const {
  double total = combined;
  for (double t : objectives) {
    total += t;
  }
  return total;
}

void canonicalizeTree(const CsrGraph &graph, int objective, int source,
                      const vector<long long> &distances, int *parent) {
  for (int u = 0; u < graph.numberOfNodes; ++u) {
    const long long du = distances[u];
    if (du >= DISTANCE_INF / 2) {
      continue;
    }
    for (int e = graph.rowPtr[u]; e < graph.rowPtr[u + 1]; ++e) {
      const int v = graph.colInd[e];
      if (v != source && u < parent[v] &&
          du + graph.weight(e, objective) == distances[v]) {
        parent[v] = u;
      }
    }
  }
}

bool mospUpdate(const CsrGraph &original, ChangeBatch &batch,
                const vector<long long> &initialDistances,
                const vector<int> &initialParents, const MospOptions &options,
                CsrGraph &updated, MospResult &result) {
  const int n = original.numberOfNodes;
  const int K = options.numberOfObjectives > 0
                    ? min(options.numberOfObjectives,
                          original.numberOfObjectives)
                    : original.numberOfObjectives;
  const size_t treeSize = static_cast<size_t>(K) * n;
  result = MospResult();
  result.numberOfObjectives = K;
  MospTimings &timings = result.timings;
  if (n <= 0 || K <= 0 || options.source < 0 || options.source >= n ||
      initialDistances.size() < treeSize || initialParents.size() < treeSize) {
    cerr << "Error: invalid MOSP update input.\n";
    return false;
  }
  // The initial trees must be rooted at the source (e.g. `mospPrep init
  // --source` and `mosp --source` must agree).
  for (int k = 0; k < K; ++k) {
    const size_t at = static_cast<size_t>(k) * n + options.source;
    if (initialDistances[at] != 0 || initialParents[at] != -1) {
      cerr << "Error: the initial tree of objective " << k
           << " is not rooted at the source " << options.source
           << " (its distance is ";
      if (initialDistances[at] >= DISTANCE_INF / 2) {
        cerr << "INF";
      } else {
        cerr << initialDistances[at];
      }
      cerr << ", its parent " << initialParents[at]
           << "); compute the trees for this source.\n";
      return false;
    }
  }

  // --- Apply the batch once (host) ------------------------------------------
  auto start = chrono::steady_clock::now();
  {
    ScopedStage stage("apply_batch_host");
    if (!applyChangeBatch(original, batch, updated)) {
      return false;
    }
  }
  timings.applyBatch = msSince(start);

  // Per objective: largest weight before and after the batch (bounds the
  // distances for the packed format) and the average weight (default
  // near-far bucket width).
  const int KG = original.numberOfObjectives;
  const size_t m = static_cast<size_t>(original.numberOfEdges());
  vector<long long> maxWeight(K, 1), weightSum(K, 0);
  for (size_t e = 0; e < m; ++e) {
    for (int k = 0; k < K; ++k) {
      const int w = original.weights[e * KG + k];
      maxWeight[k] = max(maxWeight[k], static_cast<long long>(w));
      weightSum[k] += w;
    }
  }
  for (int i = 0; i < batch.numberOfInserts(); ++i) {
    for (int k = 0; k < K; ++k) {
      maxWeight[k] = max(maxWeight[k], static_cast<long long>(
                                           batch.insertWeights[static_cast<size_t>(i) * KG + k]));
    }
  }

  // Changes that may invalidate a subtree, per objective: every deletion
  // and every insertion that raised the objective's weight of an edge.
  vector<vector<int>> changedFrom(K), changedTo(K);
  for (int k = 0; k < K; ++k) {
    changedFrom[k] = batch.deleteFrom;
    changedTo[k] = batch.deleteTo;
    for (int i = 0; i < batch.numberOfInserts(); ++i) {
      if (batch.weightIncreaseMask[i] & (1u << k)) {
        changedFrom[k].push_back(batch.insertFrom[i]);
        changedTo[k].push_back(batch.insertTo[i]);
      }
    }
  }

  // --- Upload once: graph, trees, batch; allocate all buffers -------------
  start = chrono::steady_clock::now();
  DeviceGraph graph;
  DeviceArray<long long> d_distances, d_combinedDistances(n);
  DeviceArray<int> d_parents, d_combinedParent(n), d_insertHeads;
  vector<DeviceArray<int>> d_changedFrom(K), d_changedTo(K);
  SospWorkspace sospWorkspace;
  CombineWorkspace combineWorkspace;
  {
    ScopedStage stage("upload", true);
    vector<long long> distances(initialDistances.begin(),
                                initialDistances.begin() + treeSize);
    vector<int> parents(initialParents.begin(),
                        initialParents.begin() + treeSize);
    bool ok = uploadDeviceGraph(updated, graph) &&
              d_distances.upload(distances) && d_parents.upload(parents) &&
              d_insertHeads.upload(batch.insertTo) &&
              sospWorkspace.reserve(n) && combineWorkspace.reserve(n, K);
    for (int k = 0; ok && k < K; ++k) {
      ok = d_changedFrom[k].upload(changedFrom[k]) &&
           d_changedTo[k].upload(changedTo[k]);
    }
    if (!ok || cudaDeviceSynchronize() != cudaSuccess) {
      cerr << "Error: could not copy the MOSP inputs to the GPU.\n";
      return false;
    }
  }
  timings.upload = msSince(start);

  // --- Step 1 of MOSP: K SOSP updates ---------------------------------------
  const long long edges = max<long long>(updated.numberOfEdges(), 1);
  result.objectiveStats.resize(K);
  for (int k = 0; k < K; ++k) {
    DeviceChanges changes;
    changes.changedFrom = d_changedFrom[k].data();
    changes.changedTo = d_changedTo[k].data();
    changes.numberOfChanged = static_cast<int>(changedFrom[k].size());
    changes.insertHeads = d_insertHeads.data();
    changes.numberOfInsertHeads = batch.numberOfInserts();
    const long long delta =
        options.delta > 0
            ? options.delta
            : defaultDelta(m > 0 ? static_cast<long long>(m) : edges, n,
                           weightSum[k]);
    start = chrono::steady_clock::now();
    ScopedStage stage("obj" + to_string(k) + "/sosp_update_gpu", true);
    if (!sospUpdateGpu(graph.out(k), graph.in(k), changes, options.source,
                       delta, maxWeight[k], sospWorkspace,
                       d_distances.data() + static_cast<size_t>(k) * n,
                       d_parents.data() + static_cast<size_t>(k) * n,
                       &result.objectiveStats[k])) {
      cerr << "Error: SOSP update of objective " << k << " failed.\n";
      return false;
    }
    stage.stop();
    timings.objectives.push_back(msSince(start));
  }

  // --- Steps 2-3 of MOSP: combined graph and its SOSP tree ------------------
  start = chrono::steady_clock::now();
  {
    ScopedStage stage("combined_graph_gpu", true);
    if (!combinedGraphSospGpu(d_parents.data(), n, K, options.preferences,
                              options.source, 0, combineWorkspace,
                              sospWorkspace, d_combinedDistances.data(),
                              d_combinedParent.data(),
                              &result.combineStats)) {
      cerr << "Error: combined graph step failed.\n";
      return false;
    }
  }
  timings.combined = msSince(start);

  // --- Results back to the host ---------------------------------------------
  start = chrono::steady_clock::now();
  {
    ScopedStage stage("download", true);
    if (!d_distances.download(result.distances) ||
        !d_parents.download(result.parents) ||
        !d_combinedDistances.download(result.combinedDistances) ||
        !d_combinedParent.download(result.combinedParent)) {
      cerr << "Error: could not copy the results from the GPU.\n";
      return false;
    }
  }
  timings.download = msSince(start);
  return true;
}
