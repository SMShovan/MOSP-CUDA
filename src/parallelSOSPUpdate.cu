/**
 * @file parallelSOSPUpdate.cu
 * @brief Parallel (CUDA) Single-Objective Shortest Path (SOSP) Update.
 *
 * Produces the same distances as Dijkstra on the updated graph and, with
 * the lowest-id tie-break, the same SSSP tree.
 *
 * ============================================================================
 * ALGORITHM
 * ============================================================================
 *
 * Phase 0 (host): read the graph, the initial tree and the change batch and
 * apply the batch to forward/reverse adjacency lists.
 *
 * Step 1 (GPU, straight from the change list): the head v of every deleted
 * or weight-increased edge (u,v) with Parent[v] == u is a *root*. Every
 * vertex in the SOSP subtree of a root has lost its shortest path, so the
 * subtrees are invalidated (distance INF, parent -1) by pointer jumping
 * over the parent array (ceil(log2 n) rounds, no host synchronization).
 * The invalidated vertices and the heads of all inserted edges form the
 * first candidate set; each candidate pulls the best (distance, id) over
 * its in-neighbours in the updated graph, i.e. the changes are grouped by
 * destination vertex as in Step 0/1 of the thesis.
 *
 * Step 2 (GPU): the thesis' propagation loop -- collect the out-neighbours
 * of the affected vertices, re-evaluate each candidate over its
 * in-neighbours -- with a *monotone* update: a vertex only takes a strictly
 * better (distance, parent id) pair. Every distance is an upper bound that
 * only decreases (valid vertices keep an intact tree path, invalidated
 * ones start at INF), so the loop terminates without an iteration cap,
 * cannot "count to infinity" through a stale cycle, and vertices cut off
 * from the source stay at INF without a reachability post-pass.
 *
 * Race condition analysis:
 *   - flag arrays: atomicCAS deduplication; lists: atomicAdd compaction;
 *   - d_distances[v]/d_parent[v]: written only by the thread that owns
 *     candidate v (candidates are deduplicated);
 *   - reads of in-neighbour distances: benign race under chaotic
 *     Bellman-Ford semantics -- any change marks the neighbour affected,
 *     so v is re-evaluated in the next iteration;
 *   - pointer jumping: each vertex only reads its ancestors' (ancestor,
 *     flag) pairs and every value it can observe is valid (see
 *     pointerJumpKernel).
 *
 * ============================================================================
 */

#include "parallelSOSPUpdate.cuh"

#include "read.cuh"
#include "stageTimer.cuh"

#include <cuda_runtime.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

using namespace std;

// ============================================================================
// CUDA ERROR CHECKING
// ============================================================================

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = (call);                                                  \
    if (err != cudaSuccess) {                                                  \
      cerr << "CUDA error: " << cudaGetErrorString(err) << " at "             \
           << __FILE__ << ":" << __LINE__ << "\n";                             \
      return false;                                                            \
    }                                                                          \
  } while (0)

// ============================================================================
// CUDA KERNELS
// ============================================================================

/**
 * @brief Collect candidate vertices from affected vertices' out-neighbors.
 *
 * Each thread processes one affected vertex, iterates its out-neighbors
 * in the CSR, and uses atomicCAS for deduplication and atomicAdd for
 * worklist compaction.
 *
 * @param d_affectedList    Array of affected vertex indices.
 * @param numAffected       Number of affected vertices.
 * @param d_outRowPtr       CSR row pointer for forward graph.
 * @param d_outColInd       CSR column indices for forward graph.
 * @param d_isCandidate     Flag array for deduplication (0/1).
 * @param d_candidateList   Output worklist of candidate vertices.
 * @param d_candidateCount  Atomic counter for candidateList size.
 * @param d_isAffected      Flag array for affected vertices (cleared here).
 * @param source            Source vertex (never becomes a candidate).
 */
__global__ void collectCandidatesKernel(
    const int *d_affectedList, int numAffected, const int *d_outRowPtr,
    const int *d_outColInd, int *d_isCandidate, int *d_candidateList,
    int *d_candidateCount, int *d_isAffected, int source) {

  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= numAffected)
    return;

  int affectedVertex = d_affectedList[tid];
  d_isAffected[affectedVertex] = 0; // Clear affected flag

  int rowStart = d_outRowPtr[affectedVertex];
  int rowEnd = d_outRowPtr[affectedVertex + 1];

  for (int e = rowStart; e < rowEnd; ++e) {
    int neighborVertex = d_outColInd[e];

    // CRITICAL: Never update the source vertex
    if (neighborVertex == source)
      continue;

    // Atomic test-and-set for deduplication
    int old = atomicCAS(&d_isCandidate[neighborVertex], 0, 1);
    if (old == 0) {
      int pos = atomicAdd(d_candidateCount, 1);
      d_candidateList[pos] = neighborVertex;
    }
  }
}

/**
 * @brief Re-evaluate candidate vertices over their in-neighbours
 *        (monotone: keep the old value unless strictly better).
 *
 * Each thread processes one candidate vertex and computes the best
 * (distance, parent id) pair over its in-neighbours; ties go to the lowest
 * parent id. The pair replaces the current one only if it is smaller in
 * that order. If the distance decreased, the vertex becomes affected.
 *
 * @param d_candidateList   Array of candidate vertex indices.
 * @param numCandidates     Number of candidate vertices.
 * @param d_inRowPtr        CSR row pointer for reverse graph.
 * @param d_inColInd        CSR column indices for reverse graph.
 * @param d_inWeights       CSR edge weights for reverse graph.
 * @param d_distances       Distance array (read/write).
 * @param d_parent          Parent array (read/write).
 * @param d_isAffected      Flag array for newly affected vertices.
 * @param d_affectedList    Output worklist of newly affected vertices.
 * @param d_affectedCount   Atomic counter for affectedList size.
 * @param INF_VALUE         Sentinel value for unreachable vertices.
 */
__global__ void updateDistancesKernel(
    const int *d_candidateList, int numCandidates, const int *d_inRowPtr,
    const int *d_inColInd, const long long *d_inWeights,
    long long *d_distances, int *d_parent, int *d_isAffected,
    int *d_affectedList, int *d_affectedCount, long long INF_VALUE) {

  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= numCandidates)
    return;

  int candidateVertex = d_candidateList[tid];

  // Start from the current value: the update is monotone.
  const long long currentDistance = d_distances[candidateVertex];
  const int currentParent = d_parent[candidateVertex];
  long long bestDistance = currentDistance;
  int bestParent = currentParent;

  int rowStart = d_inRowPtr[candidateVertex];
  int rowEnd = d_inRowPtr[candidateVertex + 1];

  for (int e = rowStart; e < rowEnd; ++e) {
    int candidateParent = d_inColInd[e];
    long long candidateWeight = d_inWeights[e];

    // Skip unreachable in-neighbors to avoid overflow
    long long parentDist = d_distances[candidateParent];
    if (parentDist >= INF_VALUE / 2)
      continue;

    long long candidateDistance = parentDist + candidateWeight;
    // Ties go to the lowest parent id (canonical SOSP tree).
    if (candidateDistance < bestDistance ||
        (candidateDistance == bestDistance &&
         (bestParent < 0 || candidateParent < bestParent))) {
      bestDistance = candidateDistance;
      bestParent = candidateParent;
    }
  }

  if (bestParent == currentParent && bestDistance == currentDistance)
    return;

  // No race: each candidateVertex is unique in the list
  d_parent[candidateVertex] = bestParent;
  d_distances[candidateVertex] = bestDistance;

  if (bestDistance < currentDistance) {
    // Atomic test-and-set for deduplication in affected list
    int old = atomicCAS(&d_isAffected[candidateVertex], 0, 1);
    if (old == 0) {
      int pos = atomicAdd(d_affectedCount, 1);
      d_affectedList[pos] = candidateVertex;
    }
  }
}

/**
 * @brief Step 1: flag the head of every changed tree edge as a root.
 *
 * Edge (from[i], to[i]) was deleted or its weight increased. If it is the
 * tree edge of its head (parent[to] == from), the head and its SOSP
 * subtree lose their shortest paths.
 */
__global__ void markRootsKernel(const int *d_from, const int *d_to, int count,
                                const int *d_parent, int *d_invalid) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count)
    return;
  int v = d_to[i];
  if (d_parent[v] == d_from[i])
    d_invalid[v] = 1;
}

/** @brief ancestor[v] = parent[v] (start of pointer jumping). */
__global__ void initAncestorsKernel(int numberOfNodes, const int *d_parent,
                                    int *d_ancestor) {
  int v = blockIdx.x * blockDim.x + threadIdx.x;
  if (v < numberOfNodes)
    d_ancestor[v] = d_parent[v];
}

/**
 * @brief One pointer-jumping round of subtree invalidation.
 *
 * Invariant for every vertex x: invalid[x] == 1 implies a root among x and
 * its ancestors; invalid[x] == 0 implies no root on the tree path from x
 * up to (excluding) ancestor[x]. A round either inherits the flag of the
 * current ancestor or jumps to the ancestor's ancestor, which at least
 * doubles the distance covered, so ceil(log2 n) rounds reach the tree
 * root. The rounds update in place; every (ancestor, flag) value a thread
 * can observe satisfies the invariant, so the races are benign.
 */
__global__ void pointerJumpKernel(int numberOfNodes, int *d_ancestor,
                                  int *d_invalid) {
  int v = blockIdx.x * blockDim.x + threadIdx.x;
  if (v >= numberOfNodes)
    return;
  int a = d_ancestor[v];
  if (a < 0 || d_invalid[v])
    return;
  if (d_invalid[a]) {
    d_invalid[v] = 1;
    return;
  }
  d_ancestor[v] = d_ancestor[a];
}

/**
 * @brief Invalidate flagged vertices (distance INF, parent -1) and make
 *        them the first candidates.
 */
__global__ void invalidateKernel(int numberOfNodes, const int *d_invalid,
                                 long long *d_distances, int *d_parent,
                                 int *d_isCandidate, int *d_candidateList,
                                 int *d_candidateCount, long long INF_VALUE) {
  int v = blockIdx.x * blockDim.x + threadIdx.x;
  if (v >= numberOfNodes || !d_invalid[v])
    return;
  d_distances[v] = INF_VALUE;
  d_parent[v] = -1;
  d_isCandidate[v] = 1;
  d_candidateList[atomicAdd(d_candidateCount, 1)] = v;
}

/**
 * @brief Add the heads of inserted edges to the candidate list (deduped).
 */
__global__ void addCandidatesKernel(const int *d_vertices, int count,
                                    int source, int *d_isCandidate,
                                    int *d_candidateList,
                                    int *d_candidateCount) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count)
    return;
  int v = d_vertices[i];
  if (v == source)
    return;
  if (atomicCAS(&d_isCandidate[v], 0, 1) == 0)
    d_candidateList[atomicAdd(d_candidateCount, 1)] = v;
}

// ============================================================================
// HOST HELPER FUNCTIONS
// ============================================================================

namespace {

/// A lightweight edge structure for the internal adjacency lists.
struct WeightedNeighbor {
  int vertex;
  long long weight;
};

/**
 * @brief Parse a line of space-separated integers from a string.
 */
vector<int> parseIntTokens(const string &line) {
  vector<int> tokens;
  istringstream stream(line);
  int value;
  while (stream >> value) {
    tokens.push_back(value);
  }
  return tokens;
}

/**
 * @brief Build forward and reverse adjacency lists from a Graph.
 */
void buildAdjacencyLists(const Graph &graph, int objectiveIndex,
                         vector<vector<WeightedNeighbor>> &outAdjacency,
                         vector<vector<WeightedNeighbor>> &inAdjacency) {
  int numberOfNodes = static_cast<int>(graph.size());
  outAdjacency.assign(numberOfNodes, {});
  inAdjacency.assign(numberOfNodes, {});

  for (int u = 0; u < numberOfNodes; ++u) {
    for (const auto &edge : graph[u]) {
      int v = edge.to;
      long long w = edge.weights[objectiveIndex];
      outAdjacency[u].push_back({v, w});
      inAdjacency[v].push_back({u, w});
    }
  }
}

/**
 * @brief Remove a specific directed edge from an adjacency list entry.
 */
void removeEdgeFromList(vector<WeightedNeighbor> &neighbors, int targetVertex) {
  for (auto it = neighbors.begin(); it != neighbors.end(); ++it) {
    if (it->vertex == targetVertex) {
      neighbors.erase(it);
      return;
    }
  }
}

/**
 * @brief Read distances from a Dijkstra output file.
 */
bool readDistancesFromFile(const string &path, vector<long long> &distances,
                           int numberOfNodes, long long INF_VALUE) {
  ifstream file(path);
  if (!file.is_open()) {
    cout << "Error: Could not open distances file: " << path << "\n";
    return false;
  }

  distances.assign(numberOfNodes, INF_VALUE);

  string line;
  while (getline(file, line)) {
    if (line.empty())
      continue;
    istringstream stream(line);
    int vertexId;
    string distanceStr;
    stream >> vertexId >> distanceStr;

    if (vertexId < 0 || vertexId >= numberOfNodes) {
      cout << "Error: Vertex ID out of range in distances file.\n";
      return false;
    }

    if (distanceStr == "INF") {
      distances[vertexId] = INF_VALUE;
    } else {
      distances[vertexId] = stoll(distanceStr);
    }
  }

  return true;
}

/**
 * @brief Read SSSP tree (parent array) from a Dijkstra output file.
 */
bool readParentFromFile(const string &path, vector<int> &parent,
                        int numberOfNodes) {
  ifstream file(path);
  if (!file.is_open()) {
    cout << "Error: Could not open SSSP tree file: " << path << "\n";
    return false;
  }

  parent.assign(numberOfNodes, -1);

  string line;
  while (getline(file, line)) {
    if (line.empty())
      continue;
    istringstream stream(line);
    int vertexId, parentId;
    stream >> vertexId >> parentId;

    if (vertexId < 0 || vertexId >= numberOfNodes) {
      cout << "Error: Vertex ID out of range in SSSP tree file.\n";
      return false;
    }

    parent[vertexId] = parentId;
  }

  return true;
}

/**
 * @brief Flatten adjacency list to CSR format for device transfer.
 *
 * @param adjacency  Adjacency list (vector of vectors).
 * @param rowPtr     Output CSR row pointer.
 * @param colInd     Output CSR column indices.
 * @param weights    Output CSR edge weights.
 */
void flattenToCSR(const vector<vector<WeightedNeighbor>> &adjacency,
                  vector<int> &rowPtr, vector<int> &colInd,
                  vector<long long> &weights) {
  int n = static_cast<int>(adjacency.size());
  rowPtr.resize(n + 1);
  rowPtr[0] = 0;
  for (int i = 0; i < n; ++i) {
    rowPtr[i + 1] = rowPtr[i] + static_cast<int>(adjacency[i].size());
  }

  int nnz = rowPtr[n];
  colInd.resize(nnz);
  weights.resize(nnz);

  for (int i = 0; i < n; ++i) {
    int offset = rowPtr[i];
    for (int j = 0; j < static_cast<int>(adjacency[i].size()); ++j) {
      colInd[offset + j] = adjacency[i][j].vertex;
      weights[offset + j] = adjacency[i][j].weight;
    }
  }
}

} // namespace

// ============================================================================
// MAIN FUNCTION
// ============================================================================

/**
 * @brief Run the parallel (CUDA) SOSP Update algorithm.
 *
 * @see parallelSOSPUpdate.cuh for full parameter documentation.
 */
bool parallelSOSPUpdate(const string &originalCsrPrefix,
                        const string &distancesInputPath,
                        const string &treeInputPath, const string &insertPath,
                        const string &deletePath, int objectiveIndex,
                        int source, const string &distancesOutputPath,
                        const string &treeOutputPath) {
  const long long INF_VALUE = numeric_limits<long long>::max() / 4;

  // ========================================================================
  // PHASE 0: PREPARATION (Host — I/O dominated)
  // ========================================================================

  // --- 0a. Read original graph from CSR and determine dimensions ---
  ScopedStage readStage("sosp/0a_read_csr_text");
  Graph originalGraph;
  int numberOfObjectives = 0;
  if (!readCSR(originalCsrPrefix, originalGraph, numberOfObjectives)) {
    cout << "Error: Could not read original CSR graph.\n";
    return false;
  }

  int numberOfNodes = static_cast<int>(originalGraph.size());
  if (numberOfNodes == 0) {
    cout << "Error: Graph has no vertices.\n";
    return false;
  }

  if (objectiveIndex < 0 || objectiveIndex >= numberOfObjectives) {
    cout << "Error: objectiveIndex out of range.\n";
    return false;
  }

  if (source < 0 || source >= numberOfNodes) {
    cout << "Error: source vertex out of range.\n";
    return false;
  }

  readStage.stop();

  // --- 0b. Build forward and reverse adjacency lists ---
  ScopedStage adjacencyStage("sosp/0b_build_adjacency_host");
  vector<vector<WeightedNeighbor>> outAdjacency;
  vector<vector<WeightedNeighbor>> inAdjacency;
  buildAdjacencyLists(originalGraph, objectiveIndex, outAdjacency, inAdjacency);

  originalGraph.clear();
  adjacencyStage.stop();

  // --- 0c. Read original distances and parent arrays ---
  ScopedStage treeStage("sosp/0c_read_tree_text");
  vector<long long> distances;
  if (!readDistancesFromFile(distancesInputPath, distances, numberOfNodes,
                             INF_VALUE)) {
    return false;
  }

  vector<int> parent;
  if (!readParentFromFile(treeInputPath, parent, numberOfNodes)) {
    return false;
  }

  treeStage.stop();

  // --- 0d. Read inserted and deleted edges ---
  ScopedStage changesStage("sosp/0d_read_changes_text");
  struct InsertedEdge {
    int from;
    int to;
    long long weight;
  };

  struct DeletedEdge {
    int from;
    int to;
  };

  vector<InsertedEdge> insertedEdges;
  {
    ifstream insertFile(insertPath);
    if (!insertFile.is_open()) {
      cout << "Error: Could not open insert file: " << insertPath << "\n";
      return false;
    }
    string line;
    while (getline(insertFile, line)) {
      if (line.empty())
        continue;
      vector<int> tokens = parseIntTokens(line);
      if (static_cast<int>(tokens.size()) < 2 + numberOfObjectives) {
        cout << "Error: Invalid insert line.\n";
        return false;
      }
      int u = tokens[0];
      int v = tokens[1];
      long long w = tokens[2 + objectiveIndex];
      insertedEdges.push_back({u, v, w});
    }
  }

  vector<DeletedEdge> deletedEdges;
  {
    ifstream deleteFile(deletePath);
    if (!deleteFile.is_open()) {
      cout << "Error: Could not open delete file: " << deletePath << "\n";
      return false;
    }
    string line;
    while (getline(deleteFile, line)) {
      if (line.empty())
        continue;
      vector<int> tokens = parseIntTokens(line);
      if (tokens.size() < 2)
        continue;
      int u = tokens[0];
      int v = tokens[1];
      deletedEdges.push_back({u, v});
    }
  }

  changesStage.stop();

  // --- 0e. Apply topological changes to adjacency lists (Host) ---
  ScopedStage applyStage("sosp/0e_apply_changes_host");
  struct WeightIncrease {
    int from;
    int to;
  };
  vector<WeightIncrease> weightIncreases;

  // Deletions first
  for (const auto &edge : deletedEdges) {
    removeEdgeFromList(outAdjacency[edge.from], edge.to);
    removeEdgeFromList(inAdjacency[edge.to], edge.from);
  }

  // Then insertions: REPLACE if edge already exists, otherwise add.
  for (const auto &edge : insertedEdges) {
    bool replacedOut = false;
    for (auto &neighbor : outAdjacency[edge.from]) {
      if (neighbor.vertex == edge.to) {
        if (edge.weight > neighbor.weight) {
          weightIncreases.push_back({edge.from, edge.to});
        }
        neighbor.weight = edge.weight;
        replacedOut = true;
        break;
      }
    }
    if (!replacedOut) {
      outAdjacency[edge.from].push_back({edge.to, edge.weight});
    }

    bool replacedIn = false;
    for (auto &neighbor : inAdjacency[edge.to]) {
      if (neighbor.vertex == edge.from) {
        neighbor.weight = edge.weight;
        replacedIn = true;
        break;
      }
    }
    if (!replacedIn) {
      inAdjacency[edge.to].push_back({edge.from, edge.weight});
    }
  }

  applyStage.stop();

  // Heads of inserted edges, and edges that may invalidate a subtree:
  // deletions and weight increases (as (from, to) pairs).
  vector<int> insertHeads, changedFrom, changedTo;
  for (const auto &edge : insertedEdges) {
    insertHeads.push_back(edge.to);
  }
  for (const auto &edge : deletedEdges) {
    changedFrom.push_back(edge.from);
    changedTo.push_back(edge.to);
  }
  for (const auto &wi : weightIncreases) {
    changedFrom.push_back(wi.from);
    changedTo.push_back(wi.to);
  }

  // --- Flatten adjacency lists to CSR for device transfer ---
  ScopedStage flattenStage("sosp/2a_flatten_csr_host");
  vector<int> h_outRowPtr, h_outColInd;
  vector<long long> h_outWeights;
  flattenToCSR(outAdjacency, h_outRowPtr, h_outColInd, h_outWeights);

  vector<int> h_inRowPtr, h_inColInd;
  vector<long long> h_inWeights;
  flattenToCSR(inAdjacency, h_inRowPtr, h_inColInd, h_inWeights);

  // Free host adjacency lists (no longer needed)
  outAdjacency.clear();
  inAdjacency.clear();

  int outNnz = static_cast<int>(h_outColInd.size());
  int inNnz = static_cast<int>(h_inColInd.size());
  const int numChanged = static_cast<int>(changedFrom.size());
  const int numInsertHeads = static_cast<int>(insertHeads.size());
  flattenStage.stop();

  // --- Allocate device memory ---
  ScopedStage uploadStage("sosp/2b_alloc_h2d", true);
  int *d_outRowPtr = nullptr, *d_outColInd = nullptr;
  int *d_inRowPtr = nullptr, *d_inColInd = nullptr;
  long long *d_inWeights = nullptr;
  long long *d_distances = nullptr;
  int *d_parent = nullptr;
  int *d_isAffected = nullptr, *d_isCandidate = nullptr;
  int *d_affectedList = nullptr, *d_candidateList = nullptr;
  int *d_affectedCount = nullptr, *d_candidateCount = nullptr;
  int *d_invalid = nullptr, *d_ancestor = nullptr;
  int *d_changedFrom = nullptr, *d_changedTo = nullptr;
  int *d_insertHeads = nullptr;

  CUDA_CHECK(cudaMalloc(&d_outRowPtr, (numberOfNodes + 1) * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_outColInd, max(outNnz, 1) * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_inRowPtr, (numberOfNodes + 1) * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_inColInd, max(inNnz, 1) * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_inWeights, max(inNnz, 1) * sizeof(long long)));
  CUDA_CHECK(cudaMalloc(&d_distances, numberOfNodes * sizeof(long long)));
  CUDA_CHECK(cudaMalloc(&d_parent, numberOfNodes * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_isAffected, numberOfNodes * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_isCandidate, numberOfNodes * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_affectedList, numberOfNodes * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_candidateList, numberOfNodes * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_affectedCount, sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_candidateCount, sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_invalid, numberOfNodes * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_ancestor, numberOfNodes * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_changedFrom, max(numChanged, 1) * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_changedTo, max(numChanged, 1) * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_insertHeads, max(numInsertHeads, 1) * sizeof(int)));

  // --- Copy data to device ---
  CUDA_CHECK(cudaMemcpy(d_outRowPtr, h_outRowPtr.data(),
                        (numberOfNodes + 1) * sizeof(int),
                        cudaMemcpyHostToDevice));
  if (outNnz > 0) {
    CUDA_CHECK(cudaMemcpy(d_outColInd, h_outColInd.data(),
                          outNnz * sizeof(int), cudaMemcpyHostToDevice));
  }
  CUDA_CHECK(cudaMemcpy(d_inRowPtr, h_inRowPtr.data(),
                        (numberOfNodes + 1) * sizeof(int),
                        cudaMemcpyHostToDevice));
  if (inNnz > 0) {
    CUDA_CHECK(cudaMemcpy(d_inColInd, h_inColInd.data(),
                          inNnz * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_inWeights, h_inWeights.data(),
                          inNnz * sizeof(long long), cudaMemcpyHostToDevice));
  }
  CUDA_CHECK(cudaMemcpy(d_distances, distances.data(),
                        numberOfNodes * sizeof(long long),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_parent, parent.data(), numberOfNodes * sizeof(int),
                        cudaMemcpyHostToDevice));
  if (numChanged > 0) {
    CUDA_CHECK(cudaMemcpy(d_changedFrom, changedFrom.data(),
                          numChanged * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_changedTo, changedTo.data(),
                          numChanged * sizeof(int), cudaMemcpyHostToDevice));
  }
  if (numInsertHeads > 0) {
    CUDA_CHECK(cudaMemcpy(d_insertHeads, insertHeads.data(),
                          numInsertHeads * sizeof(int),
                          cudaMemcpyHostToDevice));
  }
  uploadStage.stop();

  const int BLOCK_SIZE = 256;
  auto blocks = [&](int work) {
    return (max(work, 1) + BLOCK_SIZE - 1) / BLOCK_SIZE;
  };

  // ========================================================================
  // STEP 1: ROOTS, SUBTREE INVALIDATION AND FIRST CANDIDATES (GPU)
  // ========================================================================
  ScopedStage step1Stage("sosp/1_invalidate_gpu", true);
  CUDA_CHECK(cudaMemset(d_invalid, 0, numberOfNodes * sizeof(int)));
  CUDA_CHECK(cudaMemset(d_isAffected, 0, numberOfNodes * sizeof(int)));
  CUDA_CHECK(cudaMemset(d_isCandidate, 0, numberOfNodes * sizeof(int)));
  CUDA_CHECK(cudaMemset(d_candidateCount, 0, sizeof(int)));
  if (numChanged > 0) {
    markRootsKernel<<<blocks(numChanged), BLOCK_SIZE>>>(
        d_changedFrom, d_changedTo, numChanged, d_parent, d_invalid);
    CUDA_CHECK(cudaGetLastError());

    // ceil(log2 n) rounds: the jump distance doubles every round.
    int rounds = 0;
    while ((1LL << rounds) < numberOfNodes) {
      ++rounds;
    }
    initAncestorsKernel<<<blocks(numberOfNodes), BLOCK_SIZE>>>(
        numberOfNodes, d_parent, d_ancestor);
    for (int r = 0; r < rounds; ++r) {
      pointerJumpKernel<<<blocks(numberOfNodes), BLOCK_SIZE>>>(
          numberOfNodes, d_ancestor, d_invalid);
    }
    CUDA_CHECK(cudaGetLastError());
    invalidateKernel<<<blocks(numberOfNodes), BLOCK_SIZE>>>(
        numberOfNodes, d_invalid, d_distances, d_parent, d_isCandidate,
        d_candidateList, d_candidateCount, INF_VALUE);
    CUDA_CHECK(cudaGetLastError());
  }
  int h_invalidated = 0;
  CUDA_CHECK(cudaMemcpy(&h_invalidated, d_candidateCount, sizeof(int),
                        cudaMemcpyDeviceToHost));
  recordCounter("sosp/invalidated", h_invalidated);
  if (numInsertHeads > 0) {
    addCandidatesKernel<<<blocks(numInsertHeads), BLOCK_SIZE>>>(
        d_insertHeads, numInsertHeads, source, d_isCandidate, d_candidateList,
        d_candidateCount);
    CUDA_CHECK(cudaGetLastError());
  }
  int h_candidateCount = 0;
  CUDA_CHECK(cudaMemcpy(&h_candidateCount, d_candidateCount, sizeof(int),
                        cudaMemcpyDeviceToHost));

  // First pull pass: every candidate takes its best valid in-neighbour.
  int h_affectedCount = 0;
  CUDA_CHECK(cudaMemset(d_affectedCount, 0, sizeof(int)));
  if (h_candidateCount > 0) {
    updateDistancesKernel<<<blocks(h_candidateCount), BLOCK_SIZE>>>(
        d_candidateList, h_candidateCount, d_inRowPtr, d_inColInd, d_inWeights,
        d_distances, d_parent, d_isAffected, d_affectedList, d_affectedCount,
        INF_VALUE);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(&h_affectedCount, d_affectedCount, sizeof(int),
                          cudaMemcpyDeviceToHost));
  }
  step1Stage.stop();
  recordCounter("sosp/initial_affected", h_affectedCount);

  // ========================================================================
  // STEP 2: PROPAGATE THE UPDATE (CUDA Kernels, monotone)
  // ========================================================================
  ScopedStage propagateStage("sosp/2c_propagate_gpu", true);
  long long totalCandidates = 0;
  int iterationCount = 0;

  while (h_affectedCount > 0) {
    ++iterationCount;
    if (iterationCount > numberOfNodes) {
      // Distances only decrease and every sweep settles at least one more
      // hop of every shortest path, so this cannot happen.
      cout << "Error: SOSP update did not converge.\n";
      return false;
    }

    // Reset candidate structures
    CUDA_CHECK(
        cudaMemset(d_isCandidate, 0, numberOfNodes * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_candidateCount, 0, sizeof(int)));

    // --- 2a. Collect candidates ---
    collectCandidatesKernel<<<blocks(h_affectedCount), BLOCK_SIZE>>>(
        d_affectedList, h_affectedCount, d_outRowPtr, d_outColInd,
        d_isCandidate, d_candidateList, d_candidateCount, d_isAffected,
        source);
    CUDA_CHECK(cudaGetLastError());

    // Get candidate count
    CUDA_CHECK(cudaMemcpy(&h_candidateCount, d_candidateCount, sizeof(int),
                          cudaMemcpyDeviceToHost));
    totalCandidates += h_candidateCount;
    if (h_candidateCount == 0)
      break;

    // Reset affected structures for next iteration
    CUDA_CHECK(cudaMemset(d_affectedCount, 0, sizeof(int)));

    // --- 2b. Update distances ---
    updateDistancesKernel<<<blocks(h_candidateCount), BLOCK_SIZE>>>(
        d_candidateList, h_candidateCount, d_inRowPtr, d_inColInd, d_inWeights,
        d_distances, d_parent, d_isAffected, d_affectedList, d_affectedCount,
        INF_VALUE);
    CUDA_CHECK(cudaGetLastError());

    // Get affected count for next iteration
    CUDA_CHECK(cudaMemcpy(&h_affectedCount, d_affectedCount, sizeof(int),
                          cudaMemcpyDeviceToHost));
  }
  propagateStage.stop();
  recordCounter("sosp/iterations", iterationCount);
  recordCounter("sosp/candidates", totalCandidates);

  // ========================================================================
  // COPY RESULTS BACK TO HOST
  // ========================================================================
  ScopedStage downloadStage("sosp/4_d2h_free", true);

  CUDA_CHECK(cudaMemcpy(distances.data(), d_distances,
                        numberOfNodes * sizeof(long long),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(parent.data(), d_parent, numberOfNodes * sizeof(int),
                        cudaMemcpyDeviceToHost));

  // --- Free device memory ---
  CUDA_CHECK(cudaFree(d_outRowPtr));
  CUDA_CHECK(cudaFree(d_outColInd));
  CUDA_CHECK(cudaFree(d_inRowPtr));
  CUDA_CHECK(cudaFree(d_inColInd));
  CUDA_CHECK(cudaFree(d_inWeights));
  CUDA_CHECK(cudaFree(d_distances));
  CUDA_CHECK(cudaFree(d_parent));
  CUDA_CHECK(cudaFree(d_isAffected));
  CUDA_CHECK(cudaFree(d_isCandidate));
  CUDA_CHECK(cudaFree(d_affectedList));
  CUDA_CHECK(cudaFree(d_candidateList));
  CUDA_CHECK(cudaFree(d_affectedCount));
  CUDA_CHECK(cudaFree(d_candidateCount));
  CUDA_CHECK(cudaFree(d_invalid));
  CUDA_CHECK(cudaFree(d_ancestor));
  CUDA_CHECK(cudaFree(d_changedFrom));
  CUDA_CHECK(cudaFree(d_changedTo));
  CUDA_CHECK(cudaFree(d_insertHeads));

  downloadStage.stop();

  // ========================================================================
  // WRITE OUTPUT (Host — I/O)
  // ========================================================================
  ScopedStage writeStage("sosp/5_write_text");

  filesystem::path distOutPath(distancesOutputPath);
  if (!distOutPath.parent_path().empty()) {
    filesystem::create_directories(distOutPath.parent_path());
  }

  filesystem::path treeOutPath(treeOutputPath);
  if (!treeOutPath.parent_path().empty()) {
    filesystem::create_directories(treeOutPath.parent_path());
  }

  ofstream distancesOut(distancesOutputPath);
  if (!distancesOut.is_open()) {
    cout << "Error: Could not write updated distances file.\n";
    return false;
  }

  for (int i = 0; i < numberOfNodes; ++i) {
    distancesOut << i << " ";
    if (distances[i] >= INF_VALUE / 2) {
      distancesOut << "INF";
    } else {
      distancesOut << distances[i];
    }
    distancesOut << "\n";
  }

  ofstream treeOut(treeOutputPath);
  if (!treeOut.is_open()) {
    cout << "Error: Could not write updated SSSP tree file.\n";
    return false;
  }

  for (int i = 0; i < numberOfNodes; ++i) {
    treeOut << i << " " << parent[i] << "\n";
  }

  return true;
}
