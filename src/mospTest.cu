/**
 * @file mospTest.cu
 * @brief Oracle tests for the SOSP update and the combined-graph step.
 *
 * Every case builds a seeded graph, initial SOSP trees (Dijkstra) and a
 * change batch, runs the update through the public file-based API and
 * checks the result against Dijkstra on the updated graph:
 *   - distances equal (unreachable vertices = INF),
 *   - parent consistency: an existing edge (p,v) with d[p]+w(p,v) == d[v],
 *   - canonical parents: the lowest id among equal-distance parents, so
 *     every tree equals the (canonical) Dijkstra tree exactly,
 *   - determinism: running the parallel update twice gives identical files.
 *
 * Change sets: uniform (connectivity-safe and unsafe), deletions only,
 * disconnecting deletions, insertions only, tree-edge weight increases,
 * re-weighting, targeted (thesis) and local batches. Graphs: random
 * directed graphs (the repository's generator) and road-like grids.
 *
 * Usage: mospTest [--seed S] [--work DIR] [--only GROUP]
 *   GROUP: thesis-example, regressions, large-weights, packing-boundary,
 *          input-validation, binary-cache, generator, apply, sosp
 *          (default: all). Exit code 0 = all checks passed.
 */

#include "changeGenerator.cuh"
#include "csrGraph.cuh"
#include "deviceGraph.cuh"
#include "dijkstra.cuh"
#include "generateChangedEdges.cuh"
#include "generateGraphCSR.cuh"
#include "mospUpdate.cuh"
#include "parallelCombinedGraph.cuh"
#include "parallelSOSPUpdate.cuh"
#include "sequentialSOSPUpdate.cuh"
#include "sospUpdateGpu.cuh"
#include "updateGraphCSR.cuh"
#include "validation.cuh"

#include <algorithm>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

using namespace std;

namespace {

string g_work = "mospTest-work";
int g_failures = 0;

// ============================================================================
// Graph builders
// ============================================================================

/// Road-like graph: a width x height grid, both directions, K weights in
/// [minWeight, maxWeight]; a fraction of the grid edges is dropped (both
/// ways) so the graph has dead ends and long detours.
CsrGraph gridGraph(int width, int height, int K, int maxWeight,
                   double dropFraction, unsigned int seed, int minWeight = 1) {
  mt19937 rng(seed);
  uniform_int_distribution<int> weight(minWeight, maxWeight);
  uniform_real_distribution<double> coin(0.0, 1.0);
  const int n = width * height;
  vector<vector<pair<int, vector<int>>>> rows(n);
  auto id = [&](int x, int y) { return y * width + x; };
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      for (auto [dx, dy] : {pair<int, int>{1, 0}, pair<int, int>{0, 1}}) {
        int nx = x + dx, ny = y + dy;
        if (nx >= width || ny >= height || coin(rng) < dropFraction) {
          continue;
        }
        int u = id(x, y), v = id(nx, ny);
        vector<int> w1(K), w2(K);
        for (int k = 0; k < K; ++k) {
          w1[k] = weight(rng);
          w2[k] = weight(rng);
        }
        rows[u].push_back({v, w1});
        rows[v].push_back({u, w2});
      }
    }
  }
  CsrGraph graph;
  graph.numberOfNodes = n;
  graph.numberOfObjectives = K;
  graph.rowPtr.assign(n + 1, 0);
  for (int u = 0; u < n; ++u) {
    graph.rowPtr[u + 1] = graph.rowPtr[u] + static_cast<int>(rows[u].size());
    for (auto &edge : rows[u]) {
      graph.colInd.push_back(edge.first);
      graph.weights.insert(graph.weights.end(), edge.second.begin(),
                           edge.second.end());
    }
  }
  return graph;
}

/// Random directed graph from the repository's generator (spanning chain
/// 0 -> 1 -> ... -> n-1 plus random edges).
CsrGraph randomGraph(const string &dir, int n, int m, int K, int maxWeight,
                     unsigned int seed) {
  CsrGraph graph;
  generateGraphCSR(n, m, true, dir + "/random/graphCsr", K, 1, maxWeight, seed);
  readCsrGraph(dir + "/random/graphCsr", graph);
  return graph;
}

// ============================================================================
// Checks
// ============================================================================

void report(const string &name, bool ok, const string &detail) {
  if (!ok) {
    ++g_failures;
    cout << "  FAIL " << name << ": " << detail << "\n";
  }
}

struct CaseFiles {
  string dir, graph, insert, remove, init;
};

/// Write graph, change batch and initial trees for one case.
CaseFiles writeCase(const string &dir, const CsrGraph &graph,
                    const ChangeBatch &batch, int source) {
  CaseFiles files{dir, dir + "/graph/graphCsr", dir + "/changes/insert.txt",
                  dir + "/changes/delete.txt", dir + "/init"};
  writeCsrGraph(files.graph, graph);
  writeChangeBatch(batch, files.insert, files.remove);
  for (int k = 0; k < graph.numberOfObjectives; ++k) {
    vector<long long> dist;
    vector<int> parent;
    dijkstraCsrGraph(graph, k, source, dist, parent);
    string obj = files.init + "/obj" + to_string(k);
    writeDistances(obj + "/distances.txt", dist);
    writeParents(obj + "/tree.txt", parent);
  }
  return files;
}

/// Output directory of one implementation ("label") for objective k.
string outputDir(const CaseFiles &files, const string &label, int k) {
  return files.dir + "/" + label + "/obj" + to_string(k);
}

using UpdateFunction = function<bool(const CaseFiles &, int objective,
                                     int source, const string &distOut,
                                     const string &treeOut)>;

/// Run one update implementation for every objective and validate.
/// Returns the updated trees (parents) for the combined-graph check.
bool runAndCheck(const string &label, const UpdateFunction &update,
                 const CaseFiles &files, const CsrGraph &updated,
                 const CsrGraph &reverse, int source,
                 vector<string> *treePaths = nullptr) {
  bool allOk = true;
  for (int k = 0; k < updated.numberOfObjectives; ++k) {
    string out = outputDir(files, label, k);
    string distOut = out + "/distances.txt", treeOut = out + "/tree.txt";
    if (!update(files, k, source, distOut, treeOut)) {
      report(label, false, files.dir + " obj" + to_string(k) + ": call failed");
      allOk = false;
      continue;
    }
    vector<long long> dist, refDist;
    vector<int> parent, refParent;
    readDistances(distOut, updated.numberOfNodes, dist);
    readParents(treeOut, updated.numberOfNodes, parent);
    dijkstraCsrGraph(updated, k, source, refDist, refParent);
    TreeCheck check =
        checkSospTree(reverse, k, source, dist, parent, refDist, &refParent);
    bool ok = check.ok(true);
    report(label, ok, files.dir + " obj" + to_string(k) + ": " + check.summary());
    allOk = allOk && ok;
    if (treePaths != nullptr) {
      treePaths->push_back(treeOut);
    }
  }
  return allOk;
}

void checkCombined(const CaseFiles &files, const vector<string> &trees,
                   int n, int source, const vector<int> &pref) {
  const int K = static_cast<int>(trees.size());
  string dir = files.dir + "/combined" + to_string(pref.empty() ? 0 : pref[0]);
  if (!parallelCombinedGraph(files.graph, trees, K, source, dir,
                             dir + "/distances.txt", dir + "/tree.txt",
                             pref)) {
    report("combined", false, files.dir + ": call failed");
    return;
  }
  vector<vector<int>> parents(K);
  for (int k = 0; k < K; ++k) {
    readParents(trees[k], n, parents[k]);
  }
  vector<int> flat;
  for (const auto &tree : parents) {
    flat.insert(flat.end(), tree.begin(), tree.end());
  }
  CsrGraph combined = combinedGraphReference(flat, n, K, source, pref), reverse;
  transposeCsrGraph(combined, reverse);
  vector<long long> dist, refDist;
  vector<int> parent, refParent;
  readDistances(dir + "/distances.txt", n, dist);
  readParents(dir + "/tree.txt", n, parent);
  dijkstraCsrGraph(combined, 0, source, refDist, refParent);
  TreeCheck check =
      checkSospTree(reverse, 0, source, dist, parent, refDist, &refParent);
  report("combined", check.ok(true), files.dir + ": " + check.summary());
}

// ============================================================================
// Test sets
// ============================================================================

/// Run the parallel update a second time; outputs must be identical.
void checkDeterminism(const CaseFiles &files, const string &firstLabel, int n,
                      int K, int source) {
  for (int k = 0; k < K; ++k) {
    string obj = files.init + "/obj" + to_string(k);
    string first = outputDir(files, firstLabel, k);
    string again = outputDir(files, "rerun", k);
    parallelSOSPUpdate(files.graph, obj + "/distances.txt", obj + "/tree.txt",
                       files.insert, files.remove, k, source,
                       again + "/distances.txt", again + "/tree.txt");
    vector<int> firstTree, againTree;
    vector<long long> firstDist, againDist;
    readParents(first + "/tree.txt", n, firstTree);
    readParents(again + "/tree.txt", n, againTree);
    readDistances(first + "/distances.txt", n, firstDist);
    readDistances(again + "/distances.txt", n, againDist);
    report("determinism", firstTree == againTree && firstDist == againDist,
           files.dir + " obj" + to_string(k) + ": two runs differ");
  }
}

/// The in-memory pipeline (mospUpdate): every tree and the MOSP tree must
/// equal the host references exactly.
void checkPipeline(const string &dir, const CsrGraph &graph,
                   const ChangeBatch &input, int source,
                   const vector<int> &pref) {
  const int n = graph.numberOfNodes, K = graph.numberOfObjectives;
  vector<long long> distances;
  vector<int> parents;
  for (int k = 0; k < K; ++k) {
    vector<long long> dist;
    vector<int> parent;
    dijkstraCsrGraph(graph, k, source, dist, parent);
    distances.insert(distances.end(), dist.begin(), dist.end());
    parents.insert(parents.end(), parent.begin(), parent.end());
  }
  ChangeBatch batch = input;
  MospOptions options;
  options.source = source;
  options.preferences = pref;
  CsrGraph updated, reverse;
  MospResult result;
  if (!mospUpdate(graph, batch, distances, parents, options, updated,
                  result)) {
    report("pipeline", false, dir + ": mospUpdate failed");
    return;
  }
  transposeCsrGraph(updated, reverse);
  for (int k = 0; k < K; ++k) {
    vector<long long> refDist;
    vector<int> refParent;
    dijkstraCsrGraph(updated, k, source, refDist, refParent);
    vector<long long> dist(result.distances.begin() + static_cast<size_t>(k) * n,
                           result.distances.begin() + static_cast<size_t>(k + 1) * n);
    vector<int> parent(result.parents.begin() + static_cast<size_t>(k) * n,
                       result.parents.begin() + static_cast<size_t>(k + 1) * n);
    TreeCheck check =
        checkSospTree(reverse, k, source, dist, parent, refDist, &refParent);
    report("pipeline", check.ok(true),
           dir + " obj" + to_string(k) + ": " + check.summary());
  }
  CsrGraph combined =
      combinedGraphReference(result.parents, n, K, source, pref);
  CsrGraph combinedReverse;
  transposeCsrGraph(combined, combinedReverse);
  vector<long long> refDist;
  vector<int> refParent;
  dijkstraCsrGraph(combined, 0, source, refDist, refParent);
  TreeCheck check =
      checkSospTree(combinedReverse, 0, source, result.combinedDistances,
                    result.combinedParent, refDist, &refParent);
  report("pipeline", check.ok(true), dir + " combined: " + check.summary());
}

struct ChangeSet {
  string name;
  ChangeGeneratorOptions options;
  /// Optional custom batch (overrides the generator).
  function<void(const CsrGraph &, unsigned int, ChangeBatch &)> custom;
};

/// Delete every in-edge of a few random vertices: they (and whatever is
/// only reachable through them) become unreachable.
void disconnectingBatch(const CsrGraph &graph, unsigned int seed,
                        ChangeBatch &batch) {
  batch = ChangeBatch();
  batch.numberOfObjectives = graph.numberOfObjectives;
  CsrGraph reverse;
  transposeCsrGraph(graph, reverse);
  mt19937 rng(seed);
  uniform_int_distribution<int> pick(1, graph.numberOfNodes - 1);
  const int victims = max(1, graph.numberOfNodes / 50);
  for (int i = 0; i < victims; ++i) {
    int v = pick(rng);
    for (int e = reverse.rowPtr[v]; e < reverse.rowPtr[v + 1]; ++e) {
      batch.deleteFrom.push_back(reverse.colInd[e]);
      batch.deleteTo.push_back(v);
    }
  }
}

vector<ChangeSet> changeSets(int numberOfChanges) {
  auto make = [&](const string &name, ChangeMode mode, double ins, int local,
                  bool safe) {
    ChangeSet set;
    set.name = name;
    set.options.numberOfChanges = numberOfChanges;
    set.options.mode = mode;
    set.options.insertionPercentage = ins;
    set.options.localHops = local;
    set.options.safeDeletions = safe;
    set.options.weightMax = 50;
    return set;
  };
  vector<ChangeSet> sets = {
      make("uniform-safe", ChangeMode::Uniform, 50, 0, true),
      make("uniform-unsafe", ChangeMode::Uniform, 50, 0, false),
      make("deletions-only", ChangeMode::Uniform, 0, 0, false),
      make("insertions-only", ChangeMode::Uniform, 100, 0, false),
      make("tree-weight-increases", ChangeMode::Increase, 0, 0, false),
      make("reweight", ChangeMode::Reweight, 0, 0, false),
      make("targeted", ChangeMode::Targeted, 50, 0, false),
      make("local", ChangeMode::Uniform, 50, 4, false),
      make("local-reweight", ChangeMode::Reweight, 0, 4, false),
  };
  ChangeSet cut;
  cut.name = "disconnecting";
  cut.custom = disconnectingBatch;
  sets.push_back(cut);
  return sets;
}

void runSosp(unsigned int seed) {
  struct GraphSpec {
    string name;
    function<CsrGraph(const string &, unsigned int)> build;
    int changes;
  };
  vector<GraphSpec> graphs = {
      {"random-tiny",
       [](const string &d, unsigned int s) { return randomGraph(d, 12, 30, 3, 50, s); },
       6},
      {"random-small",
       [](const string &d, unsigned int s) { return randomGraph(d, 60, 240, 2, 50, s); },
       20},
      {"random-medium",
       [](const string &d, unsigned int s) { return randomGraph(d, 800, 3200, 3, 100, s); },
       200},
      {"grid-small",
       [](const string &, unsigned int s) { return gridGraph(8, 8, 2, 50, 0.1, s); },
       12},
      {"grid-medium",
       [](const string &, unsigned int s) { return gridGraph(40, 30, 3, 100, 0.15, s); },
       150},
  };
  const int source = 0;
  int cases = 0;
  for (const auto &spec : graphs) {
    for (int rep = 0; rep < 3; ++rep) {
      unsigned int caseSeed = seed * 1000u + static_cast<unsigned>(rep) * 97u +
                              static_cast<unsigned>(spec.name.size());
      string base = g_work + "/" + spec.name + "_" + to_string(rep);
      CsrGraph graph = spec.build(base, caseSeed);
      for (const auto &set : changeSets(spec.changes)) {
        ChangeBatch batch;
        if (set.custom) {
          set.custom(graph, caseSeed, batch);
        } else {
          ChangeGeneratorOptions options = set.options;
          options.seed = caseSeed;
          options.source = source;
          if (!generateChangeBatch(graph, options, batch)) {
            continue; // e.g. too few edges in a local region
          }
        }
        string dir = base + "/" + set.name;
        CaseFiles files = writeCase(dir, graph, batch, source);
        CsrGraph updated, reverse;
        ChangeBatch copy = batch;
        applyChangeBatch(graph, copy, updated);
        transposeCsrGraph(updated, reverse);

        auto parallel = [](const CaseFiles &f, int k, int s, const string &d,
                           const string &t) {
          string obj = f.init + "/obj" + to_string(k);
          return parallelSOSPUpdate(f.graph, obj + "/distances.txt",
                                    obj + "/tree.txt", f.insert, f.remove, k, s,
                                    d, t);
        };
        auto sequential = [](const CaseFiles &f, int k, int s, const string &d,
                             const string &t) {
          string obj = f.init + "/obj" + to_string(k);
          return sequentialSOSPUpdate(f.graph, obj + "/distances.txt",
                                      obj + "/tree.txt", f.insert, f.remove, k,
                                      s, d, t);
        };
        vector<string> trees;
        runAndCheck("parallel/" + set.name, parallel, files, updated, reverse,
                    source, &trees);
        runAndCheck("sequential/" + set.name, sequential, files, updated,
                    reverse, source);
        checkDeterminism(files, "parallel/" + set.name, graph.numberOfNodes,
                         graph.numberOfObjectives, source);
        vector<int> skewedPref(graph.numberOfObjectives, 2);
        skewedPref[0] = 1;
        checkPipeline(dir, graph, batch, source, rep == 0 ? vector<int>() : skewedPref);
        if (static_cast<int>(trees.size()) == graph.numberOfObjectives) {
          // Default Pref (all 1s) and a skewed Pref = (K+1, 1, K+1, ...).
          checkCombined(files, trees, graph.numberOfNodes, source, {});
          vector<int> skewed(graph.numberOfObjectives, graph.numberOfObjectives + 1);
          skewed[graph.numberOfObjectives > 1 ? 1 : 0] = 1;
          checkCombined(files, trees, graph.numberOfNodes, source, skewed);
        }
        ++cases;
      }
    }
  }
  cout << "sosp: " << cases << " cases (parallel, sequential, combined, pipeline)\n";
}

/// The worked example of thesis Ch. 4 (Fig. "Finding a single MOSP"):
/// three SOSP updates, then the combined graph with Pref = {4,1,4} and
/// {4,4,1}. Vertices u1..u7 are 0..6, source u1.
///
/// The edge u3->u6 (4,2,8) drawn in the preliminaries figure is omitted: with
/// it, the objective-1 tree of the example (u6 reached through u5 at 11)
/// would not be a shortest-path tree (u1->u3->u6 costs 8). Without it the
/// three updated trees are exactly those of sub-figures (a)-(c).
void runThesisExample(unsigned int) {
  const int n = 7, K = 3, source = 0;
  struct E { int u, v, w1, w2, w3; };
  vector<E> edges = {{0, 1, 2, 1, 5},   {0, 2, 4, 1, 1},  {2, 1, 10, 15, 2},
                     {1, 3, 2, 4, 2},   {2, 3, 5, 16, 3}, {3, 4, 1, 1, 1},
                     {4, 1, 4, 3, 2},   {4, 5, 1, 2, 2},  {4, 6, 5, 6, 2},
                     {5, 6, 1, 1, 1}};
  stable_sort(edges.begin(), edges.end(),
              [](const E &a, const E &b) { return a.u < b.u; });
  CsrGraph graph;
  graph.numberOfNodes = n;
  graph.numberOfObjectives = K;
  graph.rowPtr.assign(n + 1, 0);
  for (const auto &e : edges) {
    ++graph.rowPtr[e.u + 1];
  }
  for (int u = 0; u < n; ++u) {
    graph.rowPtr[u + 1] += graph.rowPtr[u];
  }
  for (const auto &e : edges) { // now in row order
    graph.colInd.push_back(e.v);
    graph.weights.insert(graph.weights.end(), {e.w1, e.w2, e.w3});
  }
  ChangeBatch batch;
  batch.numberOfObjectives = K;
  batch.deleteFrom = {1, 4}; // (u2,u4), (u5,u2)
  batch.deleteTo = {3, 1};
  batch.insertFrom = {3, 1}; // (u4,u6):(10,2,12), (u2,u6):(12,1,14)
  batch.insertTo = {5, 5};
  batch.insertWeights = {10, 2, 12, 12, 1, 14};

  string dir = g_work + "/thesis-example";
  CaseFiles files = writeCase(dir, graph, batch, source);
  CsrGraph updated;
  ChangeBatch copy = batch;
  applyChangeBatch(graph, copy, updated);

  // Updated SOSP trees of sub-figures (a), (b), (c).
  const vector<vector<int>> expectedTrees = {{-1, 0, 0, 2, 3, 4, 5},
                                             {-1, 0, 0, 2, 3, 1, 5},
                                             {-1, 2, 0, 2, 3, 4, 4}};
  vector<string> trees;
  for (int k = 0; k < K; ++k) {
    string obj = files.init + "/obj" + to_string(k);
    string out = outputDir(files, "parallel", k);
    parallelSOSPUpdate(files.graph, obj + "/distances.txt", obj + "/tree.txt",
                       files.insert, files.remove, k, source,
                       out + "/distances.txt", out + "/tree.txt");
    vector<int> parent;
    readParents(out + "/tree.txt", n, parent);
    report("thesis-example", parent == expectedTrees[k],
           "objective " + to_string(k + 1) + " tree differs from the thesis");
    trees.push_back(out + "/tree.txt");
  }

  struct Expectation {
    vector<int> pref;
    vector<int> pathToU7;           // u7, its parent, ... , u1
    vector<long long> costOfU7;     // original objective values
  };
  const vector<Expectation> expectations = {
      {{4, 1, 4}, {6, 5, 1, 0}, {15, 3, 20}},     // sub-figures (d), (e)
      {{4, 4, 1}, {6, 4, 3, 2, 0}, {15, 24, 7}},  // sub-figures (f), (g)
  };
  for (const auto &expect : expectations) {
    string out = dir + "/combined_" + to_string(expect.pref[0]) +
                 to_string(expect.pref[1]) + to_string(expect.pref[2]);
    parallelCombinedGraph(files.graph, trees, K, source, out,
                          out + "/distances.txt", out + "/tree.txt",
                          expect.pref);
    vector<int> parent;
    readParents(out + "/tree.txt", n, parent);
    vector<int> path{6};
    while (parent[path.back()] >= 0) {
      path.push_back(parent[path.back()]);
    }
    vector<long long> costs;
    mospPathCosts(updated, parent, source, costs);
    vector<long long> costOfU7(costs.begin() + 6 * K, costs.begin() + 7 * K);
    string label = "Pref={" + to_string(expect.pref[0]) + "," +
                   to_string(expect.pref[1]) + "," + to_string(expect.pref[2]) +
                   "}";
    report("thesis-example", path == expect.pathToU7,
           label + ": MOSP path to u7 differs from the thesis");
    report("thesis-example", costOfU7 == expect.costOfU7,
           label + ": MOSP cost of u7 differs from the thesis");
  }
  cout << "thesis-example: 1 case (3 trees, 2 preference vectors)\n";
}

/// Count-to-infinity regressions found by the stress tests of the original
/// code: after a tree-edge deletion the head picks a descendant as its new
/// parent and the stale cycle only counts up by its (small) weight per
/// round, so the original loop hit maxIterations = n before the distances
/// of reachable vertices were correct (the BFS post-pass only repairs
/// unreachable vertices).
void runRegressions(unsigned int) {
  struct Case {
    int n, m, K, objective, changes, insertPct, maxWeight;
    unsigned int graphSeed, changeSeed;
  };
  const vector<Case> cases = {
      {6, 10, 2, 0, 6, 55, 50, 621705, 250813},  // stress test, 1 in ~500
      {13, 22, 3, 0, 6, 26, 50, 770968, 694580},
      {9, 17, 2, 1, 6, 81, 50, 115080, 943676},
  };
  int index = 0;
  for (const auto &c : cases) {
    string base = g_work + "/regression_" + to_string(index++);
    CsrGraph graph =
        randomGraph(base, c.n, c.m, c.K, c.maxWeight, c.graphSeed);
    generateChangedEdges(1, c.maxWeight, c.K, c.n, c.changes, c.insertPct,
                         100 - c.insertPct, true, true, true, false,
                         base + "/random/graphCsr", base + "/c/insert.txt",
                         base + "/c/delete.txt", c.changeSeed);
    ChangeBatch batch;
    readChangeBatch(base + "/c/insert.txt", base + "/c/delete.txt", c.K, c.n,
                    batch);
    CaseFiles files = writeCase(base + "/case", graph, batch, 0);
    CsrGraph updated, reverse;
    applyChangeBatch(graph, batch, updated);
    transposeCsrGraph(updated, reverse);
    auto parallel = [](const CaseFiles &f, int k, int s, const string &d,
                       const string &t) {
      string obj = f.init + "/obj" + to_string(k);
      return parallelSOSPUpdate(f.graph, obj + "/distances.txt",
                                obj + "/tree.txt", f.insert, f.remove, k, s, d,
                                t);
    };
    auto sequential = [](const CaseFiles &f, int k, int s, const string &d,
                         const string &t) {
      string obj = f.init + "/obj" + to_string(k);
      return sequentialSOSPUpdate(f.graph, obj + "/distances.txt",
                                  obj + "/tree.txt", f.insert, f.remove, k, s,
                                  d, t);
    };
    runAndCheck("regression/parallel", parallel, files, updated, reverse, 0);
    runAndCheck("regression/sequential", sequential, files, updated, reverse,
                0);
  }
  cout << "regressions: " << cases.size() << " cases\n";
}

/// Distance-only fallback with many equal-distance parents: every weight
/// is 2 * 10^9, so on the grid most vertices have two tight in-neighbours
/// and the parent recovery must pick the lower id (with random weights,
/// ties almost never occur, and a last-writer-wins recovery would pass).
void runLargeWeightTies(unsigned int seed) {
  const int W = 2000000000, source = 0;
  CsrGraph graph = gridGraph(320, 320, 2, W, 0.1, seed * 7u + 3u, W);
  ChangeGeneratorOptions options;
  options.numberOfChanges = 2000;
  options.insertionPercentage = 50;
  options.weightMin = W;
  options.weightMax = W;
  options.seed = seed + 3u;
  ChangeBatch batch;
  generateChangeBatch(graph, options, batch);
  const string dir = g_work + "/large-weights-ties";
  vector<long long> distances;
  vector<int> parents;
  for (int k = 0; k < 2; ++k) {
    vector<long long> dist;
    vector<int> parent;
    dijkstraCsrGraph(graph, k, source, dist, parent);
    distances.insert(distances.end(), dist.begin(), dist.end());
    parents.insert(parents.end(), parent.begin(), parent.end());
  }
  ChangeBatch copy = batch;
  CsrGraph updated;
  MospResult result;
  if (!mospUpdate(graph, copy, distances, parents, MospOptions(), updated,
                  result)) {
    report("large-weights", false, dir + ": mospUpdate failed");
    return;
  }
  for (int k = 0; k < 2; ++k) {
    report("large-weights", !result.objectiveStats[k].packedParents,
           dir + ": expected the distance-only fallback");
  }
  // Compares every tree with the canonical (lowest-id) Dijkstra tree.
  checkPipeline(dir, graph, batch, source, {});
}

/// Large weights: (n - 1) * maxWeight does not fit next to the parent ids
/// in a 64-bit word, so the GPU search keeps distances only and recovers
/// the parents afterwards (the path used beyond ~2^25 vertices).
void runLargeWeights(unsigned int seed) {
  const int source = 0;
  CsrGraph graph = gridGraph(320, 320, 2, INT_MAX, 0.1, seed * 13u + 5u);
  int cases = 0;
  for (bool safe : {true, false}) {
    ChangeGeneratorOptions options;
    options.numberOfChanges = 2000;
    options.insertionPercentage = 50;
    options.weightMax = INT_MAX;
    options.seed = seed + (safe ? 1u : 2u);
    options.safeDeletions = safe;
    ChangeBatch batch;
    generateChangeBatch(graph, options, batch);
    string dir = g_work + "/large-weights-" + (safe ? "safe" : "unsafe");

    // In-memory pipeline, plus a check that the fallback was used.
    vector<long long> distances;
    vector<int> parents;
    for (int k = 0; k < 2; ++k) {
      vector<long long> dist;
      vector<int> parent;
      dijkstraCsrGraph(graph, k, source, dist, parent);
      distances.insert(distances.end(), dist.begin(), dist.end());
      parents.insert(parents.end(), parent.begin(), parent.end());
    }
    ChangeBatch copy = batch;
    MospOptions mospOptions;
    CsrGraph updated;
    MospResult result;
    if (!mospUpdate(graph, copy, distances, parents, mospOptions, updated,
                    result)) {
      report("large-weights", false, dir + ": mospUpdate failed");
      continue;
    }
    for (int k = 0; k < 2; ++k) {
      report("large-weights", !result.objectiveStats[k].packedParents,
             dir + ": expected the distance-only fallback");
    }
    checkPipeline(dir, graph, batch, source, {});

    // File-based update on the same inputs.
    CaseFiles files = writeCase(dir, graph, batch, source);
    CsrGraph reverse;
    transposeCsrGraph(updated, reverse);
    auto parallel = [](const CaseFiles &f, int k, int s, const string &d,
                       const string &t) {
      string obj = f.init + "/obj" + to_string(k);
      return parallelSOSPUpdate(f.graph, obj + "/distances.txt",
                                obj + "/tree.txt", f.insert, f.remove, k, s, d,
                                t);
    };
    runAndCheck("large-weights/parallel", parallel, files, updated, reverse,
                source);
    ++cases;
  }
  runLargeWeightTies(seed);
  ++cases;
  cout << "large-weights: " << cases
       << " cases (distance-only fallback, with and without ties)\n";
}

/// false if following the parents from some vertex runs into a cycle (or
/// out of the id range) instead of ending at a vertex without parent.
bool parentsAcyclic(const vector<int> &parent) {
  const int n = static_cast<int>(parent.size());
  vector<char> state(n, 0); // 0: not seen, 1: on the current walk, 2: done
  for (int start = 0; start < n; ++start) {
    int v = start;
    while (v >= 0 && v < n && state[v] == 0) {
      state[v] = 1;
      v = parent[v];
    }
    if (v >= n || (v >= 0 && state[v] == 1)) {
      return false;
    }
    for (int u = start; u >= 0 && state[u] == 1; u = parent[u]) {
      state[u] = 2;
    }
  }
  return true;
}

/// Packed words at their limit: n = 2^17 - 1 vertices need b = 17 parent
/// bits and leave 47 bits for distances. On the path 0 -> 1 -> ... -> n-1
/// with every weight W = 2^30 + 2^14, (n - 1) * W = 2^47 - 2^15 still fits,
/// so the packed format is used, but an edge n-1 -> 1 forms the candidate
/// n * W > 2^47 - 1, which must not be packed (it would wrap around to a
/// small word and win). Three entry points: the pull pass (the edge is
/// inserted), the push loop (the edge exists and a cheaper last path edge
/// makes n-1 push it) and the search from scratch (Step 3's engine). Each
/// result must equal Dijkstra (distances and parents) and be acyclic.
void runPackingBoundary(unsigned int) {
  const int n = (1 << 17) - 1, source = 0;
  const int W = (1 << 30) + (1 << 14);
  auto path = [&](bool backEdge) {
    CsrGraph graph;
    graph.numberOfNodes = n;
    graph.numberOfObjectives = 1;
    graph.rowPtr.assign(n + 1, 0);
    for (int u = 0; u < n; ++u) {
      if (u + 1 < n) {
        graph.colInd.push_back(u + 1);
        graph.weights.push_back(W);
      } else if (backEdge) {
        graph.colInd.push_back(1);
        graph.weights.push_back(W);
      }
      graph.rowPtr[u + 1] = static_cast<int>(graph.colInd.size());
    }
    return graph;
  };
  struct Case {
    string name;
    bool backEdge;
    int from, to, weight; // the one inserted edge; from < 0: from scratch
  };
  const vector<Case> cases = {
      {"pull", false, n - 1, 1, W},
      {"push", true, n - 2, n - 1, W - 1},
      {"from-scratch", true, -1, -1, 0},
  };
  int ran = 0;
  for (const auto &c : cases) {
    const string name = "packing-boundary/" + c.name;
    CsrGraph original = path(c.backEdge), updated, reverse;
    ChangeBatch batch;
    batch.numberOfObjectives = 1;
    if (c.from >= 0) {
      batch.insertFrom = {c.from};
      batch.insertTo = {c.to};
      batch.insertWeights = {c.weight};
    }
    applyChangeBatch(original, batch, updated);
    transposeCsrGraph(updated, reverse);
    vector<long long> dist, refDist;
    vector<int> parent, refParent;
    dijkstraCsrGraph(original, 0, source, dist, parent);
    dijkstraCsrGraph(updated, 0, source, refDist, refParent);

    DeviceGraph graph;
    DeviceArray<long long> d_dist;
    DeviceArray<int> d_parent, d_heads, d_none;
    SospWorkspace workspace;
    SospStats stats;
    bool ok = uploadDeviceGraph(updated, graph) && d_dist.upload(dist) &&
              d_parent.upload(parent) && d_heads.upload(batch.insertTo) &&
              d_none.upload({});
    const long long delta = defaultDelta(updated.numberOfEdges(), n,
                                         static_cast<long long>(W) * n);
    if (ok && c.from >= 0) {
      DeviceChanges changes;
      changes.changedFrom = changes.changedTo = d_none.data();
      changes.insertHeads = d_heads.data();
      changes.numberOfInsertHeads = 1;
      ok = sospUpdateGpu(graph.out(0), graph.in(0), changes, source, delta, W,
                         workspace, d_dist.data(), d_parent.data(), &stats);
    } else if (ok) {
      ok = sospFromScratchGpu(graph.out(0), source, delta, W, workspace,
                              d_dist.data(), d_parent.data(), &stats);
    }
    ok = ok && d_dist.download(dist) && d_parent.download(parent);
    if (!ok) {
      report(name, false, "GPU call failed");
      continue;
    }
    report(name, stats.packedParents,
           "expected the packed format (the case tests its limit)");
    TreeCheck check =
        checkSospTree(reverse, 0, source, dist, parent, refDist, &refParent);
    report(name, check.ok(true), check.summary());
    report(name, parentsAcyclic(parent), "the parent graph has a cycle");
    ++ran;
  }
  cout << "packing-boundary: " << ran << " cases\n";
}

void writeText(const string &path, const string &text) {
  filesystem::create_directories(filesystem::path(path).parent_path());
  ofstream(path) << text;
}

/// The readers and mospUpdate reject invalid inputs instead of running on
/// them: weights outside [1, INT_MAX] (graph and inserts), ids or offsets
/// that do not fit an int, incomplete or duplicated distance and tree
/// files, negative distances, and initial trees of another source. Each
/// case differs from a valid input (accepted, checked first) in one value.
void runInputValidation(unsigned int) {
  const string dir = g_work + "/input-validation";
  int cases = 0;
  // Graph 0 -> 1 -> 2 with K = 2.
  auto graphAccepted = [&](const string &rows, const string &values) {
    const string prefix = dir + "/g" + to_string(cases++) + "/graphCsr";
    writeText(prefix + "RowPtr.txt", rows);
    writeText(prefix + "ColInd.txt", "1\n2\n");
    writeText(prefix + "Values.txt", values);
    CsrGraph graph;
    return readCsrGraph(prefix, graph);
  };
  const string rows = "0\n1\n2\n2\n";
  report("input-validation", graphAccepted(rows, "4 5\n2147483647 7\n"),
         "valid graph rejected");
  for (const string &values : {"0 5\n2 7\n", "4 5\n-3 7\n",
                               "4294967297 5\n2 7\n", "4 5\n2147483648 7\n"}) {
    report("input-validation", !graphAccepted(rows, values),
           "graph with weights '" + values + "' accepted");
  }
  report("input-validation",
         !graphAccepted("0\n1\n4294967298\n2\n", "4 5\n2 7\n"),
         "row pointer above INT_MAX accepted");

  auto batchAccepted = [&](const string &insert) {
    const string changes = dir + "/c" + to_string(cases++);
    writeText(changes + "/insert.txt", insert);
    writeText(changes + "/delete.txt", "0 1\n");
    ChangeBatch batch;
    return readChangeBatch(changes + "/insert.txt", changes + "/delete.txt",
                           2, 3, batch);
  };
  report("input-validation", batchAccepted("2 0 1 2147483647\n"),
         "valid batch rejected");
  for (const string &insert : {"2 0 0 3\n", "2 0 -5 3\n",
                               "2 0 4294967297 3\n", "2 3 1 1\n"}) {
    report("input-validation", !batchAccepted(insert),
           "insert line '" + insert + "' accepted");
  }

  auto distancesAccepted = [&](const string &text) {
    const string path = dir + "/d" + to_string(cases++) + ".txt";
    writeText(path, text);
    vector<long long> distances;
    return readDistances(path, 3, distances);
  };
  auto parentsAccepted = [&](const string &text) {
    const string path = dir + "/p" + to_string(cases++) + ".txt";
    writeText(path, text);
    vector<int> parents;
    return readParents(path, 3, parents);
  };
  report("input-validation", distancesAccepted("0 0\n1 4\n2 INF\n"),
         "valid distances rejected");
  report("input-validation", parentsAccepted("0 -1\n1 0\n2 -1\n"),
         "valid tree rejected");
  for (const string &text : {"0 0\n1 4\n", "0 0\n1 4\n1 4\n2 6\n",
                             "0 0\n1 -4\n2 6\n"}) {
    report("input-validation", !distancesAccepted(text),
           "distances '" + text + "' accepted");
  }
  for (const string &text : {"0 -1\n1 0\n", "0 -1\n1 0\n1 0\n2 1\n"}) {
    report("input-validation", !parentsAccepted(text),
           "tree '" + text + "' accepted");
  }

  // Trees of source 0 used with source 1.
  CsrGraph graph = gridGraph(4, 4, 2, 9, 0.0, 7);
  vector<long long> distances;
  vector<int> parents;
  for (int k = 0; k < 2; ++k) {
    vector<long long> dist;
    vector<int> parent;
    dijkstraCsrGraph(graph, k, 0, dist, parent);
    distances.insert(distances.end(), dist.begin(), dist.end());
    parents.insert(parents.end(), parent.begin(), parent.end());
  }
  for (int source : {0, 1}) {
    ChangeBatch batch;
    batch.numberOfObjectives = 2;
    MospOptions options;
    options.source = source;
    CsrGraph updated;
    MospResult result;
    const bool ok =
        mospUpdate(graph, batch, distances, parents, options, updated, result);
    report("input-validation", ok == (source == 0),
           source == 0 ? "mospUpdate rejected trees of its source"
                       : "mospUpdate accepted trees of another source");
    ++cases;
  }
  cout << "input-validation: " << cases << " cases\n";
}

/// The binary cache is used only for the text graph it was made from: not
/// for another graph with the same shape, not after the text files
/// changed (even to an older mtime), and not if its arrays are corrupt.
void runBinaryCache(unsigned int seed) {
  const string dir = g_work + "/binary-cache";
  const string a = dir + "/a/graphCsr", b = dir + "/b/graphCsr";
  const string cache = dir + "/cache.bin";
  CsrGraph graphA = gridGraph(6, 5, 2, 40, 0.1, seed * 3u + 1u);
  CsrGraph graphB = graphA; // same topology, other weights
  for (int &w : graphB.weights) {
    w = w % 7 + 1;
  }
  writeCsrGraph(a, graphA);
  writeCsrGraph(b, graphB);
  int cases = 0;
  auto sameGraph = [](const CsrGraph &x, const CsrGraph &y) {
    return x.numberOfNodes == y.numberOfNodes &&
           x.numberOfObjectives == y.numberOfObjectives &&
           x.rowPtr == y.rowPtr && x.colInd == y.colInd &&
           x.weights == y.weights;
  };
  auto check = [&](const string &label, const string &prefix,
                   bool expectCacheUsed, const CsrGraph &expected) {
    CsrGraph fromCache, loaded;
    const bool used = loadCsrGraphBinary(cache, fromCache, prefix);
    report("binary-cache", used == expectCacheUsed,
           label + (expectCacheUsed ? ": cache not used" : ": cache used"));
    report("binary-cache",
           loadCsrGraph(prefix, loaded, cache) && sameGraph(loaded, expected),
           label + ": loadCsrGraph returned the wrong graph");
    ++cases;
  };
  filesystem::remove(cache);
  check("no cache yet", a, false, graphA);   // writes the cache for a
  check("cache of a", a, true, graphA);
  check("a's cache for b", b, false, graphB); // rebuilds it for b
  check("cache of b", b, true, graphB);
  // b rewritten (same bytes, then other weights) with an older mtime.
  const auto values = b + "Values.txt";
  auto rewrite = [&](const CsrGraph &graph) {
    const auto before = filesystem::last_write_time(values);
    writeCsrGraph(b, graph);
    filesystem::last_write_time(values, before - chrono::hours(24 * 365));
  };
  rewrite(graphB);
  check("b rewritten, older mtime", b, false, graphB);
  rewrite(graphA);
  check("b changed, older mtime", b, false, graphA);
  // A corrupt column index behind a valid identity is rejected.
  {
    fstream file(cache, ios::in | ios::out | ios::binary);
    file.seekp(-static_cast<streamoff>(graphA.weights.size() * sizeof(int) +
                                       sizeof(int)),
               ios::end);
    const int bad = graphA.numberOfNodes + 5;
    file.write(reinterpret_cast<const char *>(&bad), sizeof(bad));
  }
  CsrGraph corrupt;
  report("binary-cache", !loadCsrGraphBinary(cache, corrupt, b),
         "cache with a column index out of range accepted");
  ++cases;
  cout << "binary-cache: " << cases << " cases\n";
}

/// Uniform generator mode reproduces generateChangedEdges() exactly.
void runGeneratorEquivalence(unsigned int seed) {
  int cases = 0;
  for (int rep = 0; rep < 5; ++rep) {
    unsigned int s = seed * 31u + rep;
    string dir = g_work + "/generator_" + to_string(rep);
    CsrGraph graph = randomGraph(dir, 30 + 10 * rep, 120 + 40 * rep, 2, 40, s);
    generateChangedEdges(1, 40, 2, graph.numberOfNodes, 25, 60, 40, true, true,
                         true, false, dir + "/random/graphCsr",
                         dir + "/old/insert.txt", dir + "/old/delete.txt", s);
    ChangeGeneratorOptions options;
    options.numberOfChanges = 25;
    options.insertionPercentage = 60;
    options.weightMax = 40;
    options.seed = s;
    ChangeBatch batch, old;
    generateChangeBatch(graph, options, batch);
    readChangeBatch(dir + "/old/insert.txt", dir + "/old/delete.txt", 2,
                    graph.numberOfNodes, old);
    bool same = batch.insertFrom == old.insertFrom &&
                batch.insertTo == old.insertTo &&
                batch.insertWeights == old.insertWeights &&
                batch.deleteFrom == old.deleteFrom &&
                batch.deleteTo == old.deleteTo;
    report("generator", same, dir + ": uniform mode differs from generateChangedEdges");
    ++cases;
  }
  cout << "generator: " << cases << " cases\n";
}

/// applyChangeBatch() produces the same graph as updateGraphCSR().
void runApplyEquivalence(unsigned int seed) {
  int cases = 0;
  for (int rep = 0; rep < 5; ++rep) {
    unsigned int s = seed * 17u + rep;
    string dir = g_work + "/apply_" + to_string(rep);
    CsrGraph graph = randomGraph(dir, 25, 90, 3, 30, s);
    generateChangedEdges(1, 30, 3, graph.numberOfNodes, 30, 50, 50, true, true,
                         true, false, dir + "/random/graphCsr",
                         dir + "/c/insert.txt", dir + "/c/delete.txt", s);
    updateGraphCSR(dir + "/random/graphCsr", dir + "/updated/graphCsr",
                   dir + "/c/insert.txt", dir + "/c/delete.txt", true);
    ChangeBatch batch;
    CsrGraph mine, theirs;
    readChangeBatch(dir + "/c/insert.txt", dir + "/c/delete.txt", 3,
                    graph.numberOfNodes, batch);
    applyChangeBatch(graph, batch, mine);
    readCsrGraph(dir + "/updated/graphCsr", theirs);
    auto edges = [](const CsrGraph &g) {
      vector<tuple<int, int, vector<int>>> list;
      for (int u = 0; u < g.numberOfNodes; ++u) {
        for (int e = g.rowPtr[u]; e < g.rowPtr[u + 1]; ++e) {
          vector<int> w(g.weights.begin() + static_cast<size_t>(e) * g.numberOfObjectives,
                        g.weights.begin() + static_cast<size_t>(e + 1) * g.numberOfObjectives);
          list.emplace_back(u, g.colInd[e], w);
        }
      }
      sort(list.begin(), list.end());
      return list;
    };
    report("apply", edges(mine) == edges(theirs),
           dir + ": applyChangeBatch differs from updateGraphCSR");
    ++cases;
  }
  cout << "apply: " << cases << " cases\n";
}

/// Marks a directory as mospTest's own work directory.
const char *const kWorkMarker = ".mospTest-work";

/// Empty the work directory, or create it, and mark it as ours. A
/// directory that is not empty is only deleted if an earlier run marked
/// it, so `--work .` or `--work ~` cannot wipe a checkout or a home
/// directory.
bool prepareWorkDirectory(const string &work) {
  namespace fs = std::filesystem;
  error_code ec;
  const fs::path dir(work);
  if (work.empty()) {
    cerr << "Error: --work needs a directory.\n";
    return false;
  }
  if (fs::exists(dir, ec)) {
    if (!fs::is_directory(dir, ec)) {
      cerr << "Error: --work " << work << " is not a directory.\n";
      return false;
    }
    const bool empty = fs::is_empty(dir, ec);
    const bool marked = fs::exists(dir / kWorkMarker, ec);
    if (!empty && !marked) {
      cerr << "Error: --work " << work
           << " is not empty and was not created by mospTest; refusing to "
              "delete it (choose a new or empty directory).\n";
      return false;
    }
    fs::remove_all(dir, ec);
    if (ec) {
      cerr << "Error: could not empty --work " << work << ": "
           << ec.message() << "\n";
      return false;
    }
  }
  fs::create_directories(dir, ec);
  ofstream marker(dir / kWorkMarker);
  if (ec || !marker) {
    cerr << "Error: could not create --work " << work << "\n";
    return false;
  }
  marker << "Work directory of mospTest; deleted by the next run.\n";
  return true;
}

} // namespace

int main(int argc, char **argv) {
  unsigned int seed = 1;
  string only;
  for (int i = 1; i < argc; ++i) {
    if (!strcmp(argv[i], "--seed") && i + 1 < argc) {
      seed = static_cast<unsigned int>(strtoul(argv[++i], nullptr, 10));
    } else if (!strcmp(argv[i], "--work") && i + 1 < argc) {
      g_work = argv[++i];
    } else if (!strcmp(argv[i], "--only") && i + 1 < argc) {
      only = argv[++i];
    } else {
      cerr << "usage: mospTest [--seed S] [--work DIR] [--only GROUP]\n";
      return 2;
    }
  }
  if (!prepareWorkDirectory(g_work)) {
    return 2;
  }
  // Silence the chatty library calls; results are reported below.
  streambuf *saved = cout.rdbuf();
  cout << "mospTest seed " << seed << "\n";

  auto quiet = [&](const string &name,
                   const function<void(unsigned int)> &test) {
    ostringstream sink;
    cout.rdbuf(sink.rdbuf());
    streambuf *savedErr = cerr.rdbuf(sink.rdbuf()); // expected rejections
    test(seed);
    cout.rdbuf(saved);
    cerr.rdbuf(savedErr);
    // Forward the group's summary line ("<name>: ...") and failures only.
    istringstream lines(sink.str());
    string line;
    while (getline(lines, line)) {
      if (line.rfind("  FAIL", 0) == 0 || line.rfind(name + ": ", 0) == 0) {
        cout << line << "\n";
      }
    }
  };
  const vector<pair<string, function<void(unsigned int)>>> groups = {
      {"thesis-example", runThesisExample},
      {"regressions", runRegressions},
      {"large-weights", runLargeWeights},
      {"packing-boundary", runPackingBoundary},
      {"input-validation", runInputValidation},
      {"binary-cache", runBinaryCache},
      {"generator", runGeneratorEquivalence},
      {"apply", runApplyEquivalence},
      {"sosp", runSosp},
  };
  bool ran = false;
  for (const auto &group : groups) {
    if (only.empty() || only == group.first) {
      quiet(group.first, group.second);
      ran = true;
    }
  }
  if (!ran) {
    cerr << "unknown group: " << only << "\n";
    return 2;
  }

  cout << (g_failures == 0 ? "=== mospTest: all checks passed ===\n"
                           : "=== mospTest: " + to_string(g_failures) +
                                 " checks FAILED ===\n");
  return g_failures == 0 ? 0 : 1;
}
