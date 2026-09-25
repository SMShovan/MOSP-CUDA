// Driver that times the original code (bench/baseline; not part of the tag
// baseline-2026-09). Reproduces main.cu steps 6-7 on prepared inputs, with
// stage timers (stage-timers.patch):
//   [optional] per-objective runDijkstraCSR (initial SOSP tree, as main.cu:98)
//   per-objective parallelSOSPUpdate                         (main.cu:104)
//   parallelCombinedGraph                                    (main.cu:113)
// usage: mospBench <csrPrefix> <K> <source> <changesDir> <initDir> <outDir>
//                  [--with-dijkstra] [--seq] [--expected <dir>]
//   --with-dijkstra  also time the initial trees (Dijkstra per objective)
//   --seq            sequentialSOSPUpdate instead of the GPU update
//   --expected dir   compare the distances with dir/obj<k>/distancesUpdated.txt
//                    (`bin/mospPrep expected`)
// Environment: MOSP_SKIP_COMB skips the combined graph.
#include "dijkstra.cuh"
#include "parallelCombinedGraph.cuh"
#include "parallelSOSPUpdate.cuh"
#include "sequentialSOSPUpdate.cuh"
#include "prof.cuh"
#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>
using namespace std;

static bool sameFile(const string &a, const string &b, long long &mism) {
  ifstream fa(a), fb(b);
  if (!fa || !fb) return false;
  string la, lb;
  mism = 0;
  while (getline(fa, la) && getline(fb, lb))
    if (la != lb) ++mism;
  return mism == 0;
}

int main(int argc, char **argv) {
  if (argc < 7) { fprintf(stderr, "usage in source\n"); return 1; }
  string csr = argv[1];
  int K = atoi(argv[2]), source = atoi(argv[3]);
  string chg = argv[4], init = argv[5], out = argv[6];
  bool withDij = false, seqOnly = false;
  string expected;
  for (int i = 7; i < argc; ++i) {
    if (!strcmp(argv[i], "--with-dijkstra")) withDij = true;
    if (!strcmp(argv[i], "--seq")) seqOnly = true;
    if (!strcmp(argv[i], "--expected") && i + 1 < argc) expected = argv[++i];
  }
  auto T0 = chrono::steady_clock::now();
  {
    StageTimer st("");
#ifdef __CUDACC__
    st.begin("cuda_context_init");
    cudaFree(0);
#endif
    st.end();
  }
  vector<string> trees;
  for (int k = 0; k < K; ++k) {
    string pre = "obj" + to_string(k) + "/";
    string d0 = init + "/obj" + to_string(k);
    if (withDij) {
      StageTimer st(pre);
      st.begin("dijkstra_initial_tree_host");
      runDijkstraCSR(csr, k, source, d0 + "/distancesOriginal.txt",
                     d0 + "/SSSPTreeOriginal.txt");
    }
    string od = out + "/obj" + to_string(k);
    profPrefix() = pre;
    if (seqOnly) {
      StageTimer st(pre);
      st.begin("TOTAL_sequentialSOSPUpdate");
      sequentialSOSPUpdate(csr, d0 + "/distancesOriginal.txt",
                           d0 + "/SSSPTreeOriginal.txt", chg + "/insert.txt",
                           chg + "/delete.txt", k, source,
                           od + "/distancesParallelUpdate.txt",
                           od + "/SSSPTreeParallelUpdate.txt");
      st.end();
      profPrefix() = "";
      trees.push_back(od + "/SSSPTreeParallelUpdate.txt");
      continue;
    }
    {
      StageTimer st(pre);
      st.begin("TOTAL_parallelSOSPUpdate");
      if (!parallelSOSPUpdate(csr, d0 + "/distancesOriginal.txt",
                              d0 + "/SSSPTreeOriginal.txt", chg + "/insert.txt",
                              chg + "/delete.txt", k, source,
                              od + "/distancesParallelUpdate.txt",
                              od + "/SSSPTreeParallelUpdate.txt")) {
        fprintf(stderr, "parallelSOSPUpdate failed\n");
        return 2;
      }
    }
    profPrefix() = "";
    trees.push_back(od + "/SSSPTreeParallelUpdate.txt");
  }
  if (!seqOnly && !getenv("MOSP_SKIP_COMB")) {
    StageTimer st("");
    st.begin("TOTAL_parallelCombinedGraph");
    if (!parallelCombinedGraph(csr, trees, K, source, out + "/combinedGraph",
                               out + "/combinedGraph/distancesCsr.txt",
                               out + "/combinedGraph/SSSPTreeCsr.txt")) {
      fprintf(stderr, "combined failed\n");
      return 3;
    }
  }
  double tot = chrono::duration<double, milli>(chrono::steady_clock::now() - T0).count();
  profDump();
  printf("TOTAL_WALL_MS %.3f\n", tot);
  {
    ifstream ps("/proc/self/status");
    string l;
    while (getline(ps, l))
      if (l.rfind("VmHWM", 0) == 0) printf("HOST_PEAK_RSS %s\n", l.c_str());
  }
  if (!expected.empty()) {
    for (int k = 0; k < K; ++k) {
      long long mm = 0;
      bool ok = sameFile(expected + "/obj" + to_string(k) + "/distancesUpdated.txt",
                         out + "/obj" + to_string(k) + "/distancesParallelUpdate.txt", mm);
      printf("VALIDATE obj%d distances vs Dijkstra(updated): %s (mismatch lines=%lld)\n",
             k, ok ? "PASS" : "FAIL", mm);
    }
  }
  return 0;
}
