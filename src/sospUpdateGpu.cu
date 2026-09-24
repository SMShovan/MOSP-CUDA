/**
 * @file sospUpdateGpu.cu
 * @brief Work-efficient, disconnection-safe SOSP update on the GPU, run by
 *        one persistent cooperative kernel (device-side control).
 *
 * ============================================================================
 * ALGORITHM
 * ============================================================================
 *
 * Step 1 (from the change list, grouped by destination):
 *   - Roots: the head v of every deleted or weight-increased edge (u,v)
 *     with Parent[v] == u.
 *   - Subtree invalidation: pointer jumping over the parent array marks
 *     every descendant of a root. Rounds stop as soon as no vertex is still
 *     jumping (at most ceil(log2 n) rounds). The marked vertices lose their
 *     distance (INF) and parent.
 *   - Pull pass: every invalidated vertex and every head of an inserted
 *     edge takes the best (distance, parent id) pair over its in-neighbours
 *     (a destination-grouped Step 1: one thread per destination).
 *
 * Step 2 (propagation) is a push-based near-far worklist (a Delta-stepping
 * variant, Davidson et al., IPDPS'14): vertices whose distance improved
 * relax their out-edges with a 64-bit atomicMin on the packed pair
 * (distance << b | parent). A vertex whose new distance is below the
 * current threshold joins the next near frontier, otherwise the far pile;
 * when the near frontier is empty the threshold moves to the smallest far
 * distance plus Delta.
 *
 * Properties:
 *   - Monotone: a distance only decreases, and every distance is an upper
 *     bound (valid vertices keep an intact tree path, invalidated ones
 *     start at INF). The loop terminates without an iteration cap, cannot
 *     count to infinity through a stale cycle, and vertices cut off from
 *     the source keep INF (no reachability post-pass).
 *   - Canonical: the packed atomicMin keeps, among equal distances, the
 *     lowest parent id. Every in-neighbour whose final distance differs
 *     from its old one pushes that final value, the pull pass covers the
 *     unchanged in-neighbours of invalidated vertices and insert heads,
 *     so a canonical input tree gives the canonical output tree.
 *   - Work-efficient: only improved vertices are expanded (no candidate
 *     re-scan of all in-edges, no O(n) reset per iteration).
 *
 * Control: everything above runs inside one cooperative kernel; the phases
 * are separated by grid-wide barriers and the loop decisions are taken on
 * the device, so an update costs one launch and one final copy of the
 * statistics instead of several host round trips per iteration. Counters
 * that are appended to in one phase and reset for a later one rotate over
 * three slots, which lets every push iteration and every pointer-jumping
 * round get by with a single grid barrier. Data written by other blocks is
 * read with __ldcg (L2), since L1 is not coherent within a kernel.
 *
 * Packing: b = number of bits needed for the vertex ids plus a "no parent"
 * value; the remaining 64 - b bits hold the distance, and the all-ones word
 * is INF. The distance bound (n - 1) * maxWeight must fit (checked).
 * ============================================================================
 */

#include "sospUpdateGpu.cuh"

#include "csrGraph.cuh"

#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <climits>
#include <iostream>

using namespace std;
namespace cg = cooperative_groups;

namespace {

using u64 = unsigned long long;
constexpr u64 PACKED_INF = ~0ULL;
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

/// Packed (distance, parent) words.
struct Packing {
  int parentBits;
  u64 noParent; // all-ones parent field

  __host__ __device__ u64 pack(u64 distance, int parent) const {
    return (distance << parentBits) |
           (parent < 0 ? noParent : static_cast<u64>(parent));
  }
  __device__ u64 distance(u64 word) const { return word >> parentBits; }
  __device__ int parent(u64 word) const {
    u64 p = word & noParent;
    return p == noParent ? -1 : static_cast<int>(p);
  }
  /// Largest distance that can be stored (the all-ones field is INF).
  u64 maxDistance() const { return (PACKED_INF >> parentBits) - 1; }
};

Packing makePacking(int numberOfNodes) {
  int bits = 1;
  while ((1ULL << bits) - 1 < static_cast<u64>(numberOfNodes)) {
    ++bits;
  }
  return {bits, (1ULL << bits) - 1};
}

/// Device-side control block of one update (initialized before launch).
struct Control {
  int listCount;     ///< candidates (invalidated + insert heads)
  int frontierCount; ///< vertices improved by the pull pass
  int nearCount[3];  ///< near-frontier appends, rotating slots
  int farCount;      ///< far pile size
  int far2Count;     ///< re-split far pile size
  int active[3];     ///< pointer jumping: a vertex still jumps
  int overflow;      ///< an input distance does not fit the packing
  int invalidated;
  int rounds;
  int iterations;
  int epochs;
  int generation;    ///< last stamp generation used
  long long pushes;
  u64 minimum;       ///< min-reduction slot (PACKED_INF when idle)
};

/// Parameters of the persistent kernel.
struct Params {
  DeviceCsr out, in;
  DeviceChanges changes;
  int source;
  bool fromScratch;
  Packing packing;
  u64 maxDistance;
  u64 delta;
  int maxRounds;
  int generation; ///< first stamp generation to use
  long long *distances;
  int *parent;
  u64 *packed;
  int *stamp, *inFar, *flag, *ancestor;
  int *candidates, *frontier, *nearA, *nearB, *farA, *farB;
  Control *control;
};

__device__ __forceinline__ int load(const int *p) { return __ldcg(p); }
__device__ __forceinline__ u64 load(const u64 *p) { return __ldcg(p); }

__device__ __forceinline__ bool claim(int *stamp, int v, int generation) {
  return atomicExch(&stamp[v], generation) != generation;
}

/// Warp-wide minimum of per-thread values, folded into *target.
__device__ void minimumInto(u64 value, u64 *target) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    value = min(value, __shfl_down_sync(0xffffffffu, value, offset));
  }
  if ((threadIdx.x & 31) == 0 && value != PACKED_INF) {
    atomicMin(target, value);
  }
}

__global__ void __launch_bounds__(BLOCK_SIZE)
    sospPersistentKernel(Params p) {
  cg::grid_group grid = cg::this_grid();
  const int tid = static_cast<int>(grid.thread_rank());
  const int threads = static_cast<int>(grid.size());
  const int n = p.out.numberOfNodes;
  const Packing packing = p.packing;
  Control *c = p.control;
  int generation = p.generation;
  int frontierCount = 0;

  if (p.fromScratch) {
    // ---- From scratch: only the source is finite. -------------------------
    for (int v = tid; v < n; v += threads) {
      p.packed[v] = v == p.source ? packing.pack(0, -1) : PACKED_INF;
    }
    if (tid == 0) {
      p.frontier[0] = p.source;
    }
    frontierCount = 1;
    grid.sync();
  } else {
    // ---- Pack the old tree; roots; ancestors. -----------------------------
    for (int v = tid; v < n; v += threads) {
      long long d = p.distances[v];
      if (d >= DISTANCE_INF / 2) {
        p.packed[v] = PACKED_INF;
      } else if (d < 0 || static_cast<u64>(d) > p.maxDistance) {
        c->overflow = 1;
        p.packed[v] = PACKED_INF;
      } else {
        p.packed[v] = packing.pack(static_cast<u64>(d), p.parent[v]);
      }
      p.ancestor[v] = p.parent[v];
    }
    for (int i = tid; i < p.changes.numberOfChanged; i += threads) {
      int v = p.changes.changedTo[i];
      if (p.parent[v] == p.changes.changedFrom[i]) {
        p.flag[v] = 1;
      }
    }
    grid.sync();

    // ---- Subtree invalidation by pointer jumping. -------------------------
    // Invariant for every vertex x: flag[x] == 1 implies a root among x and
    // its ancestors; flag[x] == 0 implies no root on the tree path from x up
    // to (excluding) ancestor[x]. A vertex either inherits its ancestor's
    // flag or jumps to the ancestor's ancestor, at least doubling the
    // covered distance, so ceil(log2 n) rounds suffice; the loop stops
    // earlier once no vertex is still jumping. Only v's thread writes
    // (ancestor[v], flag[v]) and every value another thread can observe
    // satisfies the invariant, so updating in place is safe.
    if (p.changes.numberOfChanged > 0) {
      for (int round = 0; round < p.maxRounds; ++round) {
        if (tid == 0) {
          c->active[(round + 1) % 3] = 0; // slot of the next round
          c->rounds = round + 1;
        }
        int jumping = 0;
        for (int v = tid; v < n; v += threads) {
          int a = load(&p.ancestor[v]);
          if (a < 0 || load(&p.flag[v])) {
            continue;
          }
          if (load(&p.flag[a])) {
            p.flag[v] = 1;
            continue;
          }
          int next = load(&p.ancestor[a]);
          p.ancestor[v] = next;
          jumping |= next >= 0 ? 1 : 0;
        }
        if (__any_sync(0xffffffffu, jumping) && (threadIdx.x & 31) == 0) {
          c->active[round % 3] = 1;
        }
        grid.sync();
        if (load(&c->active[round % 3]) == 0) {
          break;
        }
      }
    }

    // ---- Invalidate; candidates = invalidated + insert heads. ------------
    ++generation;
    for (int v = tid; v < n; v += threads) {
      if (load(&p.flag[v])) {
        p.flag[v] = 0; // leave the flags clean for the next update
        p.packed[v] = PACKED_INF;
        p.stamp[v] = generation;
        p.candidates[atomicAdd(&c->listCount, 1)] = v;
      }
    }
    grid.sync();
    if (tid == 0) {
      c->invalidated = load(&c->listCount);
    }
    for (int i = tid; i < p.changes.numberOfInsertHeads; i += threads) {
      int v = p.changes.insertHeads[i];
      if (v != p.source && claim(p.stamp, v, generation)) {
        p.candidates[atomicAdd(&c->listCount, 1)] = v;
      }
    }
    grid.sync();

    // ---- Pull pass. -------------------------------------------------------
    ++generation;
    const int candidateCount = load(&c->listCount);
    for (int i = tid; i < candidateCount; i += threads) {
      int v = load(&p.candidates[i]);
      u64 current = load(&p.packed[v]);
      u64 best = current;
      for (int e = p.in.rowPtr[v]; e < p.in.rowPtr[v + 1]; ++e) {
        int u = p.in.colInd[e];
        u64 word = load(&p.packed[u]);
        if (word != PACKED_INF) {
          best = min(best,
                     packing.pack(packing.distance(word) +
                                      static_cast<u64>(p.in.weights[e]),
                                  u));
        }
      }
      if (best < current) {
        u64 old = atomicMin(&p.packed[v], best);
        if (packing.distance(best) < packing.distance(old) &&
            claim(p.stamp, v, generation)) {
          p.frontier[atomicAdd(&c->frontierCount, 1)] = v;
        }
      }
    }
    grid.sync();
    frontierCount = load(&c->frontierCount);
  }

  // ---- Near-far propagation. ----------------------------------------------
  // Threshold = smallest frontier distance + delta.
  u64 local = PACKED_INF;
  for (int i = tid; i < frontierCount; i += threads) {
    u64 word = load(&p.packed[load(&p.frontier[i])]);
    if (word != PACKED_INF) {
      local = min(local, packing.distance(word));
    }
  }
  minimumInto(local, &c->minimum);
  grid.sync();
  const u64 smallest = load(&c->minimum);
  u64 threshold = (smallest == PACKED_INF ? 0 : smallest) + p.delta;

  // Every phase that produces a near list appends to nearCount[phase % 3];
  // at its start thread 0 zeroes the slot of the following phase, which was
  // last used two phases ago and read by everybody before the previous
  // barrier.
  int phase = 0;
  int *current = p.nearA, *next = p.nearB, *far = p.farA, *far2 = p.farB;
  ++generation;
  if (tid == 0) {
    c->nearCount[1] = 0;
  }
  for (int i = tid; i < frontierCount; i += threads) {
    int v = load(&p.frontier[i]);
    u64 word = load(&p.packed[v]);
    u64 d = word == PACKED_INF ? PACKED_INF : packing.distance(word);
    if (d < threshold) {
      if (claim(p.stamp, v, generation)) {
        current[atomicAdd(&c->nearCount[0], 1)] = v;
      }
    } else if (atomicExch(&p.inFar[v], 1) == 0) {
      far[atomicAdd(&c->farCount, 1)] = v;
    }
  }
  grid.sync();
  if (tid == 0) {
    c->minimum = PACKED_INF; // everybody read it before the barrier
  }
  int nearCount = load(&c->nearCount[0]);
  int iterations = 0, epochs = 0;
  long long pushes = 0;

  while (true) {
    if (nearCount > 0) {
      // -- One push iteration over the near frontier. --
      ++phase;
      ++generation;
      ++iterations;
      pushes += nearCount;
      if (tid == 0) {
        c->nearCount[(phase + 1) % 3] = 0;
      }
      int *slot = &c->nearCount[phase % 3];
      for (int i = tid; i < nearCount; i += threads) {
        int u = load(&current[i]);
        u64 word = load(&p.packed[u]);
        if (word == PACKED_INF) {
          continue;
        }
        u64 du = packing.distance(word);
        for (int e = p.out.rowPtr[u]; e < p.out.rowPtr[u + 1]; ++e) {
          int w = p.out.colInd[e];
          if (w == p.source) {
            continue;
          }
          u64 nd = du + static_cast<u64>(p.out.weights[e]);
          u64 candidate = packing.pack(nd, u);
          if (candidate >= load(&p.packed[w])) {
            continue;
          }
          u64 old = atomicMin(&p.packed[w], candidate);
          if (nd < packing.distance(old)) {
            if (nd < threshold) {
              if (claim(p.stamp, w, generation)) {
                next[atomicAdd(slot, 1)] = w;
              }
            } else if (atomicExch(&p.inFar[w], 1) == 0) {
              far[atomicAdd(&c->farCount, 1)] = w;
            }
          }
        }
      }
      grid.sync();
      nearCount = load(slot);
      int *t = current;
      current = next;
      next = t;
      continue;
    }

    // -- Near frontier empty: raise the threshold past the far pile. --
    const int farCount = load(&c->farCount);
    if (farCount == 0) {
      break;
    }
    ++epochs;
    local = PACKED_INF;
    for (int i = tid; i < farCount; i += threads) {
      u64 word = load(&p.packed[load(&far[i])]);
      if (word != PACKED_INF) {
        local = min(local, packing.distance(word));
      }
    }
    minimumInto(local, &c->minimum);
    grid.sync();
    threshold = max(threshold, load(&c->minimum)) + p.delta;
    ++phase;
    ++generation;
    if (tid == 0) {
      c->nearCount[(phase + 1) % 3] = 0;
    }
    int *slot = &c->nearCount[phase % 3];
    for (int i = tid; i < farCount; i += threads) {
      int v = load(&far[i]);
      u64 word = load(&p.packed[v]);
      u64 d = word == PACKED_INF ? PACKED_INF : packing.distance(word);
      if (d < threshold) {
        p.inFar[v] = 0;
        if (claim(p.stamp, v, generation)) {
          current[atomicAdd(slot, 1)] = v;
        }
      } else {
        far2[atomicAdd(&c->far2Count, 1)] = v;
      }
    }
    grid.sync();
    nearCount = load(slot);
    if (tid == 0) {
      c->farCount = load(&c->far2Count);
      c->far2Count = 0;
      c->minimum = PACKED_INF;
    }
    int *t = far;
    far = far2;
    far2 = t;
    grid.sync(); // farCount is updated before the next push appends to it
  }

  // ---- Unpack the result. ---------------------------------------------------
  for (int v = tid; v < n; v += threads) {
    u64 word = load(&p.packed[v]);
    if (word == PACKED_INF) {
      p.distances[v] = DISTANCE_INF;
      p.parent[v] = -1;
    } else {
      p.distances[v] = static_cast<long long>(packing.distance(word));
      p.parent[v] = packing.parent(word);
    }
  }
  if (tid == 0) {
    c->iterations = iterations;
    c->epochs = epochs;
    c->pushes = pushes;
    c->generation = generation;
  }
}

} // namespace

// ============================================================================
// Workspace
// ============================================================================

SospWorkspace::~SospWorkspace() { release(); }

void SospWorkspace::release() {
  cudaFree(packed);
  cudaFree(stamp);
  cudaFree(inFar);
  cudaFree(flag);
  cudaFree(ancestor);
  cudaFree(listA);
  cudaFree(listB);
  cudaFree(farA);
  cudaFree(farB);
  cudaFree(candidates);
  cudaFree(frontier);
  cudaFree(control);
  cudaFreeHost(hostControl);
  packed = nullptr;
  stamp = inFar = flag = ancestor = nullptr;
  listA = listB = farA = farB = candidates = frontier = nullptr;
  control = nullptr;
  hostControl = nullptr;
  capacity = 0;
  generation = 0;
  gridBlocks = 0;
}

bool SospWorkspace::reserve(int requested) {
  if (requested <= capacity) {
    return true;
  }
  release();
  int device = 0, cooperative = 0, sms = 0, perSm = 0;
  GPU_CHECK(cudaGetDevice(&device));
  GPU_CHECK(cudaDeviceGetAttribute(&cooperative, cudaDevAttrCooperativeLaunch,
                                   device));
  if (!cooperative) {
    cerr << "Error: the GPU does not support cooperative launches.\n";
    return false;
  }
  GPU_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount,
                                   device));
  GPU_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &perSm, sospPersistentKernel, BLOCK_SIZE, 0));
  if (perSm < 1) {
    cerr << "Error: the persistent kernel does not fit on an SM.\n";
    return false;
  }
  const size_t n = static_cast<size_t>(max(requested, 1));
  GPU_CHECK(cudaMalloc(&packed, n * sizeof(u64)));
  GPU_CHECK(cudaMalloc(&stamp, n * sizeof(int)));
  GPU_CHECK(cudaMalloc(&inFar, n * sizeof(int)));
  GPU_CHECK(cudaMalloc(&flag, n * sizeof(int)));
  GPU_CHECK(cudaMalloc(&ancestor, n * sizeof(int)));
  GPU_CHECK(cudaMalloc(&listA, n * sizeof(int)));
  GPU_CHECK(cudaMalloc(&listB, n * sizeof(int)));
  GPU_CHECK(cudaMalloc(&farA, n * sizeof(int)));
  GPU_CHECK(cudaMalloc(&farB, n * sizeof(int)));
  GPU_CHECK(cudaMalloc(&candidates, n * sizeof(int)));
  GPU_CHECK(cudaMalloc(&frontier, n * sizeof(int)));
  GPU_CHECK(cudaMalloc(&control, sizeof(Control)));
  GPU_CHECK(cudaMallocHost(&hostControl, sizeof(Control)));
  GPU_CHECK(cudaMemset(stamp, 0, n * sizeof(int)));
  GPU_CHECK(cudaMemset(inFar, 0, n * sizeof(int)));
  GPU_CHECK(cudaMemset(flag, 0, n * sizeof(int)));
  capacity = requested;
  generation = 0;
  gridBlocks = perSm * sms;
  return true;
}

int SospWorkspace::nextGeneration() {
  // An update uses a few generations plus one per push iteration and
  // threshold increase; restart well before the counter could wrap.
  if (generation > INT_MAX / 2) {
    cudaMemset(stamp, 0, static_cast<size_t>(capacity) * sizeof(int));
    generation = 0;
  }
  return ++generation;
}

long long defaultDelta(long long numberOfEdges, int numberOfNodes,
                       long long weightSum) {
  if (numberOfEdges <= 0 || numberOfNodes <= 0) {
    return 1;
  }
  // 32 * (average weight) / (average out-degree)
  double averageWeight = static_cast<double>(weightSum) / numberOfEdges;
  double averageDegree = static_cast<double>(numberOfEdges) / numberOfNodes;
  return max(1LL, static_cast<long long>(32.0 * averageWeight / averageDegree));
}

// ============================================================================
// Public entry points
// ============================================================================

namespace {

bool runPersistent(Params &params, SospWorkspace &ws, SospStats &stats) {
  params.packed = ws.packed;
  params.stamp = ws.stamp;
  params.inFar = ws.inFar;
  params.flag = ws.flag;
  params.ancestor = ws.ancestor;
  params.candidates = ws.candidates;
  params.frontier = ws.frontier;
  params.nearA = ws.listA;
  params.nearB = ws.listB;
  params.farA = ws.farA;
  params.farB = ws.farB;
  params.control = static_cast<Control *>(ws.control);
  params.generation = ws.nextGeneration();

  Control initial{};
  initial.minimum = PACKED_INF;
  GPU_CHECK(cudaMemcpyAsync(ws.control, &initial, sizeof(Control),
                            cudaMemcpyHostToDevice));
  void *args[] = {&params};
  GPU_CHECK(cudaLaunchCooperativeKernel(
      reinterpret_cast<void *>(sospPersistentKernel), ws.gridBlocks,
      BLOCK_SIZE, args, 0, 0));
  GPU_CHECK(cudaMemcpyAsync(ws.hostControl, ws.control, sizeof(Control),
                            cudaMemcpyDeviceToHost));
  GPU_CHECK(cudaStreamSynchronize(0));
  const Control &result = *static_cast<const Control *>(ws.hostControl);
  if (result.overflow) {
    cerr << "Error: an input distance does not fit the packed format.\n";
    return false;
  }
  ws.generation = max(ws.generation, result.generation);
  stats.invalidated = result.invalidated;
  stats.jumpRounds = result.rounds;
  stats.iterations = result.iterations;
  stats.epochs = result.epochs;
  stats.pushes = result.pushes;
  return true;
}

bool checkPacking(int n, long long maxWeight, const Packing &packing,
                  u64 &bound) {
  bound = static_cast<u64>(max(maxWeight, 1LL)) * static_cast<u64>(n - 1);
  if (bound > packing.maxDistance()) {
    cerr << "Error: distances up to " << bound << " do not fit the packed "
         << (64 - packing.parentBits) << "-bit distance field.\n";
    return false;
  }
  return true;
}

} // namespace

bool sospUpdateGpu(const DeviceCsr &out, const DeviceCsr &in,
                   const DeviceChanges &changes, int source, long long delta,
                   long long maxWeight, SospWorkspace &ws,
                   long long *d_distances, int *d_parent, SospStats *stats) {
  const int n = out.numberOfNodes;
  SospStats local;
  SospStats &s = stats != nullptr ? *stats : local;
  s = SospStats();
  if (n == 0) {
    return true;
  }
  if (!ws.reserve(n) || delta <= 0) {
    return false;
  }
  Params params{};
  params.packing = makePacking(n);
  if (!checkPacking(n, maxWeight, params.packing, params.maxDistance)) {
    return false;
  }
  int rounds = 0;
  while ((1LL << rounds) < n) {
    ++rounds;
  }
  params.out = out;
  params.in = in;
  params.changes = changes;
  params.source = source;
  params.fromScratch = false;
  params.delta = static_cast<u64>(delta);
  params.maxRounds = rounds + 1;
  params.distances = d_distances;
  params.parent = d_parent;
  return runPersistent(params, ws, s);
}

bool sospFromScratchGpu(const DeviceCsr &out, int source, long long delta,
                        long long maxWeight, SospWorkspace &ws,
                        long long *d_distances, int *d_parent,
                        SospStats *stats) {
  const int n = out.numberOfNodes;
  SospStats local;
  SospStats &s = stats != nullptr ? *stats : local;
  s = SospStats();
  if (n == 0) {
    return true;
  }
  if (!ws.reserve(n) || delta <= 0 || source < 0 || source >= n) {
    return false;
  }
  Params params{};
  params.packing = makePacking(n);
  if (!checkPacking(n, maxWeight, params.packing, params.maxDistance)) {
    return false;
  }
  params.out = out;
  params.in = out;
  params.source = source;
  params.fromScratch = true;
  params.delta = static_cast<u64>(delta);
  params.distances = d_distances;
  params.parent = d_parent;
  return runPersistent(params, ws, s);
}
