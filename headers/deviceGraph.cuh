#ifndef DEVICE_GRAPH_CUH
#define DEVICE_GRAPH_CUH

/**
 * @file deviceGraph.cuh
 * @brief A multi-objective graph resident on the GPU, shared by all
 *        objectives: out- and in-edge CSR with one weight column per
 *        objective.
 */

#include "deviceArray.cuh"
#include "sospUpdateGpu.cuh"

struct CsrGraph;

struct DeviceGraph {
  int numberOfNodes = 0;
  int numberOfEdges = 0;
  int numberOfObjectives = 0;
  DeviceArray<int> outRowPtr, outColInd, outWeights; ///< weights: K columns
  DeviceArray<int> inRowPtr, inColInd, inWeights;    ///< reverse graph

  /// Out-edge view of objective k.
  DeviceCsr out(int k) const;
  /// In-edge view of objective k.
  DeviceCsr in(int k) const;
};

/**
 * @brief Copy @p graph to the GPU once and build the reverse CSR and the
 *        objective-major weight columns there.
 *
 * Only the out-edge CSR crosses PCIe; the in-edge CSR is built on the GPU
 * (count, scan, fill), which also splits the edge-major weights into one
 * contiguous column per objective.
 */
bool uploadDeviceGraph(const CsrGraph &graph, DeviceGraph &device);

#endif // DEVICE_GRAPH_CUH
