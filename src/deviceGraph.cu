/**
 * @file deviceGraph.cu
 * @brief Upload a multi-objective CSR graph once and derive its reverse
 *        graph and per-objective weight columns on the GPU.
 */

#include "deviceGraph.cuh"

#include "csrGraph.cuh"

#include <cub/device/device_scan.cuh>
#include <cuda_runtime.h>

#include <algorithm>
#include <iostream>

using namespace std;

namespace {

constexpr int BLOCK_SIZE = 256;

int blocks(long long work) {
  return static_cast<int>((max(work, 1LL) + BLOCK_SIZE - 1) / BLOCK_SIZE);
}

/// Edge-major weights (e * K + k) -> objective-major columns (k * m + e).
__global__ void splitWeightsKernel(const int *edgeMajor, int m, int K,
                                   int *columns) {
  long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= static_cast<long long>(m) * K) {
    return;
  }
  int e = static_cast<int>(i / K), k = static_cast<int>(i % K);
  columns[static_cast<size_t>(k) * m + e] = edgeMajor[i];
}

__global__ void inDegreeKernel(const int *colInd, int m, int *degree) {
  int e = blockIdx.x * blockDim.x + threadIdx.x;
  if (e < m) {
    atomicAdd(&degree[colInd[e]], 1);
  }
}

/// One thread per source vertex u writes its out-edges into the rows of
/// their heads (order inside a row is irrelevant to the algorithms).
__global__ void fillReverseKernel(int n, int m, int K, const int *rowPtr,
                                  const int *colInd, const int *outColumns,
                                  int *cursor, int *inColInd,
                                  int *inColumns) {
  int u = blockIdx.x * blockDim.x + threadIdx.x;
  if (u >= n) {
    return;
  }
  for (int e = rowPtr[u]; e < rowPtr[u + 1]; ++e) {
    int position = atomicAdd(&cursor[colInd[e]], 1);
    inColInd[position] = u;
    for (int k = 0; k < K; ++k) {
      inColumns[static_cast<size_t>(k) * m + position] =
          outColumns[static_cast<size_t>(k) * m + e];
    }
  }
}

} // namespace

DeviceCsr DeviceGraph::out(int k) const {
  DeviceCsr view;
  view.numberOfNodes = numberOfNodes;
  view.numberOfEdges = numberOfEdges;
  view.rowPtr = outRowPtr.data();
  view.colInd = outColInd.data();
  view.weights = outWeights.data() + static_cast<size_t>(k) * numberOfEdges;
  return view;
}

DeviceCsr DeviceGraph::in(int k) const {
  DeviceCsr view;
  view.numberOfNodes = numberOfNodes;
  view.numberOfEdges = numberOfEdges;
  view.rowPtr = inRowPtr.data();
  view.colInd = inColInd.data();
  view.weights = inWeights.data() + static_cast<size_t>(k) * numberOfEdges;
  return view;
}

bool uploadDeviceGraph(const CsrGraph &graph, DeviceGraph &device) {
  const int n = graph.numberOfNodes;
  const int m = graph.numberOfEdges();
  const int K = graph.numberOfObjectives;
  device.numberOfNodes = n;
  device.numberOfEdges = m;
  device.numberOfObjectives = K;

  DeviceArray<int> edgeMajor, cursor(static_cast<size_t>(n) + 1);
  if (!device.outRowPtr.upload(graph.rowPtr) ||
      !device.outColInd.upload(graph.colInd) || !edgeMajor.upload(graph.weights) ||
      !device.outWeights.allocate(static_cast<size_t>(m) * K) ||
      !device.inRowPtr.allocate(static_cast<size_t>(n) + 1) ||
      !device.inColInd.allocate(m) ||
      !device.inWeights.allocate(static_cast<size_t>(m) * K)) {
    cerr << "Error: could not allocate the graph on the GPU.\n";
    return false;
  }
  splitWeightsKernel<<<blocks(static_cast<long long>(m) * K), BLOCK_SIZE>>>(
      edgeMajor.data(), m, K, device.outWeights.data());

  // Reverse CSR: in-degrees, exclusive scan, fill.
  cudaMemsetAsync(cursor.data(), 0, (static_cast<size_t>(n) + 1) * sizeof(int));
  inDegreeKernel<<<blocks(m), BLOCK_SIZE>>>(device.outColInd.data(), m,
                                            cursor.data());
  size_t bytes = 0;
  cub::DeviceScan::ExclusiveSum(nullptr, bytes, cursor.data(),
                                device.inRowPtr.data(), n + 1);
  DeviceArray<unsigned char> scratch(max<size_t>(bytes, 1));
  cub::DeviceScan::ExclusiveSum(scratch.data(), bytes, cursor.data(),
                                device.inRowPtr.data(), n + 1);
  cudaMemcpyAsync(cursor.data(), device.inRowPtr.data(),
                  static_cast<size_t>(n) * sizeof(int),
                  cudaMemcpyDeviceToDevice);
  fillReverseKernel<<<blocks(n), BLOCK_SIZE>>>(
      n, m, K, device.outRowPtr.data(), device.outColInd.data(),
      device.outWeights.data(), cursor.data(), device.inColInd.data(),
      device.inWeights.data());
  cudaError_t status = cudaDeviceSynchronize();
  if (status == cudaSuccess) {
    status = cudaGetLastError();
  }
  if (status != cudaSuccess) {
    cerr << "CUDA error while building the device graph: "
         << cudaGetErrorString(status) << "\n";
    return false;
  }
  return true;
}
