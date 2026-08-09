#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#include "sql/iterators/helpers/gpu_hash_join.h"

namespace gpuhashjoinhelpers {
namespace {

__device__ uint64_t RotateLeft64(uint64_t value, unsigned int shift) {
  return (value << shift) | (value >> (64 - shift));
}

__device__ uint64_t BucketHash(const HashKey &key) {
  uint64_t value =
      key.low ^ RotateLeft64(key.high, 29) ^ UINT64_C(0x9e3779b97f4a7c15);
  value ^= value >> 30;
  value *= UINT64_C(0xbf58476d1ce4e5b9);
  value ^= value >> 27;
  value *= UINT64_C(0x94d049bb133111eb);
  return value ^ (value >> 31);
}

__global__ void BuildBucketIndexKernel(HashNode *nodes, size_t node_count,
                                       uint32_t *bucket_heads,
                                       size_t bucket_capacity) {
  const size_t node_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (node_index >= node_count)
    return;

  const size_t bucket =
      BucketHash(nodes[node_index].key) & (bucket_capacity - 1);
  nodes[node_index].next =
      atomicExch(&bucket_heads[bucket], static_cast<uint32_t>(node_index));
}

__global__ void
ProbeBucketIndexKernel(const HashKey *probe_keys, size_t n_probe_keys,
                       const HashNode *nodes, const uint32_t *bucket_heads,
                       size_t bucket_capacity, uint32_t *result_buffer) {
  const size_t probe_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (probe_index >= n_probe_keys)
    return;

  const HashKey key = probe_keys[probe_index];
  const size_t bucket = BucketHash(key) & (bucket_capacity - 1);
  uint32_t node_index = bucket_heads[bucket];
  uint32_t result = kNotFound;

  while (node_index != kNotFound) {
    const HashNode node = nodes[node_index];
    if (node.key.low == key.low && node.key.high == key.high) {
      result = node.group_index;
      break;
    }
    node_index = node.next;
  }
  result_buffer[probe_index] = result;
}

}  // namespace

bool LaunchInitBucketHeadsKernel(uint32_t *d_bucket_heads,
                                 size_t bucket_capacity) {
  const cudaError_t error =
      cudaMemset(d_bucket_heads, 0xff, bucket_capacity * sizeof(uint32_t));
  return error != cudaSuccess;
}

bool LaunchBuildBucketIndexKernel(HashNode *d_nodes, size_t node_count,
                                  uint32_t *d_bucket_heads,
                                  size_t bucket_capacity) {
  if (node_count == 0)
    return false;
  constexpr int kBlockSize = 256;
  const int grid_size =
      static_cast<int>((node_count + kBlockSize - 1) / kBlockSize);
  BuildBucketIndexKernel<<<grid_size, kBlockSize>>>(
      d_nodes, node_count, d_bucket_heads, bucket_capacity);
  return cudaGetLastError() != cudaSuccess;
}

bool LaunchProbeKernel(const HashKey *d_probe_keys, size_t n_probe_keys,
                       const HashNode *d_nodes, const uint32_t *d_bucket_heads,
                       size_t bucket_capacity, uint32_t *d_result_buffer) {
  if (n_probe_keys == 0)
    return false;
  constexpr int kBlockSize = 256;
  const int grid_size =
      static_cast<int>((n_probe_keys + kBlockSize - 1) / kBlockSize);
  ProbeBucketIndexKernel<<<grid_size, kBlockSize>>>(
      d_probe_keys, n_probe_keys, d_nodes, d_bucket_heads, bucket_capacity,
      d_result_buffer);
  return cudaGetLastError() != cudaSuccess;
}

}  // namespace gpuhashjoinhelpers
