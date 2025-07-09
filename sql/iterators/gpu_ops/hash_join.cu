#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include "sql/iterators/gpu_helpers/gpu_hash_join.h"

namespace gpuhashjoinhelpers {

__global__ void InitHashTableKernel(HashEntry* table, size_t capacity, uint32_t not_found) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < capacity) {
    table[idx].index = not_found;
    // Optionally zero key bytes, if needed:
    // memset(&table[idx].key, 0, sizeof(PackedKey));
  }
}

// Kernel: Insert keys and indices into hash table using linear probing
__global__ void BuildHashTableKernel(const PackedKey* keys,
                                     const uint32_t* indices,
                                     HashEntry* hash_table,
                                     size_t capacity,
                                     size_t n_keys) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_keys) return;

  PackedKey key = keys[idx];
  uint32_t val = indices[idx];
  uint32_t hash = 5381;

  // Simple djb2 hash on key bytes
  for (int i = 0; i < MAX_KEY_SIZE; i++) {
    hash = ((hash << 5) + hash) + key.data[i];
  }

  uint32_t pos = hash & (capacity - 1);

  for (uint32_t i = 0; i < capacity; i++) {
    uint32_t slot = (pos + i) & (capacity - 1);

    uint32_t old_idx = atomicCAS(&hash_table[slot].index, NOT_FOUND, val);
    if (old_idx == NOT_FOUND) {
      // Insert key
      for (int j = 0; j < MAX_KEY_SIZE; j++) {
        hash_table[slot].key.data[j] = key.data[j];
      }
      return;
    } else {
      // Check key equality to avoid duplicate inserts
      bool match = true;
      for (int j = 0; j < MAX_KEY_SIZE; j++) {
        if (hash_table[slot].key.data[j] != key.data[j]) {
          match = false;
          break;
        }
      }
      if (match) {
        // Key already exists, no insertion needed
        return;
      }
    }
  }
  // Table full, insertion failed (optional: could log or track)
}

// Kernel: Probe keys in hash table and write matched indices to result buffer
__global__ void ProbeHashTableKernel(const PackedKey* probe_keys,
                                     size_t n_probe_keys,
                                     const HashEntry* hash_table,
                                     size_t capacity,
                                     uint32_t* result_buffer) {
  size_t probe_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (probe_idx >= n_probe_keys)
    return;

  PackedKey key = probe_keys[probe_idx];
  uint32_t hash = 5381;
  for (int i = 0; i < MAX_KEY_SIZE; i++) {
    hash = ((hash << 5) + hash) + key.data[i];
  }
  uint32_t pos = hash & (capacity - 1);

  // Default: no match found
  uint32_t match_index = NOT_FOUND;

  for (uint32_t i = 0; i < capacity; i++) {
    uint32_t slot = (pos + i) & (capacity - 1);
    if (hash_table[slot].index == NOT_FOUND) {
      break;  // empty slot, stop search
    }

    bool match = true;
    for (int j = 0; j < MAX_KEY_SIZE; j++) {
      if (hash_table[slot].key.data[j] != key.data[j]) {
        match = false;
        break;
      }
    }

    if (match) {
      match_index = hash_table[slot].index;
      break;
    }
  }
  result_buffer[probe_idx] = match_index;
}

bool LaunchInitHashTableKernel(HashEntry* d_hash_table, size_t capacity, uint32_t not_found, cudaStream_t stream) {
  const int blockSize = 256;
  const int gridSize = (capacity + blockSize - 1) / blockSize;
  InitHashTableKernel<<<gridSize, blockSize, 0, stream>>>(d_hash_table, capacity, not_found);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    log_to_file(std::string("InitHashTableKernel launch failed: ") + cudaGetErrorString(err));
    return true;
  }
  return false;
}

// Launch kernel wrapper for build stage
bool LaunchBuildKernel(const PackedKey* d_keys, const uint32_t* d_indices,
                       HashEntry* d_hash_table, size_t capacity, size_t n_rows,
                       cudaStream_t stream) {
  int blockSize = 256;
  int gridSize = (int)((n_rows + blockSize - 1) / blockSize);

  BuildHashTableKernel<<<gridSize, blockSize, 0, stream>>>(d_keys, d_indices, 
                                                          d_hash_table,
                                                          capacity, n_rows);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("BuildHashTableKernel launch failed: %s\n", cudaGetErrorString(err));
    return true;
  }
  return false;
}

// Launch kernel wrapper for probe stage
bool LaunchProbeKernel(const PackedKey* d_probe_keys, size_t n_probe_keys,
                       const HashEntry* d_hash_table, size_t capacity,
                       uint32_t* d_result_buffer, cudaStream_t stream) {

  // Reset result buffer to NOT_FOUND (0xFFFFFFFF)
  cudaError_t err = cudaMemsetAsync(d_result_buffer, NOT_FOUND, BATCH_SIZE * sizeof(uint32_t), stream);
  if (err != cudaSuccess) {
    printf("cudaMemsetAsync failed for result buffer: %s\n", cudaGetErrorString(err));
    return true;
  }

  int blockSize = 256;
  int gridSize = (int)((n_probe_keys + blockSize - 1) / blockSize);
  ProbeHashTableKernel<<<gridSize, blockSize, 0, stream>>>(d_probe_keys, n_probe_keys,
                                                           d_hash_table, capacity,
                                                           d_result_buffer);

  err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("ProbeHashTableKernel launch failed: %s\n", cudaGetErrorString(err));
    return true;
  }
  return false;
}

}  // namespace gpuhashjoinhelpers
