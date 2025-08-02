#ifndef SQL_ITERATORS_GPU_HELPERS_GPU_HASH_JOIN_H_
#define SQL_ITERATORS_GPU_HELPERS_GPU_HASH_JOIN_H_

#include "sql/iterators/external_helper_interface.h"
#include <cuda_runtime.h>
#include "sql_string.h"
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>


struct KeyIndexPair {
  std::string key; // Join key
  size_t index;  // Index of the full row in CPU build buffer
};

namespace gpuhashjoinhelpers {

static constexpr int MAX_KEY_SIZE = 32;
static constexpr size_t MIN_TABLE_CAPACITY = 1 << 20;
static constexpr size_t MAX_TABLE_CAPACITY = 1 << 27;
static constexpr uint32_t NOT_FOUND = 0xFFFFFFFF;

struct PackedKey {
  uint8_t data[MAX_KEY_SIZE];
};

struct HashEntry {
  PackedKey key;
  uint32_t index;
};

// Kernel launch wrappers (declarations)
bool LaunchInitHashTableKernel(HashEntry* d_hash_table, size_t capacity, 
                                uint32_t not_found, cudaStream_t stream);

bool LaunchBuildKernel(const PackedKey* d_keys, const uint32_t* d_indices,
                       HashEntry* d_hash_table, size_t capacity, size_t n_rows,
                       cudaStream_t stream);

bool LaunchProbeKernel(const PackedKey* d_probe_keys, size_t n_probe_keys,
                       const HashEntry* d_hash_table, size_t capacity,
                       uint32_t* d_result_buffer, cudaStream_t stream);

class GPUHashJoinHelper : public ExternalHelperInterface {
public:
  GPUHashJoinHelper();
  GPUHashJoinHelper(size_t batch_size_);
  ~GPUHashJoinHelper() override;

  bool Init(size_t capacity) override;

  bool SubmitBatch(const void* host_data, size_t n_rows) override;

  bool FetchResults(void* out_buffer, size_t* out_result_count) override;

  bool Synchronize() override;

  void Destroy() override;

  void SetStatus(const std::string& status) override;

private:
  bool SubmitBuildBatch(const void* host_data, size_t n_rows);
  bool SubmitProbeBatch(const void* host_data, size_t n_rows);
  void PrintHashTable();

  // GPU buffers
  PackedKey* d_keys_ = nullptr;
  uint32_t* d_indices_ = nullptr;
  HashEntry* d_hash_table_ = nullptr;
  PackedKey* d_probe_keys_ = nullptr;
  uint32_t* d_result_indices_ = nullptr;

  size_t batch_size;
  size_t capacity_ = 0;
  size_t last_n_tuples = 0;
  cudaStream_t stream_ = nullptr;

  std::string current_status_ = "UNINITIALIZED";
};

}  // namespace gpuhashjoinhelpers

#endif  // SQL_ITERATORS_GPU_HELPERS_GPU_HASH_JOIN_H_
