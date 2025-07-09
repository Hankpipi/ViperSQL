#include "sql/iterators/gpu_helpers/gpu_hash_join.h"


namespace gpuhashjoinhelpers {

// Constructor
GPUHashJoinHelper::GPUHashJoinHelper() : d_keys_(nullptr), d_indices_(nullptr), d_hash_table_(nullptr),
                                         d_probe_keys_(nullptr), d_result_indices_(nullptr),
                                         capacity_(0), stream_(nullptr),
                                         current_status_("UNINITIALIZED") {}

// Destructor
GPUHashJoinHelper::~GPUHashJoinHelper() {
  Destroy();
}

bool GPUHashJoinHelper::Init(size_t capacity) {
  // Make sure the capacity_ is power of 2
  if (capacity < 1048576) {
    capacity_ = 1048576;
  } else {
    capacity_ = 1;
    while (capacity_ < capacity) {
      capacity_ <<= 1;
    }
  }

  cudaError_t err;

  // Allocate GPU hash table: size = capacity (uint32_t indices or flags)
  err = cudaMalloc(&d_hash_table_, capacity_ * sizeof(HashEntry));
  if (err != cudaSuccess) {
    log_to_file("cudaMalloc failed for d_hash_table_");
    return true;
  }

  // Zero entire memory (optional, to zero keys)
  err = cudaMemset(d_hash_table_, 0, capacity_ * sizeof(HashEntry));
  if (err != cudaSuccess) {
    log_to_file("cudaMemset zero failed for d_hash_table_");
    return true;
  }

  // Launch kernel to initialize index fields
  if (gpuhashjoinhelpers::LaunchInitHashTableKernel(d_hash_table_, capacity_, NOT_FOUND, stream_)) {
    return true;
  }

  // Allocate keys buffer (max capacity * max key size)
  err = cudaMalloc(&d_keys_, BATCH_SIZE * MAX_KEY_SIZE * sizeof(uint8_t));
  if (err != cudaSuccess) {
    log_to_file("cudaMalloc failed for d_keys_");
    return true;
  }

  // Allocate keys buffer (max capacity * max key size)
  err = cudaMalloc(&d_indices_, BATCH_SIZE * sizeof(uint32_t));
  if (err != cudaSuccess) {
    log_to_file("cudaMalloc failed for d_indices_");
    return true;
  }

  // Allocate probe keys buffer (same size as build keys buffer for simplicity)
  err = cudaMalloc(&d_probe_keys_, BATCH_SIZE * MAX_KEY_SIZE * sizeof(uint8_t));
  if (err != cudaSuccess) {
    log_to_file("cudaMalloc failed for d_probe_keys_");
    return true;
  }

  // Allocate result indices buffer for matched pairs
  err = cudaMalloc(&d_result_indices_, BATCH_SIZE * sizeof(uint32_t));
  if (err != cudaSuccess) {
    log_to_file("cudaMalloc failed for d_result_indices_");
    return true;
  }

  // Create CUDA stream
  err = cudaStreamCreate(&stream_);
  if (err != cudaSuccess) {
    log_to_file("cudaStreamCreate failed");
    return true;
  }

  current_status_ = "INITIALIZED";

  return false;
}

bool GPUHashJoinHelper::SubmitBatch(const void* host_data, size_t n_rows) {
  if (current_status_ == "BUILD") {
    return SubmitBuildBatch(host_data, n_rows);
  } else if (current_status_ == "PROBE") {
    return SubmitProbeBatch(host_data, n_rows);
  } else {
    log_to_file("SubmitBatch called in unknown status: " + current_status_);
    return true;
  }
}

// Helper to pack array of SQL String into PackedKey array for GPU
static bool PackKeyForGPU(const std::string& host_string, PackedKey* out_packed_key) {
  size_t len = host_string.size();
  if (len > MAX_KEY_SIZE) {
    log_to_file("PackKeyForGPU: key length exceeds MAX_KEY_SIZE: " + std::to_string(len));
    return true;
  }

  memcpy(out_packed_key->data, host_string.data(), len);
  if (len < MAX_KEY_SIZE) {
    memset(out_packed_key->data + len, 0, MAX_KEY_SIZE - len);
  }
  return false;
}

bool GPUHashJoinHelper::SubmitBuildBatch(const void* host_data, size_t n_rows) {
  const KeyIndexPair* pairs = static_cast<const KeyIndexPair*>(host_data);

  std::vector<PackedKey> packed_keys(n_rows);
  std::vector<uint32_t> indices(n_rows);

  for (size_t i = 0; i < n_rows; ++i) {
    if (PackKeyForGPU(pairs[i].key, &packed_keys[i]))
      return true;
    indices[i] = static_cast<uint32_t>(pairs[i].index);
  }

  cudaMemcpyAsync(d_keys_, packed_keys.data(), n_rows * sizeof(PackedKey),
                  cudaMemcpyHostToDevice, stream_);
  cudaMemcpyAsync(d_indices_, indices.data(), n_rows * sizeof(uint32_t),
                  cudaMemcpyHostToDevice, stream_);

  // Launch build kernel (wrapper function implemented in .cu)
  return gpuhashjoinhelpers::LaunchBuildKernel(
      d_keys_, d_indices_, d_hash_table_, capacity_, n_rows, stream_);
}

bool GPUHashJoinHelper::SubmitProbeBatch(const void* host_data, size_t n_rows) {
  const KeyIndexPair* pairs = static_cast<const KeyIndexPair*>(host_data);

  std::vector<PackedKey> packed_probe_keys(n_rows);

  for (size_t i = 0; i < n_rows; ++i) {
    if (PackKeyForGPU(pairs[i].key, &packed_probe_keys[i])) {
      return true;
    }
  }

  cudaError_t err;

  err = cudaMemcpyAsync(d_probe_keys_, packed_probe_keys.data(),
                        n_rows * sizeof(PackedKey),
                        cudaMemcpyHostToDevice, stream_);
  if (err != cudaSuccess) {
    log_to_file("cudaMemcpyAsync failed for probe keys in SubmitProbeBatch");
    return true;
  }
  last_n_tuples = n_rows;

  // Launch probe kernel with keys and indices buffers
  return gpuhashjoinhelpers::LaunchProbeKernel(
      d_probe_keys_, n_rows,
      d_hash_table_,
      capacity_,
      d_result_indices_,
      stream_);
}

bool GPUHashJoinHelper::FetchResults(void* out_buffer, size_t* out_result_count) {
  if (!out_buffer || !out_result_count) return true;

  // Return zero results if not in PROBE phase
  if (current_status_ != "PROBE") {
    *out_result_count = 0;
    return false;
  }

  // Copy entire device result buffer (size = BATCH_SIZE) to host directly into out_buffer
  cudaError_t err = cudaMemcpy(out_buffer, d_result_indices_,
                               BATCH_SIZE * sizeof(uint32_t),
                               cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    log_to_file("cudaMemcpy failed in FetchResults");
    return true;
  }

  *out_result_count = last_n_tuples;
  return false;
}

bool GPUHashJoinHelper::Synchronize() {
  cudaError_t err = cudaStreamSynchronize(stream_);
  if (err != cudaSuccess) {
    log_to_file("cudaStreamSynchronize failed in Synchronize");
    return true;
  }
  return false;
}

void GPUHashJoinHelper::Destroy() {
  if (d_keys_) {
    cudaFree(d_keys_);
    d_keys_ = nullptr;
  }
  if (d_indices_) {
    cudaFree(d_indices_);
    d_indices_ = nullptr;
  }
  if (d_hash_table_) {
    cudaFree(d_hash_table_);
    d_hash_table_ = nullptr;
  }
  if (d_probe_keys_) {
    cudaFree(d_probe_keys_);
    d_probe_keys_ = nullptr;
  }
  if (d_result_indices_) {
    cudaFree(d_result_indices_);
    d_result_indices_ = nullptr;
  }
  if (stream_) {
    cudaStreamDestroy(stream_);
    stream_ = nullptr;
  }
  current_status_ = "DESTROYED";
  log_to_file("GPUHashJoinHelper destroyed");
}

void GPUHashJoinHelper::SetStatus(const std::string& status) {
  current_status_ = status;
  log_to_file("GPUHashJoinHelper status set to: " + status);
}

void GPUHashJoinHelper::PrintHashTable() {
  if (!d_hash_table_ || capacity_ == 0) {
    log_to_file("PrintHashTable: Hash table not initialized");
    return;
  }

  // Allocate host buffer for the entire hash table
  std::vector<HashEntry> host_table(capacity_);

  // Copy device hash table to host
  cudaError_t err = cudaMemcpy(host_table.data(), d_hash_table_,
                               capacity_ * sizeof(HashEntry),
                               cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    log_to_file(std::string("PrintHashTable: cudaMemcpy failed: ") +
                cudaGetErrorString(err));
    return;
  }

  log_to_file("PrintHashTable: Dumping hash table contents:");

  // Iterate and print valid entries
  for (size_t i = 0; i < capacity_; ++i) {
    const HashEntry& entry = host_table[i];
    if (entry.index != NOT_FOUND) {
      // Convert key bytes to hex string for readability
      std::string key_hex;
      for (int b = 0; b < MAX_KEY_SIZE; ++b) {
        char buf[4];
        snprintf(buf, sizeof(buf), "%02X ", entry.key.data[b]);
        key_hex += buf;
      }
      log_to_file("Slot " + std::to_string(i) + ": index = " +
                  std::to_string(entry.index) + ", key = " + key_hex);
    }
  }
}

}  // namespace gpuhashjoinhelpers
