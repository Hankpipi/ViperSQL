#include "sql/iterators/helpers/gpu_hash_join.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstring>
#include <limits>
#include <new>

namespace gpuhashjoinhelpers {
namespace {

size_t BucketCapacityForNodeCount(size_t node_count) {
  size_t capacity = 1;
  while (capacity < node_count && capacity < kMaxBucketCapacity) {
    capacity <<= 1;
  }
  return capacity;
}

bool AllocationSize(size_t count, size_t element_size, size_t *bytes) {
  if (count > std::numeric_limits<size_t>::max() / element_size)
    return true;
  *bytes = count * element_size;
  return false;
}

}  // namespace

GPUHashJoinHelper::GPUHashJoinHelper(size_t batch_size)
    : batch_size_(std::max<size_t>(1, batch_size)) {}

GPUHashJoinHelper::~GPUHashJoinHelper() { Destroy(); }

bool GPUHashJoinHelper::Init() {
  Destroy();

  host_probe_keys_.reset(new (std::nothrow) HashKey[batch_size_]);
  host_result_indices_.reset(new (std::nothrow) uint32_t[batch_size_]);
  if (host_probe_keys_ == nullptr || host_result_indices_ == nullptr ||
      cudaMalloc(reinterpret_cast<void **>(&d_probe_keys_),
                 batch_size_ * sizeof(HashKey)) != cudaSuccess ||
      cudaMalloc(reinterpret_cast<void **>(&d_result_indices_),
                 batch_size_ * sizeof(uint32_t)) != cudaSuccess) {
    Destroy();
    return true;
  }

  failed_ = false;
  current_status_ = "INITIALIZED";
  return false;
}

bool GPUHashJoinHelper::SubmitBatch(const void *host_data, size_t n_rows) {
  if (failed_ || n_rows > batch_size_ ||
      (n_rows != 0 && host_data == nullptr)) {
    failed_ = true;
    return true;
  }
  if (n_rows == 0) {
    last_n_tuples_ = 0;
    return false;
  }

  if (current_status_ == "BUILD") {
    return SubmitBuildBatch(host_data, n_rows);
  }
  if (current_status_ == "PROBE") {
    return SubmitProbeBatch(host_data, n_rows);
  }
  failed_ = true;
  return true;
}

bool GPUHashJoinHelper::SubmitBuildBatch(const void *host_data, size_t n_rows) {
  const auto *pairs = static_cast<const KeyIndexPair *>(host_data);
  for (size_t i = 0; i < n_rows; ++i) {
    if (pairs[i].index == kNotFound) {
      failed_ = true;
      return true;
    }
    host_build_nodes_.push_back(
        HashNode{pairs[i].key, pairs[i].index, kNotFound});
  }
  last_n_tuples_ = 0;
  return false;
}

bool GPUHashJoinHelper::FinalizeBuildIndex() {
  if (build_finalized_)
    return false;
  if (failed_)
    return true;

  ReleaseBuildIndex();
  bucket_capacity_ = BucketCapacityForNodeCount(host_build_nodes_.size());

  size_t node_bytes = 0;
  size_t bucket_bytes = 0;
  if (AllocationSize(host_build_nodes_.size(), sizeof(HashNode), &node_bytes) ||
      AllocationSize(bucket_capacity_, sizeof(uint32_t), &bucket_bytes) ||
      node_bytes > std::numeric_limits<size_t>::max() - bucket_bytes) {
    failed_ = true;
    return true;
  }

  if (cudaMalloc(reinterpret_cast<void **>(&d_bucket_heads_), bucket_bytes) !=
      cudaSuccess) {
    failed_ = true;
    ReleaseBuildIndex();
    return true;
  }

  if (!host_build_nodes_.empty()) {
    if (cudaMalloc(reinterpret_cast<void **>(&d_nodes_), node_bytes) !=
            cudaSuccess ||
        cudaMemcpy(d_nodes_, host_build_nodes_.data(), node_bytes,
                   cudaMemcpyHostToDevice) != cudaSuccess) {
      failed_ = true;
      ReleaseBuildIndex();
      return true;
    }
  }

  if (LaunchInitBucketHeadsKernel(d_bucket_heads_, bucket_capacity_) ||
      (!host_build_nodes_.empty() &&
       LaunchBuildBucketIndexKernel(d_nodes_, host_build_nodes_.size(),
                                    d_bucket_heads_, bucket_capacity_)) ||
      cudaDeviceSynchronize() != cudaSuccess) {
    failed_ = true;
    return true;
  }

  build_finalized_ = true;
  return false;
}

bool GPUHashJoinHelper::SubmitProbeBatch(const void *host_data, size_t n_rows) {
  if (FinalizeBuildIndex())
    return true;

  const auto *pairs = static_cast<const KeyIndexPair *>(host_data);
  for (size_t i = 0; i < n_rows; ++i)
    host_probe_keys_[i] = pairs[i].key;

  if (cudaMemcpy(d_probe_keys_, host_probe_keys_.get(),
                 n_rows * sizeof(HashKey), cudaMemcpyHostToDevice) !=
          cudaSuccess ||
      LaunchProbeKernel(d_probe_keys_, n_rows, d_nodes_, d_bucket_heads_,
                        bucket_capacity_, d_result_indices_) ||
      cudaDeviceSynchronize() != cudaSuccess ||
      cudaMemcpy(host_result_indices_.get(), d_result_indices_,
                 n_rows * sizeof(uint32_t), cudaMemcpyDeviceToHost) !=
          cudaSuccess) {
    failed_ = true;
    return true;
  }

  last_n_tuples_ = n_rows;
  return false;
}

bool GPUHashJoinHelper::FetchResults(void *out_buffer,
                                     size_t *out_result_count) {
  if (out_result_count == nullptr ||
      (last_n_tuples_ != 0 && out_buffer == nullptr)) {
    failed_ = true;
    return true;
  }

  if (current_status_ != "PROBE") {
    *out_result_count = 0;
    return false;
  }

  if (last_n_tuples_ != 0) {
    std::memcpy(out_buffer, host_result_indices_.get(),
                last_n_tuples_ * sizeof(uint32_t));
  }

  *out_result_count = last_n_tuples_;
  return false;
}

bool GPUHashJoinHelper::Synchronize() {
  return failed_;
}

bool GPUHashJoinHelper::IsIdle() const { return true; }

void GPUHashJoinHelper::ReleaseBuildIndex() {
  if (d_nodes_ != nullptr) {
    cudaFree(d_nodes_);
    d_nodes_ = nullptr;
  }
  if (d_bucket_heads_ != nullptr) {
    cudaFree(d_bucket_heads_);
    d_bucket_heads_ = nullptr;
  }
  bucket_capacity_ = 0;
  build_finalized_ = false;
}

void GPUHashJoinHelper::Destroy() {
  ReleaseBuildIndex();
  if (d_probe_keys_ != nullptr) {
    cudaFree(d_probe_keys_);
    d_probe_keys_ = nullptr;
  }
  if (d_result_indices_ != nullptr) {
    cudaFree(d_result_indices_);
    d_result_indices_ = nullptr;
  }
  host_probe_keys_.reset();
  host_result_indices_.reset();
  // Release retained build storage at the helper recovery boundary.
  std::vector<HashNode>().swap(host_build_nodes_);
  last_n_tuples_ = 0;
  failed_ = false;
  current_status_ = "DESTROYED";
}

void GPUHashJoinHelper::SetStatus(const std::string &status) {
  if (status == "BUILD") {
    ReleaseBuildIndex();
    host_build_nodes_.clear();
    last_n_tuples_ = 0;
  }
  current_status_ = status;
}

}  // namespace gpuhashjoinhelpers
