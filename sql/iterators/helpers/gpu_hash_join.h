#ifndef SQL_ITERATORS_GPU_HELPERS_GPU_HASH_JOIN_H_
#define SQL_ITERATORS_GPU_HELPERS_GPU_HASH_JOIN_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "sql/iterators/external_helper_interface.h"

namespace gpuhashjoinhelpers {

// The complete MySQL hash-join key can be arbitrarily long and may contain
// embedded NUL bytes.  The device index therefore uses a 128-bit fingerprint;
// GPUHashJoinIterator retains the complete key and verifies it before emitting
// a row.  A fingerprint collision can at worst produce a candidate that is
// rejected (or resolved through the iterator's exact host map), never a false
// SQL match.
struct HashKey {
  uint64_t low;
  uint64_t high;

  bool operator==(const HashKey &other) const {
    return low == other.low && high == other.high;
  }
};

inline uint64_t FinalizeHash64(uint64_t value) {
  value ^= value >> 33;
  value *= UINT64_C(0xff51afd7ed558ccd);
  value ^= value >> 33;
  value *= UINT64_C(0xc4ceb9fe1a85ec53);
  value ^= value >> 33;
  return value;
}

inline HashKey MakeHashKey(const char *data, size_t length) {
  uint64_t low = UINT64_C(14695981039346656037);
  uint64_t high = UINT64_C(7809847782465536322);
  for (size_t index = 0; index < length; ++index) {
    const uint8_t byte = static_cast<uint8_t>(data[index]);
    low = (low ^ byte) * UINT64_C(1099511628211);
    high = (high ^ static_cast<uint8_t>(byte + 0x9dU)) *
           UINT64_C(14029467366897019727);
  }
  low ^= static_cast<uint64_t>(length);
  high ^= static_cast<uint64_t>(length) << 1;
  return {FinalizeHash64(low), FinalizeHash64(high)};
}

static constexpr size_t kMaxBucketCapacity = 1U << 27;
static constexpr uint32_t kNotFound = UINT32_MAX;

// Each node represents one distinct exact host key group.  Nodes are linked
// into buckets, so an underestimated cardinality can only lengthen a chain; it
// cannot fill a fixed-capacity open-addressed table or silently drop a row.
struct HashNode {
  HashKey key;
  uint32_t group_index;
  uint32_t next;
};

struct KeyIndexPair {
  HashKey key;
  uint32_t index;
};

bool LaunchInitBucketHeadsKernel(uint32_t *d_bucket_heads,
                                 size_t bucket_capacity);

bool LaunchBuildBucketIndexKernel(HashNode *d_nodes, size_t node_count,
                                  uint32_t *d_bucket_heads,
                                  size_t bucket_capacity);

bool LaunchProbeKernel(const HashKey *d_probe_keys, size_t n_probe_keys,
                       const HashNode *d_nodes, const uint32_t *d_bucket_heads,
                       size_t bucket_capacity, uint32_t *d_result_buffer);

class GPUHashJoinHelper : public ExternalHelperInterface {
 public:
  GPUHashJoinHelper() = delete;
  explicit GPUHashJoinHelper(size_t batch_size);
  ~GPUHashJoinHelper() override;

  bool Init() override;
  bool SubmitBatch(const void *host_data, size_t n_rows) override;
  bool FetchResults(void *out_buffer, size_t *out_result_count) override;
  size_t ResultBufferCapacity(size_t submitted_rows) const override {
    // BUILD submissions only append candidate metadata on the host and never
    // produce output. PROBE is one-to-one, including not-found entries.
    return current_status_ == "PROBE" ? submitted_rows : 0;
  }
  bool Synchronize() override;
  bool IsIdle() const override;
  void Destroy() override;
  void SetStatus(const std::string &status) override;

 private:
  bool SubmitBuildBatch(const void *host_data, size_t n_rows);
  bool SubmitProbeBatch(const void *host_data, size_t n_rows);
  bool FinalizeBuildIndex();
  void ReleaseBuildIndex();

  std::vector<HashNode> host_build_nodes_;
  std::unique_ptr<HashKey[]> host_probe_keys_;
  std::unique_ptr<uint32_t[]> host_result_indices_;

  HashKey *d_probe_keys_ = nullptr;
  uint32_t *d_result_indices_ = nullptr;
  HashNode *d_nodes_ = nullptr;
  uint32_t *d_bucket_heads_ = nullptr;

  size_t batch_size_;
  size_t bucket_capacity_ = 0;
  size_t last_n_tuples_ = 0;
  bool build_finalized_ = false;
  bool failed_ = false;
  std::string current_status_ = "UNINITIALIZED";
};

}  // namespace gpuhashjoinhelpers

static_assert(sizeof(gpuhashjoinhelpers::HashKey) == 16,
              "GPU hash-key ABI changed");
static_assert(sizeof(gpuhashjoinhelpers::HashNode) == 24,
              "GPU hash-node ABI changed");

#endif  // SQL_ITERATORS_GPU_HELPERS_GPU_HASH_JOIN_H_
