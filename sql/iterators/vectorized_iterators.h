#ifndef SQL_ITERATORS_GPU_ITERATORS_H_
#define SQL_ITERATORS_GPU_ITERATORS_H_

#include "sql/iterators/composite_iterators.h"
#include "sql/iterators/external_helper_buffer.h"
#include "sql/iterators/hash_join_iterator.h"
#include "sql/iterators/helpers/gpu_hash_join.h"

#include <algorithm>
#include <deque>
#include <limits>
#include <memory>
#include <queue>
#include <string>
#include <unordered_map>
#include <vector>

class GPUHashJoinIterator : public RowIterator {
 public:
  static constexpr size_t kMinHostMemoryBudget =
      kExternalHelperBatchSize *
      (sizeof(gpuhashjoinhelpers::KeyIndexPair) + sizeof(uint32_t));

  static bool HasSufficientHostMemoryBudget(size_t max_memory_available) {
    return max_memory_available >= kMinHostMemoryBudget;
  }

  GPUHashJoinIterator(THD *thd,
                      unique_ptr_destroy_only<RowIterator> build_input,
                      const Prealloced_array<TABLE *, 4> &build_input_tables,
                      double estimated_build_rows,
                      unique_ptr_destroy_only<RowIterator> probe_input,
                      const Prealloced_array<TABLE *, 4> &probe_input_tables,
                      bool store_rowids, table_map tables_to_get_rowid_for,
                      size_t max_memory_available,
                      const std::vector<HashJoinCondition> &join_conditions,
                      bool allow_spill_to_disk, JoinType join_type,
                      const Mem_root_array<Item *> &extra_conditions,
                      bool probe_input_batch_mode,
                      uint64_t *hash_table_generation);

  bool Init() override;
  int Read() override;

  void SetNullRowFlag(bool is_null_row) override {
    if (m_native_fallback != nullptr) {
      m_native_fallback->SetNullRowFlag(is_null_row);
      return;
    }
    m_build_input->SetNullRowFlag(is_null_row);
    m_probe_input->SetNullRowFlag(is_null_row);
  }

  void EndPSIBatchModeIfStarted() override {
    if (m_native_fallback != nullptr) {
      m_native_fallback->EndPSIBatchModeIfStarted();
      return;
    }
    m_build_input->EndPSIBatchModeIfStarted();
    m_probe_input->EndPSIBatchModeIfStarted();
  }

  void UnlockRow() override {
    // Since both inputs may have been materialized to disk, we cannot unlock
    // them.
  }

 private:
  class BuildPrefixIterator;

  enum class JoinKeyStatus { kOk, kNull, kError };

  struct PackedRowBlock {
    std::vector<uint8_t> bytes;
    size_t used{0};
    // Every PackedRowRef into this block owns one reference. A probe's packed
    // snapshot and immutable serialized-key suffix, when present, share that
    // single reference.
    size_t outstanding_refs{0};
  };

  struct PackedRowRef {
    const uint8_t *data{nullptr};
    size_t size{0};
    PackedRowBlock *block{nullptr};
  };

  struct BuildKeyGroup {
    PackedRowRef key;
    size_t first_row;
    size_t last_row;
    size_t next_hash_collision;
  };

  struct HashKeyHasher {
    size_t operator()(const gpuhashjoinhelpers::HashKey &key) const noexcept {
      return static_cast<size_t>(gpuhashjoinhelpers::FinalizeHash64(
          key.low ^ ((key.high << 29) | (key.high >> 35))));
    }
  };

  struct ProbeRow {
    gpuhashjoinhelpers::HashKey hash_key;
    PackedRowRef packed_row;
    size_t key_length{0};
    size_t memory_charge{0};
  };

  static constexpr size_t kNoBuildRow = std::numeric_limits<size_t>::max();

  // Underlying build and probe iterators
  unique_ptr_destroy_only<RowIterator> m_build_input;
  unique_ptr_destroy_only<RowIterator> m_probe_input;

  // Preserve the original table arrays for a lossless handoff to MySQL's
  // native HashJoinIterator. TableCollection intentionally stores a packed
  // projection and is not the constructor API of the native iterator.
  Prealloced_array<TABLE *, 4> m_build_input_table_array;
  Prealloced_array<TABLE *, 4> m_probe_input_table_array;

  // Table collections for build and probe inputs
  pack_rows::TableCollection m_build_input_tables;
  pack_rows::TableCollection m_probe_input_tables;

  table_map m_tables_to_get_rowid_for;
  bool m_store_rowids;

  // Join conditions
  Prealloced_array<HashJoinCondition, 4> m_join_conditions;

  // Combined extra conditions
  Item *m_extra_condition{nullptr};

  JoinType m_join_type;
  bool m_allow_spill_to_disk;
  double m_estimated_build_rows;
  bool m_probe_input_batch_mode;
  uint64_t *m_hash_table_generation;
  size_t m_max_memory_available;
  size_t m_build_host_memory_limit{0};
  // Metadata and arena allocations are accounted independently. Arena fields
  // track the exact retained vector capacities of their stable blocks.
  size_t m_build_host_memory_charge{0};
  size_t m_build_row_arena_memory_charge{0};
  size_t m_build_key_arena_memory_charge{0};
  size_t m_probe_host_memory_charge{0};
  size_t m_probe_arena_memory_charge{0};
  std::vector<PackedRowRef> build_rows_buffer;
  std::deque<std::unique_ptr<PackedRowBlock>> m_build_row_blocks;
  std::deque<std::unique_ptr<PackedRowBlock>> m_build_key_blocks;
  // Index the exact host fallback by the fingerprint we already compute for
  // the GPU. Distinct exact keys with the same 128-bit fingerprint are linked
  // explicitly and compared byte-for-byte, preserving collision correctness
  // without hashing/allocating a second copy of every serialized key.
  std::unordered_map<gpuhashjoinhelpers::HashKey, size_t, HashKeyHasher>
      m_build_hash_to_group;
  std::vector<BuildKeyGroup> m_build_groups;
  std::vector<size_t> m_next_build_row;
  std::queue<ProbeRow> m_probe_rows;
  std::deque<std::unique_ptr<PackedRowBlock>> m_probe_row_blocks;
  ProbeRow m_retained_probe_frontier{};
  bool m_has_retained_probe_frontier{false};
  const uint8_t *m_probe_table_buffer_owner{nullptr};

  // Buffer manager encapsulating input batch and result queue
  String m_buffer;
  std::unique_ptr<ViperFlow<gpuhashjoinhelpers::KeyIndexPair, uint32_t>>
      m_buffer_manager;

  // Once ownership has moved here, all subsequent Init()/Read() calls use the
  // native implementation. This preserves native join_buffer_size spilling
  // and hash-table refill behavior after an underestimated GPU build.
  unique_ptr_destroy_only<HashJoinIterator> m_native_fallback;

  size_t m_row_size;
  bool m_probe_input_exhausted{false};
  bool m_probe_batch_flushed{false};
  bool m_probe_batch_mode_started{false};
  // Returning an older queued match overwrites the probe child's TABLE
  // buffers. Restore the newest queued probe before advancing that child.
  bool m_probe_frontier_overwritten{false};
  bool m_join_result_empty{false};
  bool m_use_gpu{true};
  bool m_release_probe_row_on_next_read{false};
  size_t m_active_build_row{kNoBuildRow};

  // Extract join key from the current row of the given tables' buffers into
  // m_buffer
  JoinKeyStatus
  extract_join_key_for_row(THD *thd, const pack_rows::TableCollection &tables);

  // Activate an exact build-key group for the probe row at the queue front.
  void ActivateBuildGroup(size_t group_index);

  // Resolve a fingerprint to an exact serialized build key. Returns
  // kNoBuildRow when the key is absent.
  size_t FindExactBuildGroup(const char *key, size_t key_length,
                             const gpuhashjoinhelpers::HashKey &hash_key) const;

  // Resolve an exact key starting at a previously located fingerprint-chain
  // head. The build loop retains its unordered-map iterator and calls this
  // helper so a new distinct key does not probe the host map twice.
  size_t FindExactBuildGroupInCollisionChain(const char *key, size_t key_length,
                                             size_t group_index) const;

  // Store a row in a stable block arena. New blocks are sized to the required
  // bytes and never resized, so restored BLOB/GEOMETRY pointers remain valid
  // until the owning block is released.
  PackedRowRef
  StoreRowInArena(const pack_rows::TableCollection &tables, size_t row_size,
                  std::deque<std::unique_ptr<PackedRowBlock>> *blocks);

  // Store the packed probe row and its serialized exact join key in one arena
  // allocation. The key begins immediately after packed_row.size bytes.
  PackedRowRef
  StoreProbeRowAndKeyInArena(const pack_rows::TableCollection &tables,
                             size_t row_size, const char *key,
                             size_t key_length);

  // Copy an exact serialized key into a stable arena block.
  PackedRowRef
  StoreBytesInArena(const char *data, size_t length,
                    std::deque<std::unique_ptr<PackedRowBlock>> *blocks);

  // Remove the probe snapshot at the FIFO head and release completed arena
  // blocks in order.
  void PopFrontProbeRow();

  // Remove a completed output row from result alignment while retaining its
  // arena references until a nested probe child has advanced past the TABLE
  // buffers that may still point into that row.
  void RetainFrontProbeRowUntilChildRead();
  void ReleaseRetainedProbeFrontier();

  // An emitted queued row can replace the probe child's live TABLE frontier.
  // Restore the newest queued row before either CPU-draining older alignment
  // entries or advancing a non-exhausted child.
  void RestoreProbeInputFrontierIfOverwritten();

  // Release arena blocks which no queued probe references.
  void ReclaimUnusedProbeBlocks();

  // Conservative, saturating accounting for variable host memory. Fixed GPU
  // staging is reserved separately; stable arenas use exact block capacities.
  size_t BuildRowMemoryCharge(bool creates_group) const;
  size_t PredictedArenaAllocation(
      const std::deque<std::unique_ptr<PackedRowBlock>> &blocks,
      size_t required_bytes) const;
  size_t UnusedBuildContainerReserveCharge(size_t build_rows,
                                           size_t build_groups) const;
  bool BuildMemoryWouldOverflow(size_t additional_metadata_charge,
                                size_t row_required_bytes,
                                size_t key_required_bytes,
                                bool creates_group) const;
  bool RetainedBuildMemoryExceedsLimit() const;
  bool ProbeMemoryWouldOverflow(size_t additional_metadata_charge,
                                size_t arena_required_bytes = 0) const;
  bool RetainedProbeMemoryExceedsLimit() const;

  void ClearProbeRows();

  // Initialize the probe side after building the host and device indexes.
  bool InitProbeInput();

  // Transfer child ownership to MySQL's native hash join. If the build child
  // was initialized, build_rows_buffer contains its consumed prefix in input
  // order. A completed-build handoff must not read that child again during the
  // prefix replay's first execution.
  bool InitNativeFallback(bool build_child_already_initialized,
                          bool build_child_already_exhausted);

  void EnsureGPUBufferManager();

  void DisableGPU();

  // Emit the next candidate in the active duplicate group. Returns 0 for a
  // row, -1 when the group was exhausted/rejected, and 1 on expression error.
  int ReadNextActiveJoinedRow();
};

// Store the current row from the given tables' buffers into a CPU memory buffer
std::vector<uint8_t>
store_row_to_buffer(const pack_rows::TableCollection &tables, size_t row_size);

/**
  Return a stable, privacy-preserving key for runtime semantic batch feedback.

  The key combines the current database, statement text, semantic condition,
  relational input-table bitmap, and logarithmic input-cardinality estimate.
  Thus moving the same predicate to another join prefix does not reuse rewards
  from an incompatible physical stage. Only the hash is retained by the
  controller; query text and rendered row values are never stored.
*/
std::string SemanticBatchFeedbackKey(THD *thd, const Item *condition,
                                     table_map input_tables,
                                     size_t estimated_rows);

class VectorizedFilterIterator final : public RowIterator {
 public:
  VectorizedFilterIterator(THD *thd,
                           unique_ptr_destroy_only<RowIterator> source,
                           pack_rows::TableCollection tables, Item *condition,
                           size_t num_rows_estimate)
      : RowIterator(thd), m_source(std::move(source)),
        m_tables(std::move(tables)), m_condition(condition),
        m_buffer_manager(
            64LL * 1024 * 1024,
            [&]() -> const char * {
              if (!condition)
                return "semantic_filter";
              if (condition->type() == Item::COND_ITEM) {
                Item_cond_and *and_cond =
                    static_cast<Item_cond_and *>(condition);
                List_iterator<Item> it(*and_cond->argument_list());
                Item *first_arg = it++;
                return static_cast<const Item_func *>(first_arg)->func_name();
              }
              return static_cast<const Item_func *>(condition)->func_name();
            }(),
            [&]() -> size_t {
              if (!condition || condition->type() != Item::COND_ITEM) {
                return 1;
              }
              Item_cond_and *and_cond = static_cast<Item_cond_and *>(condition);
              size_t predicate_count = 0;
              List_iterator<Item> it(*and_cond->argument_list());
              while (it++ != nullptr)
                ++predicate_count;
              return std::max<size_t>(1, predicate_count);
            }(),
            SemanticBatchFeedbackKey(thd, condition, m_tables.tables_bitmap(),
                                     num_rows_estimate)) {}

  bool Init() override;

  int Read() override;

  void SetNullRowFlag(bool is_null_row) override {
    m_source->SetNullRowFlag(is_null_row);
  }

  void StartPSIBatchMode() override { m_source->StartPSIBatchMode(); }
  void EndPSIBatchModeIfStarted() override {
    m_source->EndPSIBatchModeIfStarted();
  }
  void UnlockRow() override {
    // Rows are snapshots and the child may already have advanced. Forwarding
    // this call could unlock a different physical row.
  }

 private:
  unique_ptr_destroy_only<RowIterator> m_source;
  pack_rows::TableCollection m_tables;
  Item *m_condition;

  size_t m_row_size;
  std::queue<std::vector<uint8_t>> m_rows_queue;
  ViperFlow<std::string, uint8_t> m_buffer_manager;
  bool m_source_exhausted{false};
  bool m_final_batch_flushed{false};
  // A returned FIFO row can be older than the row at which the child paused.
  // The newest queued snapshot is that child frontier and must be restored
  // before the next child Read().
  bool m_source_frontier_overwritten{false};
};

#endif  // SQL_ITERATORS_GPU_ITERATORS_H_
