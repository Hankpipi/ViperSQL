#include "sql/iterators/vectorized_iterators.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <new>

#include "mysqld_error.h"
#include "sql/item_func_semantic.h"
#include "sql/pfs_batch_mode.h"
#include "sql/sql_optimizer.h"

namespace {

constexpr size_t kGpuReservedAuxiliaryHostBytes =
    GPUHashJoinIterator::kMinHostMemoryBudget;
constexpr size_t kPerBuildRowContainerOverhead = 128;
constexpr size_t kPerDistinctBuildGroupOverhead = 384;
constexpr size_t kPerProbeRowContainerOverhead = 256;

size_t SaturatingAdd(size_t left, size_t right) {
  if (right > std::numeric_limits<size_t>::max() - left) {
    return std::numeric_limits<size_t>::max();
  }
  return left + right;
}

size_t SaturatingMultiply(size_t left, size_t right) {
  if (left != 0 && right > std::numeric_limits<size_t>::max() / left) {
    return std::numeric_limits<size_t>::max();
  }
  return left * right;
}

void UpdateSemanticFeedbackHash(uint64_t *low, uint64_t *high, const char *data,
                                size_t length) {
  if (data == nullptr || length == 0)
    return;
  for (size_t index = 0; index < length; ++index) {
    const uint8_t byte = static_cast<uint8_t>(data[index]);
    *low ^= byte;
    *low *= 1099511628211ULL;
    *high ^= static_cast<uint8_t>(byte + 0x9dU);
    *high *= 14029467366897019727ULL;
  }
}

void ReportIteratorError(THD *thd, const char *operation) {
  // Every iterator failure must set a statement diagnostic.
  if (thd != nullptr && !thd->is_error()) {
    my_error(ER_INTERNAL_ERROR, MYF(0), operation);
  }
}

int IteratorReadError(THD *thd, const char *operation) {
  ReportIteratorError(thd, operation);
  return 1;
}

bool IteratorInitError(THD *thd, const char *operation) {
  ReportIteratorError(thd, operation);
  return true;
}

}  // namespace

std::string SemanticBatchFeedbackKey(THD *thd, const Item *condition,
                                     table_map input_tables,
                                     size_t estimated_rows) {
  if (thd == nullptr || condition == nullptr)
    return {};

  uint64_t low = 1469598103934665603ULL;
  uint64_t high = 0x9e3779b97f4a7c15ULL;
  const LEX_CSTRING database = thd->db();
  UpdateSemanticFeedbackHash(&low, &high, database.str, database.length);
  constexpr char kDatabaseSeparator[] = "\0statement\0";
  UpdateSemanticFeedbackHash(&low, &high, kDatabaseSeparator,
                             sizeof(kDatabaseSeparator));
  const LEX_CSTRING query = thd->query();
  UpdateSemanticFeedbackHash(&low, &high, query.str, query.length);

  constexpr char kSeparator[] = "\0semantic-stage\0";
  UpdateSemanticFeedbackHash(&low, &high, kSeparator, sizeof(kSeparator));
  String printed_condition;
  condition->print(thd, &printed_condition, QT_ORDINARY);
  UpdateSemanticFeedbackHash(&low, &high, printed_condition.ptr(),
                             printed_condition.length());

  constexpr char kPhysicalSeparator[] = "\0physical-input\0";
  UpdateSemanticFeedbackHash(&low, &high, kPhysicalSeparator,
                             sizeof(kPhysicalSeparator));
  for (size_t index = 0; index < sizeof(input_tables); ++index) {
    const char byte = static_cast<char>(
        (static_cast<uint64_t>(input_tables) >> (index * 8)) & 0xffU);
    UpdateSemanticFeedbackHash(&low, &high, &byte, 1);
  }
  size_t estimate_bucket = 0;
  for (size_t value = estimated_rows; value > 1; value >>= 1) {
    ++estimate_bucket;
  }
  for (size_t index = 0; index < sizeof(estimate_bucket); ++index) {
    const char byte = static_cast<char>(
        (static_cast<uint64_t>(estimate_bucket) >> (index * 8)) & 0xffU);
    UpdateSemanticFeedbackHash(&low, &high, &byte, 1);
  }

  std::array<char, 33> encoded{};
  std::snprintf(encoded.data(), encoded.size(), "%016llx%016llx",
                static_cast<unsigned long long>(low),
                static_cast<unsigned long long>(high));
  return std::string(encoded.data());
}

GPUHashJoinIterator::GPUHashJoinIterator(
    THD *thd, unique_ptr_destroy_only<RowIterator> build_input,
    const Prealloced_array<TABLE *, 4> &build_input_tables,
    double estimated_build_rows,
    unique_ptr_destroy_only<RowIterator> probe_input,
    const Prealloced_array<TABLE *, 4> &probe_input_tables, bool store_rowids,
    table_map tables_to_get_rowid_for, size_t max_memory_available,
    const std::vector<HashJoinCondition> &join_conditions,
    bool allow_spill_to_disk, JoinType join_type,
    const Mem_root_array<Item *> &extra_conditions, bool probe_input_batch_mode,
    uint64_t *hash_table_generation)
    : RowIterator(thd), m_build_input(std::move(build_input)),
      m_probe_input(std::move(probe_input)),
      m_build_input_table_array(build_input_tables),
      m_probe_input_table_array(probe_input_tables),
      m_build_input_tables(build_input_tables, store_rowids,
                           tables_to_get_rowid_for,
                           /*tables_to_store_contents_of_null_rows_for=*/0),
      m_probe_input_tables(probe_input_tables, store_rowids,
                           tables_to_get_rowid_for,
                           /*tables_to_store_contents_of_null_rows_for=*/0),
      m_tables_to_get_rowid_for(tables_to_get_rowid_for),
      m_store_rowids(store_rowids),
      m_join_conditions(PSI_NOT_INSTRUMENTED, join_conditions.data(),
                        join_conditions.data() + join_conditions.size()),
      m_join_type(join_type), m_allow_spill_to_disk(allow_spill_to_disk),
      m_estimated_build_rows(estimated_build_rows),
      m_probe_input_batch_mode(probe_input_batch_mode),
      m_hash_table_generation(hash_table_generation),
      m_max_memory_available(max_memory_available) {
  assert(m_build_input != nullptr);
  assert(m_probe_input != nullptr);

  EnsureGPUBufferManager();

  if (extra_conditions.size() == 1) {
    m_extra_condition = extra_conditions[0];
  } else if (extra_conditions.size() > 1) {
    List<Item> items;
    for (Item *cond : extra_conditions) {
      items.push_back(cond);
    }
    m_extra_condition = new Item_cond_and(items);
    m_extra_condition->quick_fix_field();
    m_extra_condition->update_used_tables();
    m_extra_condition->apply_is_true();
  }
}

void GPUHashJoinIterator::EnsureGPUBufferManager() {
  if (m_buffer_manager != nullptr ||
      !HasSufficientHostMemoryBudget(m_max_memory_available)) {
    return;
  }
  m_buffer_manager =
      std::make_unique<ViperFlow<gpuhashjoinhelpers::KeyIndexPair, uint32_t>>(
          m_max_memory_available, "GPUHashJoinHelper");
}

// Replays rows consumed while attempting a GPU build. A partial-prefix
// handoff then resumes the already-initialized child at exactly the next row;
// a completed-build handoff ends without a second Read() after observed EOF.
// The partial prefix includes the current child row, so replaying it last also
// restores all TABLE buffers a nested child may inspect when it advances.
class GPUHashJoinIterator::BuildPrefixIterator final : public RowIterator {
 public:
  BuildPrefixIterator(THD *thd, unique_ptr_destroy_only<RowIterator> child,
                      const pack_rows::TableCollection *tables,
                      std::vector<PackedRowRef> rows,
                      std::deque<std::unique_ptr<PackedRowBlock>> blocks,
                      bool child_already_exhausted)
      : RowIterator(thd), m_child(std::move(child)), m_tables(tables),
        m_rows(std::move(rows)), m_blocks(std::move(blocks)),
        m_child_already_exhausted(child_already_exhausted) {
    assert(m_child != nullptr);
    assert(m_tables != nullptr);
    assert(!m_rows.empty());
  }

  bool Init() override {
    ReleasePreviouslyReturnedRow();
    ReleaseResumedFrontier();
    m_next_row = 0;
    m_reading_child = false;

    if (!m_initialized_once) {
      // GPUHashJoinIterator already initialized and consumed the saved rows
      // from the child. Calling Init() here would rewind and duplicate them.
      m_initialized_once = true;
      return false;
    }

    // A later execution starts normally from the underlying child. The saved
    // prefix belonged only to the interrupted first execution.
    m_rows.clear();
    m_blocks.clear();
    m_reading_child = true;
    m_child_already_exhausted = false;
    return m_child->Init();
  }

  int Read() override {
    if (m_next_row < m_rows.size()) {
      ReleasePreviouslyReturnedRow();
      PackedRowRef &row = m_rows[m_next_row++];
      if (row.data == nullptr || row.block == nullptr)
        return IteratorReadError(thd(),
                                 "GPU hash join build replay row is invalid");
      LoadIntoTableBuffers(*m_tables, row.data);
      m_previously_returned_block = row.block;
      row = {};
      return 0;
    }

    if (m_child_already_exhausted) {
      // The completed-build handoff has already observed child EOF. Repeating
      // Read() after EOF is not part of RowIterator's contract; the replayed
      // prefix is the complete first execution.
      ReleasePreviouslyReturnedRow();
      m_rows.clear();
      return -1;
    }

    m_reading_child = true;
    // The last replayed row is the frontier at which the initialized child is
    // paused. A nested child can keep an outer row unchanged across many
    // successful Read() calls, so retain this block through the entire
    // resumed first execution, not merely through its first Read().
    if (m_resumed_frontier_block == nullptr) {
      m_resumed_frontier_block = m_previously_returned_block;
      m_previously_returned_block = nullptr;
    }
    const int result = m_child->Read();
    if (result != 0)
      ReleaseResumedFrontier();
    m_rows.clear();
    return result;
  }

  void SetNullRowFlag(bool is_null_row) override {
    m_child->SetNullRowFlag(is_null_row);
  }

  void StartPSIBatchMode() override { m_child->StartPSIBatchMode(); }

  void EndPSIBatchModeIfStarted() override {
    m_child->EndPSIBatchModeIfStarted();
  }

  void UnlockRow() override {
    if (m_reading_child)
      m_child->UnlockRow();
  }

 private:
  void ReleasePreviouslyReturnedRow() {
    if (m_previously_returned_block != nullptr) {
      assert(m_previously_returned_block->outstanding_refs > 0);
      --m_previously_returned_block->outstanding_refs;
      m_previously_returned_block = nullptr;
    }
    while (!m_blocks.empty() && m_blocks.front()->outstanding_refs == 0) {
      m_blocks.pop_front();
    }
  }

  void ReleaseResumedFrontier() {
    if (m_resumed_frontier_block != nullptr) {
      assert(m_resumed_frontier_block->outstanding_refs > 0);
      --m_resumed_frontier_block->outstanding_refs;
      m_resumed_frontier_block = nullptr;
    }
    while (!m_blocks.empty() && m_blocks.front()->outstanding_refs == 0) {
      m_blocks.pop_front();
    }
  }

  unique_ptr_destroy_only<RowIterator> m_child;
  const pack_rows::TableCollection *m_tables;
  std::vector<PackedRowRef> m_rows;
  std::deque<std::unique_ptr<PackedRowBlock>> m_blocks;
  size_t m_next_row{0};
  PackedRowBlock *m_previously_returned_block{nullptr};
  PackedRowBlock *m_resumed_frontier_block{nullptr};
  bool m_initialized_once{false};
  bool m_reading_child{false};
  bool m_child_already_exhausted{false};
};

size_t GPUHashJoinIterator::BuildRowMemoryCharge(bool creates_group) const {
  // std::vector growth, deque nodes and unordered-map buckets are
  // implementation details. Charge more than their observed element sizes so
  // the admission decision is conservative across supported standard-library
  // implementations. Packed row/key allocation is charged separately from
  // the actual retained arena-block capacities.
  size_t charge = kPerBuildRowContainerOverhead;
  if (creates_group) {
    charge = SaturatingAdd(charge, kPerDistinctBuildGroupOverhead);
  }
  return charge;
}

size_t GPUHashJoinIterator::PredictedArenaAllocation(
    const std::deque<std::unique_ptr<PackedRowBlock>> &blocks,
    size_t required_bytes) const {
  if (required_bytes == std::numeric_limits<size_t>::max()) {
    return required_bytes;
  }
  if (!blocks.empty() &&
      blocks.back()->bytes.size() - blocks.back()->used >= required_bytes) {
    return 0;
  }
  return std::max<size_t>(1, required_bytes);
}

size_t GPUHashJoinIterator::UnusedBuildContainerReserveCharge(
    size_t build_rows, size_t build_groups) const {
  // Unused vector capacity owns only element slots and the hash bucket array.
  // Charge retained capacity directly, using two pointers per hash bucket.
  const auto unused_slots = [](size_t capacity, size_t size) {
    return capacity > size ? capacity - size : 0;
  };

  size_t charge =
      SaturatingMultiply(unused_slots(build_rows_buffer.capacity(), build_rows),
                         sizeof(PackedRowRef));
  charge = SaturatingAdd(
      charge,
      SaturatingMultiply(unused_slots(m_next_build_row.capacity(), build_rows),
                         sizeof(size_t)));
  charge = SaturatingAdd(
      charge,
      SaturatingMultiply(unused_slots(m_build_groups.capacity(), build_groups),
                         sizeof(BuildKeyGroup)));
  charge = SaturatingAdd(
      charge, SaturatingMultiply(m_build_hash_to_group.bucket_count(),
                                 size_t{2} * sizeof(void *)));
  return charge;
}

bool GPUHashJoinIterator::BuildMemoryWouldOverflow(
    size_t additional_metadata_charge, size_t row_required_bytes,
    size_t key_required_bytes, bool creates_group) const {
  const size_t projected_build_rows =
      SaturatingAdd(build_rows_buffer.size(), size_t{1});
  const size_t projected_build_groups = SaturatingAdd(
      m_build_groups.size(), creates_group ? size_t{1} : size_t{0});
  size_t used = SaturatingAdd(m_build_host_memory_charge,
                              m_build_row_arena_memory_charge);
  used = SaturatingAdd(used, m_build_key_arena_memory_charge);
  used = SaturatingAdd(used, UnusedBuildContainerReserveCharge(
                                 projected_build_rows, projected_build_groups));
  used = SaturatingAdd(
      used, PredictedArenaAllocation(m_build_row_blocks, row_required_bytes));
  if (creates_group) {
    used = SaturatingAdd(
        used, PredictedArenaAllocation(m_build_key_blocks, key_required_bytes));
  }
  used = std::min(used, m_build_host_memory_limit);
  return additional_metadata_charge > m_build_host_memory_limit - used;
}

bool GPUHashJoinIterator::RetainedBuildMemoryExceedsLimit() const {
  size_t used = SaturatingAdd(m_build_host_memory_charge,
                              m_build_row_arena_memory_charge);
  used = SaturatingAdd(used, m_build_key_arena_memory_charge);
  used =
      SaturatingAdd(used, UnusedBuildContainerReserveCharge(
                              build_rows_buffer.size(), m_build_groups.size()));
  return used > m_build_host_memory_limit;
}

bool GPUHashJoinIterator::ProbeMemoryWouldOverflow(
    size_t additional_metadata_charge, size_t arena_required_bytes) const {
  const size_t variable_limit =
      m_max_memory_available > kGpuReservedAuxiliaryHostBytes
          ? m_max_memory_available - kGpuReservedAuxiliaryHostBytes
          : 0;
  size_t used = SaturatingAdd(m_build_host_memory_charge,
                              m_build_row_arena_memory_charge);
  used = SaturatingAdd(used, m_build_key_arena_memory_charge);
  used =
      SaturatingAdd(used, UnusedBuildContainerReserveCharge(
                              build_rows_buffer.size(), m_build_groups.size()));
  used = SaturatingAdd(used, m_probe_host_memory_charge);
  used = SaturatingAdd(used, m_probe_arena_memory_charge);
  if (arena_required_bytes != 0) {
    used = SaturatingAdd(used, PredictedArenaAllocation(m_probe_row_blocks,
                                                        arena_required_bytes));
  }
  used = std::min(used, variable_limit);
  return additional_metadata_charge > variable_limit - used;
}

bool GPUHashJoinIterator::RetainedProbeMemoryExceedsLimit() const {
  const size_t variable_limit =
      m_max_memory_available > kGpuReservedAuxiliaryHostBytes
          ? m_max_memory_available - kGpuReservedAuxiliaryHostBytes
          : 0;
  size_t used = SaturatingAdd(m_build_host_memory_charge,
                              m_build_row_arena_memory_charge);
  used = SaturatingAdd(used, m_build_key_arena_memory_charge);
  used =
      SaturatingAdd(used, UnusedBuildContainerReserveCharge(
                              build_rows_buffer.size(), m_build_groups.size()));
  used = SaturatingAdd(used, m_probe_host_memory_charge);
  used = SaturatingAdd(used, m_probe_arena_memory_charge);
  return used > variable_limit;
}

bool GPUHashJoinIterator::InitProbeInput() {
  PrepareForRequestRowId(m_probe_input_tables.tables(),
                         m_tables_to_get_rowid_for);
  if (m_probe_input->Init()) {
    return true;
  }

  m_row_size = ComputeRowSizeUpperBound(m_probe_input_tables);
  if (m_probe_input_batch_mode) {
    m_probe_input->StartPSIBatchMode();
    m_probe_batch_mode_started = true;
  }
  return false;
}

bool GPUHashJoinIterator::InitNativeFallback(
    bool build_child_already_initialized, bool build_child_already_exhausted) {
  if (m_native_fallback != nullptr)
    return m_native_fallback->Init();

  assert(!build_child_already_exhausted || build_child_already_initialized);
  m_use_gpu = false;
  m_buffer_manager.reset();

  unique_ptr_destroy_only<RowIterator> native_build_input;
  if (build_child_already_initialized) {
    assert(m_build_input != nullptr);
    assert(!build_rows_buffer.empty());
    native_build_input.reset(new (thd()->mem_root) BuildPrefixIterator(
        thd(), std::move(m_build_input), &m_build_input_tables,
        std::move(build_rows_buffer), std::move(m_build_row_blocks),
        build_child_already_exhausted));
  } else {
    native_build_input = std::move(m_build_input);
  }

  m_build_key_blocks.clear();
  // Release metadata that the native iterator cannot use.
  decltype(m_build_hash_to_group){}.swap(m_build_hash_to_group);
  decltype(m_build_groups){}.swap(m_build_groups);
  decltype(m_next_build_row){}.swap(m_next_build_row);
  assert(m_probe_rows.empty());
  ReleaseRetainedProbeFrontier();
  m_probe_row_blocks.clear();
  ClearProbeRows();
  m_build_host_memory_charge = 0;
  m_build_row_arena_memory_charge = 0;
  m_build_key_arena_memory_charge = 0;
  m_probe_host_memory_charge = 0;
  m_probe_arena_memory_charge = 0;

  std::vector<HashJoinCondition> join_conditions(m_join_conditions.begin(),
                                                 m_join_conditions.end());
  Mem_root_array<Item *> extra_conditions(thd()->mem_root);
  if (m_extra_condition != nullptr)
    extra_conditions.push_back(m_extra_condition);

  m_native_fallback.reset(new (thd()->mem_root) HashJoinIterator(
      thd(), std::move(native_build_input), m_build_input_table_array,
      m_estimated_build_rows, std::move(m_probe_input),
      m_probe_input_table_array, m_store_rowids, m_tables_to_get_rowid_for,
      m_max_memory_available, join_conditions, m_allow_spill_to_disk,
      m_join_type, extra_conditions, m_probe_input_batch_mode,
      m_hash_table_generation));
  return m_native_fallback->Init();
}

// Returns extracted raw row buffer or empty vector on failure
std::vector<uint8_t>
store_row_to_buffer(const pack_rows::TableCollection &tables, size_t row_size) {
  size_t row_size_upper_bound = row_size;
  if (tables.has_blob_column()) {
    row_size_upper_bound = ComputeRowSizeUpperBound(tables);
  }

  std::vector<uint8_t> row_buffer(row_size_upper_bound);

  // Copy raw row bytes from table buffers into row_buffer
  uchar *dest = row_buffer.data();
  dest = StoreFromTableBuffersRaw(tables, dest);

  if (dest == nullptr) {
    // Copy failed (e.g. OOM), return empty vector
    return std::vector<uint8_t>();
  }

  // Resize to actual copied size
  size_t actual_size = dest - row_buffer.data();
  row_buffer.resize(actual_size);

  return row_buffer;
}

GPUHashJoinIterator::JoinKeyStatus
GPUHashJoinIterator::extract_join_key_for_row(
    THD *thd, const pack_rows::TableCollection &tables) {
  m_buffer.length(0);
  for (const HashJoinCondition &cond : m_join_conditions) {
    bool null_found = cond.join_condition()->append_join_key_for_hash_join(
        thd, tables.tables_bitmap(), cond, m_join_conditions.size() > 1,
        &m_buffer);
    if (thd->is_error()) {
      return JoinKeyStatus::kError;
    }
    if (null_found) {
      return JoinKeyStatus::kNull;
    }
  }
  return JoinKeyStatus::kOk;
}

static void MarkGPUHashJoinCopyBlobsIfGeometry(
    const pack_rows::TableCollection &table_collection) {
  for (const pack_rows::Table &table : table_collection.tables()) {
    for (const pack_rows::Column &column : table.columns) {
      if (column.field_type == MYSQL_TYPE_GEOMETRY) {
        table.table->copy_blobs = true;
        break;
      }
    }
  }
}

GPUHashJoinIterator::PackedRowRef GPUHashJoinIterator::StoreRowInArena(
    const pack_rows::TableCollection &tables, size_t row_size,
    std::deque<std::unique_ptr<PackedRowBlock>> *blocks) {
  if (blocks == nullptr)
    return {};

  size_t row_size_upper_bound = row_size;
  if (tables.has_blob_column()) {
    row_size_upper_bound = ComputeRowSizeUpperBound(tables);
  }

  if (blocks->empty() || blocks->back()->bytes.size() - blocks->back()->used <
                             row_size_upper_bound) {
    auto block = std::make_unique<PackedRowBlock>();
    block->bytes.resize(std::max<size_t>(1, row_size_upper_bound));
    if (blocks == &m_build_row_blocks) {
      m_build_row_arena_memory_charge = SaturatingAdd(
          m_build_row_arena_memory_charge, block->bytes.capacity());
    }
    blocks->push_back(std::move(block));
  }

  PackedRowBlock *block = blocks->back().get();
  uchar *const row_begin = block->bytes.data() + block->used;
  uchar *const row_end = StoreFromTableBuffersRaw(tables, row_begin);
  if (row_end == nullptr)
    return {};

  const size_t actual_size = static_cast<size_t>(row_end - row_begin);
  if (actual_size > row_size_upper_bound ||
      actual_size > block->bytes.size() - block->used) {
    return {};
  }
  block->used += actual_size;
  ++block->outstanding_refs;
  return {row_begin, actual_size, block};
}

GPUHashJoinIterator::PackedRowRef
GPUHashJoinIterator::StoreProbeRowAndKeyInArena(
    const pack_rows::TableCollection &tables, size_t row_size, const char *key,
    size_t key_length) {
  if (key_length != 0 && key == nullptr)
    return {};

  size_t row_size_upper_bound = row_size;
  if (tables.has_blob_column()) {
    row_size_upper_bound = ComputeRowSizeUpperBound(tables);
  }
  const size_t required = SaturatingAdd(row_size_upper_bound, key_length);
  if (required == std::numeric_limits<size_t>::max())
    return {};

  if (m_probe_row_blocks.empty() || m_probe_row_blocks.back()->bytes.size() -
                                            m_probe_row_blocks.back()->used <
                                        required) {
    auto block = std::make_unique<PackedRowBlock>();
    block->bytes.resize(std::max<size_t>(1, required));
    m_probe_arena_memory_charge =
        SaturatingAdd(m_probe_arena_memory_charge, block->bytes.capacity());
    m_probe_row_blocks.push_back(std::move(block));
  }

  PackedRowBlock *const block = m_probe_row_blocks.back().get();
  uchar *const row_begin = block->bytes.data() + block->used;
  uchar *const row_end = StoreFromTableBuffersRaw(tables, row_begin);
  if (row_end == nullptr)
    return {};
  const size_t actual_size = static_cast<size_t>(row_end - row_begin);
  if (actual_size > row_size_upper_bound ||
      key_length > block->bytes.size() - block->used - actual_size) {
    return {};
  }
  if (key_length != 0)
    std::memcpy(row_end, key, key_length);
  block->used += actual_size + key_length;
  ++block->outstanding_refs;
  return {row_begin, actual_size, block};
}

GPUHashJoinIterator::PackedRowRef GPUHashJoinIterator::StoreBytesInArena(
    const char *data, size_t length,
    std::deque<std::unique_ptr<PackedRowBlock>> *blocks) {
  if (blocks == nullptr || (length != 0 && data == nullptr))
    return {};

  if (blocks->empty() ||
      blocks->back()->bytes.size() - blocks->back()->used < length) {
    auto block = std::make_unique<PackedRowBlock>();
    block->bytes.resize(std::max<size_t>(1, length));
    if (blocks == &m_build_key_blocks) {
      m_build_key_arena_memory_charge = SaturatingAdd(
          m_build_key_arena_memory_charge, block->bytes.capacity());
    }
    blocks->push_back(std::move(block));
  }

  PackedRowBlock *block = blocks->back().get();
  uint8_t *const destination = block->bytes.data() + block->used;
  if (length != 0)
    std::memcpy(destination, data, length);
  block->used += length;
  ++block->outstanding_refs;
  return {destination, length, block};
}

void GPUHashJoinIterator::ClearProbeRows() {
  while (!m_probe_rows.empty())
    m_probe_rows.pop();
}

void GPUHashJoinIterator::PopFrontProbeRow() {
  assert(!m_probe_rows.empty());

  if (!m_probe_input_exhausted && m_probe_table_buffer_owner != nullptr &&
      m_probe_table_buffer_owner == m_probe_rows.front().packed_row.data) {
    RetainFrontProbeRowUntilChildRead();
    return;
  }

  const ProbeRow &probe = m_probe_rows.front();
  if (m_probe_table_buffer_owner == probe.packed_row.data) {
    m_probe_table_buffer_owner = nullptr;
  }
  PackedRowBlock *const row_block = probe.packed_row.block;
  if (row_block != nullptr) {
    assert(row_block->outstanding_refs > 0);
    --row_block->outstanding_refs;
  }
  assert(probe.memory_charge <= m_probe_host_memory_charge);
  m_probe_host_memory_charge -= probe.memory_charge;
  m_probe_rows.pop();
  ReclaimUnusedProbeBlocks();
}

void GPUHashJoinIterator::RetainFrontProbeRowUntilChildRead() {
  assert(!m_probe_rows.empty());
  assert(!m_has_retained_probe_frontier);
  m_retained_probe_frontier = m_probe_rows.front();
  m_probe_rows.pop();
  m_has_retained_probe_frontier = true;
}

void GPUHashJoinIterator::ReleaseRetainedProbeFrontier() {
  if (!m_has_retained_probe_frontier)
    return;

  PackedRowBlock *const row_block = m_retained_probe_frontier.packed_row.block;
  if (row_block != nullptr) {
    assert(row_block->outstanding_refs > 0);
    --row_block->outstanding_refs;
  }
  assert(m_retained_probe_frontier.memory_charge <= m_probe_host_memory_charge);
  m_probe_host_memory_charge -= m_retained_probe_frontier.memory_charge;
  if (m_probe_table_buffer_owner == m_retained_probe_frontier.packed_row.data) {
    m_probe_table_buffer_owner = nullptr;
  }
  m_retained_probe_frontier = {};
  m_has_retained_probe_frontier = false;
  ReclaimUnusedProbeBlocks();
}

void GPUHashJoinIterator::RestoreProbeInputFrontierIfOverwritten() {
  if (!m_probe_frontier_overwritten)
    return;
  assert(!m_probe_input_exhausted);
  assert(!m_probe_rows.empty());

  // The FIFO tail is the most recently produced probe row and therefore the
  // frontier at which a nested child is paused. CPU fallback can drain and
  // pop older FIFO entries before reaching the normal pre-child restoration,
  // so restore the tail at both boundaries through this common helper.
  const uint8_t *const tail_row = m_probe_rows.back().packed_row.data;
  LoadIntoTableBuffers(m_probe_input_tables, tail_row);
  m_probe_table_buffer_owner = tail_row;
  ReleaseRetainedProbeFrontier();
  m_probe_frontier_overwritten = false;
}

void GPUHashJoinIterator::ReclaimUnusedProbeBlocks() {
  while (!m_probe_row_blocks.empty() &&
         m_probe_row_blocks.front()->outstanding_refs == 0) {
    const size_t block_capacity = m_probe_row_blocks.front()->bytes.capacity();
    assert(block_capacity <= m_probe_arena_memory_charge);
    m_probe_arena_memory_charge -= block_capacity;
    m_probe_row_blocks.pop_front();
  }
}

void GPUHashJoinIterator::ActivateBuildGroup(size_t group_index) {
  assert(group_index < m_build_groups.size());
  assert(!m_probe_rows.empty());
  m_active_build_row = m_build_groups[group_index].first_row;
}

size_t GPUHashJoinIterator::FindExactBuildGroup(
    const char *key, size_t key_length,
    const gpuhashjoinhelpers::HashKey &hash_key) const {
  const auto head = m_build_hash_to_group.find(hash_key);
  if (head == m_build_hash_to_group.end())
    return kNoBuildRow;

  return FindExactBuildGroupInCollisionChain(key, key_length, head->second);
}

size_t GPUHashJoinIterator::FindExactBuildGroupInCollisionChain(
    const char *key, size_t key_length, size_t group_index) const {
  while (group_index != kNoBuildRow) {
    if (group_index >= m_build_groups.size())
      return kNoBuildRow;
    const BuildKeyGroup &group = m_build_groups[group_index];
    if (group.key.size == key_length &&
        (key_length == 0 ||
         std::memcmp(group.key.data, key, key_length) == 0)) {
      return group_index;
    }
    group_index = group.next_hash_collision;
  }
  return kNoBuildRow;
}

void GPUHashJoinIterator::DisableGPU() {
  if (!m_use_gpu)
    return;

  m_use_gpu = false;
  // Host exact-key state is complete, so no GPU state is needed after a
  // helper failure or probe-memory cutoff.
  m_buffer_manager.reset();
}

int GPUHashJoinIterator::ReadNextActiveJoinedRow() {
  if (m_probe_rows.empty())
    return IteratorReadError(thd(), "GPU hash join probe queue is empty");

  while (m_active_build_row != kNoBuildRow) {
    if (m_active_build_row >= build_rows_buffer.size() ||
        m_active_build_row >= m_next_build_row.size()) {
      return IteratorReadError(thd(),
                               "GPU hash join build row index is invalid");
    }

    const size_t build_row = m_active_build_row;
    m_active_build_row = m_next_build_row[build_row];
    LoadIntoTableBuffers(m_build_input_tables,
                         build_rows_buffer[build_row].data);
    const uint8_t *const probe_row = m_probe_rows.front().packed_row.data;
    LoadIntoTableBuffers(m_probe_input_tables, probe_row);
    m_probe_table_buffer_owner = probe_row;
    // The complete replacement snapshot is installed; pointers into a prior
    // retained frontier are no longer visible through TABLE buffers.
    ReleaseRetainedProbeFrontier();
    m_probe_frontier_overwritten =
        !m_probe_input_exhausted && m_probe_rows.size() > 1;

    const bool passes_extra_condition =
        m_extra_condition == nullptr || m_extra_condition->val_int() != 0;
    if (thd()->is_error())
      return 1;
    if (thd()->killed) {
      thd()->send_kill_message();
      return 1;
    }

    if (!passes_extra_condition) {
      if (m_active_build_row == kNoBuildRow) {
        PopFrontProbeRow();
      }
      continue;
    }

    // Keep the packed probe row alive until the caller asks for the next row.
    // Blob/geometry fields may refer into this storage after restoration.
    if (m_active_build_row == kNoBuildRow) {
      m_release_probe_row_on_next_read = true;
    }
    return 0;
  }

  return -1;
}

bool GPUHashJoinIterator::Init() {
  if (m_native_fallback != nullptr) {
    return m_native_fallback->Init();
  }

  m_use_gpu = true;
  build_rows_buffer.clear();
  m_build_row_blocks.clear();
  m_build_key_blocks.clear();
  m_build_hash_to_group.clear();
  m_build_groups.clear();
  m_next_build_row.clear();
  m_build_host_memory_charge = 0;
  m_build_row_arena_memory_charge = 0;
  m_build_key_arena_memory_charge = 0;
  ClearProbeRows();
  m_probe_row_blocks.clear();
  m_retained_probe_frontier = {};
  m_has_retained_probe_frontier = false;
  m_probe_table_buffer_owner = nullptr;
  m_probe_input_exhausted = false;
  m_probe_batch_flushed = false;
  m_probe_batch_mode_started = false;
  m_probe_frontier_overwritten = false;
  m_join_result_empty = false;
  m_release_probe_row_on_next_read = false;
  m_active_build_row = kNoBuildRow;
  m_probe_host_memory_charge = 0;
  m_probe_arena_memory_charge = 0;

  const size_t variable_host_budget =
      m_max_memory_available > kGpuReservedAuxiliaryHostBytes
          ? m_max_memory_available - kGpuReservedAuxiliaryHostBytes
          : 0;
  // Leave one third of variable host memory for queued probe snapshots. A
  // build that crosses this line is transferred to the native spill path.
  m_build_host_memory_limit = variable_host_budget - variable_host_budget / 3;

  if (m_join_type != JoinType::INNER) {
    assert(false);
    return IteratorInitError(thd(), "GPU hash join type is unsupported");
  }

  EnsureGPUBufferManager();
  if (m_buffer_manager == nullptr) {
    return InitNativeFallback(/*build_child_already_initialized=*/false,
                              /*build_child_already_exhausted=*/false);
  }
  if (m_buffer_manager->Reset()) {
    DisableGPU();
  }

  MarkGPUHashJoinCopyBlobsIfGeometry(m_build_input_tables);
  MarkGPUHashJoinCopyBlobsIfGeometry(m_probe_input_tables);

  PrepareForRequestRowId(m_build_input_tables.tables(),
                         m_tables_to_get_rowid_for);
  if (m_build_input->Init()) {
    return true;
  }
  m_probe_input->EndPSIBatchModeIfStarted();

  if (m_use_gpu)
    m_buffer_manager->SetStatus("BUILD");
  m_row_size = ComputeRowSizeUpperBound(m_build_input_tables);
  m_build_input->SetNullRowFlag(/*is_null_row=*/false);
  bool hand_off_build_prefix = false;
  {
    PFSBatchMode batch_mode(m_build_input.get());
    while (true) {
      int ret = m_build_input->Read();
      if (ret == 1) {  // error
        return true;
      }
      thd()->check_yield();
      if (ret == -1) {  // EOF
        break;
      }
      assert(ret == 0);
      RequestRowId(m_build_input_tables.tables(), m_tables_to_get_rowid_for);

      gpuhashjoinhelpers::HashKey hash_key;
      const JoinKeyStatus key_status =
          extract_join_key_for_row(thd(), m_build_input_tables);
      if (key_status == JoinKeyStatus::kError)
        return true;
      if (key_status == JoinKeyStatus::kNull) {
        // The GPU iterator supports only inner joins, for which a NULL build
        // key can never match. Omitting it from a later native prefix replay
        // is therefore equivalent to native HashJoinIterator::StoreRow().
        continue;
      }
      const char *const key_data = m_buffer.ptr();
      const size_t key_length = m_buffer.length();
      hash_key = gpuhashjoinhelpers::MakeHashKey(key_data, key_length);

      size_t row_size_upper_bound = m_row_size;
      if (m_build_input_tables.has_blob_column()) {
        row_size_upper_bound = ComputeRowSizeUpperBound(m_build_input_tables);
      }
      // Retain the fingerprint-chain head through group insertion.
      const auto hash_position = m_build_hash_to_group.find(hash_key);
      const size_t existing_group =
          hash_position == m_build_hash_to_group.end()
              ? kNoBuildRow
              : FindExactBuildGroupInCollisionChain(key_data, key_length,
                                                    hash_position->second);
      const size_t memory_charge =
          BuildRowMemoryCharge(/*creates_group=*/existing_group == kNoBuildRow);

      if (BuildMemoryWouldOverflow(
              memory_charge, row_size_upper_bound, key_length,
              /*creates_group=*/existing_group == kNoBuildRow)) {
        // Snapshot the current frontier as the final replay row. This may
        // exceed the conservative envelope by one row, matching the native
        // hash buffer's guarantee that it can admit at least one row before
        // spilling. No GPU index state is allocated for this row.
        const PackedRowRef current_row = StoreRowInArena(
            m_build_input_tables, m_row_size, &m_build_row_blocks);
        if (current_row.data == nullptr)
          return IteratorInitError(thd(),
                                   "GPU hash join build row buffering failed");
        build_rows_buffer.push_back(current_row);
        hand_off_build_prefix = true;
        break;
      }

      const PackedRowRef row_buf = StoreRowInArena(
          m_build_input_tables, m_row_size, &m_build_row_blocks);
      if (row_buf.data == nullptr) {
        return IteratorInitError(thd(),
                                 "GPU hash join build row buffering failed");
      }

      const size_t build_row_index = build_rows_buffer.size();
      build_rows_buffer.push_back(row_buf);
      m_next_build_row.push_back(kNoBuildRow);
      m_build_host_memory_charge =
          SaturatingAdd(m_build_host_memory_charge, memory_charge);

      const size_t next_group_index = m_build_groups.size();
      if (existing_group != kNoBuildRow) {
        BuildKeyGroup &group = m_build_groups[existing_group];
        m_next_build_row[group.last_row] = build_row_index;
        group.last_row = build_row_index;
        continue;
      }

      size_t previous_hash_head = kNoBuildRow;
      if (hash_position != m_build_hash_to_group.end()) {
        previous_hash_head = hash_position->second;
        hash_position->second = next_group_index;
      } else {
        m_build_hash_to_group.emplace_hint(hash_position, hash_key,
                                           next_group_index);
      }
      const PackedRowRef stored_key =
          StoreBytesInArena(key_data, key_length, &m_build_key_blocks);
      if (stored_key.data == nullptr)
        return IteratorInitError(thd(),
                                 "GPU hash join build key buffering failed");
      m_build_groups.push_back(BuildKeyGroup{
          stored_key, build_row_index, build_row_index, previous_hash_head});

      if (m_use_gpu) {
        if (next_group_index >= gpuhashjoinhelpers::kNotFound) {
          DisableGPU();
        } else {
          gpuhashjoinhelpers::KeyIndexPair pair{
              hash_key, static_cast<uint32_t>(next_group_index)};
          if (m_buffer_manager->PushTuple(pair)) {
            DisableGPU();
          }
        }
      }
    }
  }

  if (hand_off_build_prefix) {
    return InitNativeFallback(/*build_child_already_initialized=*/true,
                              /*build_child_already_exhausted=*/false);
  }

  // A final insertion can grow retained container capacity after the last
  // admission check. Recheck before continuing with the GPU build.
  if (RetainedBuildMemoryExceedsLimit()) {
    return InitNativeFallback(/*build_child_already_initialized=*/true,
                              /*build_child_already_exhausted=*/true);
  }

  // An inner join with an empty build cannot produce a row.
  if (build_rows_buffer.empty()) {
    m_join_result_empty = true;
    return false;
  }

  // Flush and synchronize the host-to-helper build metadata.  If CUDA is not
  // available, the exact host key map remains a complete CPU implementation.
  if (m_use_gpu) {
    if (m_buffer_manager->FlushBatch()) {
      DisableGPU();
    } else {
      (void)m_buffer_manager->PopResult();
      if (m_buffer_manager->HasError()) {
        DisableGPU();
      }
    }
  }

  if (m_use_gpu)
    m_buffer_manager->SetStatus("PROBE");

  return InitProbeInput();
}

int GPUHashJoinIterator::Read() {
  if (m_native_fallback != nullptr)
    return m_native_fallback->Read();

  if (thd()->killed) {
    thd()->send_kill_message();
    return 1;
  }

  if (m_join_result_empty)
    return -1;

  if (m_release_probe_row_on_next_read) {
    if (m_probe_rows.empty())
      return IteratorReadError(thd(),
                               "GPU hash join probe queue is misaligned");
    PopFrontProbeRow();
    m_release_probe_row_on_next_read = false;
  }

  // Allow at most one required refill submission per public Read() call.
  bool must_submit_before_pop = m_use_gpu && !m_probe_input_exhausted &&
                                m_buffer_manager->ShouldRefillBeforePop();
  bool submitted_before_pop = false;

  for (;;) {
    if (m_active_build_row != kNoBuildRow) {
      const int result = ReadNextActiveJoinedRow();
      if (result >= 0)
        return result;
    }

    // A CUDA failure after consuming either child is recoverable because the
    // build groups and queued probe snapshots are exact host-side state.
    if (!m_use_gpu && !m_probe_rows.empty()) {
      RestoreProbeInputFrontierIfOverwritten();
      const ProbeRow &probe = m_probe_rows.front();
      const char *const exact_key = reinterpret_cast<const char *>(
          probe.packed_row.data + probe.packed_row.size);
      const size_t group_index =
          FindExactBuildGroup(exact_key, probe.key_length, probe.hash_key);
      if (group_index == kNoBuildRow) {
        PopFrontProbeRow();
        continue;
      }
      ActivateBuildGroup(group_index);
      continue;
    }

    // Fill the helper before consuming queued results when more input is
    // available.
    if (!m_probe_input_exhausted) {
      if (m_probe_frontier_overwritten && m_probe_rows.empty())
        return IteratorReadError(thd(),
                                 "GPU hash join probe frontier is invalid");
      RestoreProbeInputFrontierIfOverwritten();
      const int ret = m_probe_input->Read();
      if (ret == 1) {
        ReleaseRetainedProbeFrontier();
        return 1;  // error
      }
      thd()->check_yield();

      if (ret == -1) {
        ReleaseRetainedProbeFrontier();
        m_probe_input_exhausted = true;
        if (m_probe_batch_mode_started) {
          m_probe_input->EndPSIBatchModeIfStarted();
          m_probe_batch_mode_started = false;
        }
      } else {
        assert(ret == 0);
        RequestRowId(m_probe_input_tables.tables(), m_tables_to_get_rowid_for);
        const JoinKeyStatus key_status =
            extract_join_key_for_row(thd(), m_probe_input_tables);
        if (key_status == JoinKeyStatus::kError)
          return 1;
        if (key_status == JoinKeyStatus::kNull) {
          // A NULL probe key cannot match an inner join.
          continue;
        }
        const char *const key_data = m_buffer.ptr();
        const size_t key_length = m_buffer.length();
        const gpuhashjoinhelpers::HashKey hash_key =
            gpuhashjoinhelpers::MakeHashKey(key_data, key_length);

        size_t row_size_upper_bound = m_row_size;
        if (m_probe_input_tables.has_blob_column()) {
          row_size_upper_bound = ComputeRowSizeUpperBound(m_probe_input_tables);
        }
        const size_t memory_charge = kPerProbeRowContainerOverhead;
        const size_t probe_arena_required =
            SaturatingAdd(row_size_upper_bound, key_length);
        if (m_use_gpu &&
            ProbeMemoryWouldOverflow(memory_charge, probe_arena_required)) {
          DisableGPU();
        }

        const PackedRowRef probe_row_buf = StoreProbeRowAndKeyInArena(
            m_probe_input_tables, m_row_size, key_data, key_length);
        if (probe_row_buf.data == nullptr) {
          return IteratorReadError(thd(),
                                   "GPU hash join probe row buffering failed");
        }
        m_probe_rows.push(
            ProbeRow{hash_key, probe_row_buf, key_length, memory_charge});
        m_probe_host_memory_charge =
            SaturatingAdd(m_probe_host_memory_charge, memory_charge);
        if (m_use_gpu && RetainedProbeMemoryExceedsLimit()) {
          DisableGPU();
        }
        if (m_probe_table_buffer_owner != nullptr ||
            m_has_retained_probe_frontier) {
          // A nested child may have carried outer-table BLOB pointers from an
          // arena-backed frontier into this new row. Canonicalize the complete
          // new frontier before releasing the old arena.
          LoadIntoTableBuffers(m_probe_input_tables, probe_row_buf.data);
          m_probe_table_buffer_owner = probe_row_buf.data;
        }
        ReleaseRetainedProbeFrontier();

        if (m_use_gpu) {
          gpuhashjoinhelpers::KeyIndexPair pair{hash_key, 0};
          bool did_submit = false;
          if (m_buffer_manager->PushTuple(pair, &did_submit)) {
            DisableGPU();
            m_probe_batch_flushed = true;
            continue;
          }
          if (did_submit) {
            submitted_before_pop = true;
          }
        }
      }
    }

    if (m_use_gpu && m_probe_input_exhausted && !m_probe_batch_flushed) {
      // Flush the residual probe batch once at EOF.
      if (m_buffer_manager->FlushBatch()) {
        DisableGPU();
        m_probe_batch_flushed = true;
        continue;
      }
      m_probe_batch_flushed = true;
    }

    if (!m_use_gpu) {
      if (m_probe_input_exhausted && m_probe_rows.empty())
        return -1;
      continue;
    }

    // Refill an idle helper while its output queue is below the watermark.
    if (!m_probe_input_exhausted && !submitted_before_pop &&
        m_buffer_manager->ShouldRefillBeforePop()) {
      must_submit_before_pop = true;
    }
    if (!m_probe_input_exhausted && must_submit_before_pop &&
        !submitted_before_pop && m_buffer_manager->ShouldRefillBeforePop()) {
      continue;
    }

    uint32_t matched_build_idx = gpuhashjoinhelpers::kNotFound;
    bool has_matched_build_idx = false;
    if (m_buffer_manager->HasReadyResult()) {
      has_matched_build_idx = m_buffer_manager->PopResult(&matched_build_idx);
    } else if (m_probe_input_exhausted) {
      // With no more relational work available, drain the final request.
      has_matched_build_idx = m_buffer_manager->PopResult(&matched_build_idx);
    } else {
      continue;
    }

    if (!has_matched_build_idx) {
      if (m_buffer_manager->HasError()) {
        DisableGPU();
        continue;
      }
      if (m_probe_input_exhausted && !m_buffer_manager->HasPendingWork()) {
        if (!m_probe_rows.empty()) {
          DisableGPU();
          continue;
        }
        return -1;  // no more rows
      }
      continue;
    }

    // Skip a probe with no matching build group.
    if (matched_build_idx == gpuhashjoinhelpers::kNotFound) {
      if (m_probe_rows.empty()) {
        return IteratorReadError(thd(),
                                 "GPU hash join result queue is misaligned");
      }
      PopFrontProbeRow();
      continue;
    }

    if (m_probe_rows.empty()) {
      return IteratorReadError(thd(),
                               "GPU hash join result queue is misaligned");
    }

    size_t group_index = matched_build_idx;
    const ProbeRow &probe = m_probe_rows.front();

    // The candidate check does not inspect TABLE buffers. The active-group
    // reader installs the probe snapshot once when it emits the joined row.
    const char *const exact_key = reinterpret_cast<const char *>(
        probe.packed_row.data + probe.packed_row.size);
    if (group_index >= m_build_groups.size() ||
        m_build_groups[group_index].key.size != probe.key_length ||
        (probe.key_length != 0 &&
         std::memcmp(m_build_groups[group_index].key.data, exact_key,
                     probe.key_length) != 0)) {
      // Resolve fingerprint collisions using the exact serialized key.
      const size_t exact_group =
          FindExactBuildGroup(exact_key, probe.key_length, probe.hash_key);
      if (exact_group == kNoBuildRow) {
        PopFrontProbeRow();
        continue;
      }
      group_index = exact_group;
    }
    ActivateBuildGroup(group_index);
  }
}

bool VectorizedFilterIterator::Init() {
  if (m_buffer_manager.Reset())
    return IteratorInitError(thd(), "semantic filter buffer reset failed");
  while (!m_rows_queue.empty())
    m_rows_queue.pop();
  m_source_exhausted = false;
  m_final_batch_flushed = false;
  m_source_frontier_overwritten = false;
  m_row_size = ComputeRowSizeUpperBound(m_tables);
  return m_source->Init();
}

int VectorizedFilterIterator::Read() {
  // This latch covers one public Read() call. Rejected predicates remain part
  // of the same call and therefore cannot demand another pre-pop submission.
  bool must_submit_before_pop =
      !m_source_exhausted && m_buffer_manager.ShouldRefillBeforePop();
  bool submitted_before_pop = false;
  const bool measure_producer_time =
      m_buffer_manager.ShouldMeasureProducerTime();

  for (;;) {
    if (!m_source_exhausted) {
      if (m_source_frontier_overwritten) {
        if (m_rows_queue.empty())
          return IteratorReadError(thd(),
                                   "semantic filter row frontier is invalid");
        // Fill-before-pop may have returned an older FIFO row while the child
        // was paused on a newer row. Nested-loop and EQ_REF children consult
        // their TABLE buffers when resumed, so restore the FIFO tail first.
        LoadIntoTableBuffers(m_tables, m_rows_queue.back().data());
        m_source_frontier_overwritten = false;
      }
      std::chrono::steady_clock::time_point producer_start;
      if (measure_producer_time) {
        producer_start = std::chrono::steady_clock::now();
      }
      const int ret = m_source->Read();

      if (ret == 1)
        return 1;
      thd()->check_yield();

      if (ret == -1) {
        m_source_exhausted = true;
        if (measure_producer_time) {
          m_buffer_manager.ObserveProducerActiveSeconds(
              std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                            producer_start)
                  .count());
        }
      } else {
        assert(ret == 0);
        // Pack the row before advancing the source.
        auto row_buf = store_row_to_buffer(m_tables, m_row_size);
        if (row_buf.empty()) {
          return IteratorReadError(thd(),
                                   "semantic filter row buffering failed");
        }

        // Build the semantic-filter prompt.
        std::string prompt;
        if (m_condition->type() == Item::COND_ITEM) {
          // Concatenate prompts for all semantic filters.
          Item_cond_and *and_cond = static_cast<Item_cond_and *>(m_condition);
          List_iterator<Item> it(*and_cond->argument_list());
          Item *arg;

          while ((arg = it++)) {
            auto *sf = static_cast<Item_func_semantic_filter *>(arg);
            std::string predicate_prompt = sf->compute_prompt();
            if (thd()->is_error()) {
              return 1;
            }
            if (predicate_prompt.empty()) {
              prompt.clear();
              break;
            }
            if (!prompt.empty()) {
              prompt += "\nAnd\n";
            }
            prompt += predicate_prompt;
          }
        } else {
          // Single semantic filter
          auto *sf = static_cast<Item_func_semantic_filter *>(m_condition);
          prompt = sf->compute_prompt();
        }
        if (thd()->is_error()) {
          return 1;
        }

        if (measure_producer_time) {
          m_buffer_manager.ObserveProducerActiveSeconds(
              std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                            producer_start)
                  .count());
        }
        // SQL NULL operands do not satisfy a filter and produce no helper row.
        if (prompt.empty()) {
          continue;
        }

        m_rows_queue.push(std::move(row_buf));

        // Submit the prompt to the semantic helper.
        bool did_submit = false;
        if (m_buffer_manager.PushTuple(prompt, &did_submit)) {
          return IteratorReadError(
              thd(), "ViperSQL semantic helper submission failed");
        }
        if (did_submit) {
          submitted_before_pop = true;
        }
      }
    }

    if (m_source_exhausted && !m_final_batch_flushed) {
      // Flush one residual batch after source EOF.
      if (m_buffer_manager.FlushBatch()) {
        return IteratorReadError(thd(),
                                 "ViperSQL semantic helper flush failed");
      }
      m_final_batch_flushed = true;
    }

    // Refill an idle helper while its output queue is below the watermark.
    if (!m_source_exhausted && !submitted_before_pop &&
        m_buffer_manager.ShouldRefillBeforePop()) {
      must_submit_before_pop = true;
    }
    if (!m_source_exhausted && must_submit_before_pop &&
        !submitted_before_pop && m_buffer_manager.ShouldRefillBeforePop()) {
      continue;
    }

    std::unique_ptr<uint8_t> res_ptr;
    if (m_buffer_manager.HasReadyResult()) {
      res_ptr = m_buffer_manager.PopResult();
    } else if (m_source_exhausted) {
      // Only block once there is no more relational input to form a batch.
      res_ptr = m_buffer_manager.PopResult();
    } else {
      continue;
    }

    if (!res_ptr) {
      if (m_buffer_manager.HasError()) {
        return IteratorReadError(thd(),
                                 "ViperSQL semantic helper response failed");
      }
      if (m_source_exhausted && !m_buffer_manager.HasPendingWork()) {
        if (!m_rows_queue.empty()) {
          return IteratorReadError(
              thd(), "semantic filter result count is misaligned");
        }
        return -1;
      }
      continue;
    }

    // Consume the result aligned with the oldest buffered row.
    if (m_rows_queue.empty()) {
      return IteratorReadError(thd(),
                               "semantic filter result queue is misaligned");
    }
    bool matched = (*res_ptr != 0);
    auto row_buf = std::move(m_rows_queue.front());
    m_rows_queue.pop();

    if (!matched) {
      continue;
    }

    // Restore and emit the matching row.
    LoadIntoTableBuffers(m_tables, row_buf.data());
    m_source_frontier_overwritten =
        !m_source_exhausted && !m_rows_queue.empty();
    return 0;
  }
}
