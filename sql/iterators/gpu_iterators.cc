#include "sql/iterators/gpu_iterators.h"
#include "sql/sql_tmp_table.h"
#include "sql/opt_trace.h"
#include "sql/opt_trace_context.h"
#include "sql/pfs_batch_mode.h"
#include "sql/iterators/timing_iterator.h"
#include "sql/sql_optimizer.h"
#include "scope_guard.h"

/**
   This is a no-op class with a public interface identical to that of the
   IteratorProfilerImpl class. This allows iterators with internal time
   keeping (such as MaterializeIterator) to use the same code whether
   time keeping is enabled or not. And all the mutators are inlinable no-ops,
   so that there should be no runtime overhead.
*/
class DummyIteratorProfiler final : public IteratorProfiler {
 public:
  struct TimeStamp {};

  static TimeStamp Now() { return TimeStamp(); }

  double GetFirstRowMs() const override {
    assert(false);
    return 0.0;
  }

  double GetLastRowMs() const override {
    assert(false);
    return 0.0;
  }

  uint64_t GetNumInitCalls() const override {
    assert(false);
    return 0;
  }

  uint64_t GetNumRows() const override {
    assert(false);
    return 0;
  }

  /*
     The methods below are non-virtual with the same name and signature as
     in IteratorProfilerImpl. The compiler should thus be able to suppress
     calls to these for iterators without profiling.
  */
  void StopInit([[maybe_unused]] TimeStamp start_time) {}

  void IncrementNumRows([[maybe_unused]] uint64_t materialized_rows) {}

  void StopRead([[maybe_unused]] TimeStamp start_time,
                [[maybe_unused]] bool read_ok) {}
};

/**
  Aggregates unsorted data into a temporary table, using update operations
  to keep running aggregates. After that, works as a MaterializeIterator
  in that it allows the temporary table to be scanned.

  'Profiler' should be 'IteratorProfilerImpl' for 'EXPLAIN ANALYZE' and
  'DummyIteratorProfiler' otherwise. It is implemented as a a template parameter
  to minimize the impact this probe has on normal query execution.
 */
template <typename Profiler>
class GPUTemptableAggregateIterator final : public TableRowIterator {
 public:
  GPUTemptableAggregateIterator(
      THD *thd, unique_ptr_destroy_only<RowIterator> subquery_iterator,
      Temp_table_param *temp_table_param, TABLE *table,
      unique_ptr_destroy_only<RowIterator> table_iterator, JOIN *join,
      int ref_slice);

  bool Init() override;
  int Read() override;
  void SetNullRowFlag(bool is_null_row) override {
    m_table_iterator->SetNullRowFlag(is_null_row);
  }
  void EndPSIBatchModeIfStarted() override {
    m_table_iterator->EndPSIBatchModeIfStarted();
    m_subquery_iterator->EndPSIBatchModeIfStarted();
  }
  void UnlockRow() override {}

  const IteratorProfiler *GetProfiler() const override {
    assert(thd()->lex->is_explain_analyze);
    return &m_profiler;
  }

  const Profiler *GetTableIterProfiler() const {
    return &m_table_iter_profiler;
  }

 private:
  /// The iterator we are reading rows from.
  unique_ptr_destroy_only<RowIterator> m_subquery_iterator;

  /// The iterator used to scan the resulting temporary table.
  unique_ptr_destroy_only<RowIterator> m_table_iterator;

  Temp_table_param *m_temp_table_param;
  JOIN *const m_join;
  const int m_ref_slice;

  /**
      Profiling data for this iterator. Used for 'EXPLAIN ANALYZE'.
      @see MaterializeIterator#m_profiler for a description of how
      this is used.
  */
  Profiler m_profiler;

  /**
      Profiling data for m_table_iterator,
      @see MaterializeIterator#m_table_iter_profiler.
  */
  Profiler m_table_iter_profiler;

  // See MaterializeIterator::doing_hash_deduplication().
  bool using_hash_key() const { return table()->hash_field; }

  bool move_table_to_disk(int error, bool was_insert);
};

/**
  Move the in-memory temporary table to disk.

  @param[in] error_code The error code because of which the table
                        is being moved to disk.
  @param[in] was_insert True, if the table is moved to disk during
                        an insert operation.

  @returns true if error.
*/
template <typename Profiler>
bool GPUTemptableAggregateIterator<Profiler>::move_table_to_disk(int error_code,
                                                              bool was_insert) {
  if (create_ondisk_from_heap(thd(), table(), error_code, was_insert,
                              /*ignore_last_dup=*/false,
                              /*is_duplicate=*/nullptr)) {
    return true;
  }
  int error = table()->file->ha_index_init(0, false);
  if (error != 0) {
    PrintError(error);
    return true;
  }
  return false;
}

template <typename Profiler>
GPUTemptableAggregateIterator<Profiler>::GPUTemptableAggregateIterator(
    THD *thd, unique_ptr_destroy_only<RowIterator> subquery_iterator,
    Temp_table_param *temp_table_param, TABLE *table,
    unique_ptr_destroy_only<RowIterator> table_iterator, JOIN *join,
    int ref_slice)
    : TableRowIterator(thd, table),
      m_subquery_iterator(std::move(subquery_iterator)),
      m_table_iterator(std::move(table_iterator)),
      m_temp_table_param(temp_table_param),
      m_join(join),
      m_ref_slice(ref_slice) {}

template <typename Profiler>
bool GPUTemptableAggregateIterator<Profiler>::Init() {
  // NOTE: We never scan these tables more than once, so we don't need to
  // check whether we have already materialized.

  Opt_trace_context *const trace = &thd()->opt_trace;
  Opt_trace_object trace_wrapper(trace);
  Opt_trace_object trace_exec(trace, "temp_table_aggregate");
  trace_exec.add_select_number(m_join->query_block->select_number);
  Opt_trace_array trace_steps(trace, "steps");
  const typename Profiler::TimeStamp start_time = Profiler::Now();

  if (m_subquery_iterator->Init()) {
    return true;
  }

  if (!table()->is_created()) {
    if (instantiate_tmp_table(thd(), table())) {
      return true;
    }
    empty_record(table());
  } else {
    if (table()->file->inited) {
      // If we're being called several times (in particular, as part of a
      // LATERAL join), the table iterator may have started a scan, so end it
      // before we start our own.
      table()->file->ha_index_or_rnd_end();
    }
    table()->file->ha_delete_all_rows();
  }

  // Initialize the index used for finding the groups.
  if (table()->file->ha_index_init(0, false)) {
    return true;
  }
  auto end_unique_index =
      create_scope_guard([&] { table()->file->ha_index_end(); });

  PFSBatchMode pfs_batch_mode(m_subquery_iterator.get());
  std::unordered_map<uint64_t, std::string> hash_to_rawkey;

  // 4) Enter GPU-accelerated aggregation path
  constexpr size_t GPU_BATCH_SZ = 8192;
  GPUHashTable*   gpu_ht  = nullptr;
  GPUAccumulator* gpu_acc = nullptr;
  size_t gpu_cap = 1 << 20;

  if (!gpu_agg_init(&gpu_ht, &gpu_acc, gpu_cap)) {
    my_error(ER_OUT_OF_RESOURCES, MYF(0), "failed to init GPU aggregators");
    return true;
  }

  // Host‐pinned buffers for batching
  std::vector<uint64_t> host_keys; host_keys.reserve(GPU_BATCH_SZ);
  std::vector<Payload>  host_vals; host_vals.reserve(GPU_BATCH_SZ);

  // Device stream and buffers used for batching
  cudaStream_t stream;
  uint64_t* d_in_keys = nullptr;
  Payload*  d_in_vals = nullptr;
  cudaStreamCreate(&stream);
  cudaMalloc(&d_in_keys, GPU_BATCH_SZ * sizeof(uint64_t));
  cudaMalloc(&d_in_vals,  GPU_BATCH_SZ * sizeof(Payload));

  auto flush_batch = [&]() {
    if (host_keys.empty()) return false;
    if (!gpu_agg_batch(gpu_ht, gpu_acc,
                       host_keys.data(),
                       host_vals.data(),
                       d_in_keys,
                       d_in_vals,
                       host_keys.size(),
                       stream)) {
      return true;
    }
    host_keys.clear();
    host_vals.clear();
    return false;
  };

  // 5) Consume all rows from subquery, push to GPU
  for (;;) {
    int read_error = m_subquery_iterator->Read();
    if (read_error > 0 || thd()->is_error())       // fatal
      return true;
    if (read_error < 0)                           // EOF
      break;
    if (thd()->killed) {                          // killed by user
      thd()->send_kill_message();
      return true;
    }
    thd()->check_yield();

    // Evaluate any UDFs/expressions into record[0]
    if (copy_funcs(m_temp_table_param, thd(), CFT_FIELDS))
      return true;

    // 5.a) Compute 64-bit hash of GROUP BY columns
    uint64_t hash = compute_group_hash(table(), m_temp_table_param, &hash_to_rawkey);

    // 5.b) Compute the per-row "delta" payload for each sum/count
    Payload delta;
    compute_payload(m_join->sum_funcs, delta);

    host_keys.push_back(hash);
    host_vals.push_back(delta);
    if (host_keys.size() >= GPU_BATCH_SZ) {
      if (flush_batch()) return true;
    }
  }

  // 6) Flush any remaining batch
  if (flush_batch()) return true;

  // 7) Download final aggregates from GPU
  std::vector<uint64_t> out_keys(gpu_cap);
  std::vector<Payload>  out_vals(gpu_cap);
  size_t                out_n = 0;

  if (!gpu_agg_download(gpu_ht, gpu_acc,
                        out_keys.data(),
                        out_vals.data(),
                        &out_n)) {
    my_error(ER_OUT_OF_RESOURCES, MYF(0), "failed to download GPU aggregates");
    return true;
  }

  // 8) Tear down GPU structures
  cudaStreamSynchronize(stream);
  cudaFree(d_in_keys);
  cudaFree(d_in_vals);
  cudaStreamDestroy(stream);
  gpu_agg_destroy(gpu_ht, gpu_acc);

  // 9) Write each group → a single row in the temp table
  for (size_t i = 0; i < out_n; ++i) {
    uchar *rec0 = table()->record[0];
    if (write_group_key_to_record(table(), out_keys[i], &hash_to_rawkey)) {
      log_to_file("write_group_key_to_record error!");
      return true;
    }
    write_payload_to_record(table(), m_join->sum_funcs, out_vals[i]);

    // Log rec0 contents before writing the row
    std::string rec0_hex = uchar_to_hex(rec0, table()->s->reclength);

    int ha_err = table()->file->ha_write_row(rec0);
    if (ha_err) {
        log_to_file("ha_write_row error! code: " + std::to_string(ha_err));
        return true;
    }
  }

  // 10) Finalize temp‐table materialization
  hash_to_rawkey.clear();
  table()->file->ha_index_end();
  end_unique_index.commit();
  table()->materialized = true;

  // 11) Done: hand off to the table iterator
  m_profiler.StopInit(start_time);
  const bool err = m_table_iterator->Init();
  m_table_iter_profiler.StopInit(start_time);
  return err;
}

template <typename Profiler>
int GPUTemptableAggregateIterator<Profiler>::Read() {
  const typename Profiler::TimeStamp start_time = Profiler::Now();

  /*
    Enable the items which one should use if one wants to evaluate
    anything (e.g. functions in WHERE, HAVING) involving columns of this
    table.
  */
  if (m_join != nullptr && m_ref_slice != -1) {
    if (!m_join->ref_items[m_ref_slice].is_null()) {
      m_join->set_ref_item_slice(m_ref_slice);
    }
  }
  int err = m_table_iterator->Read();
  m_table_iter_profiler.StopRead(start_time, err == 0);
  return err;
}

RowIterator *gpu_temptable_aggregate_iterator::CreateIterator(
    THD *thd, unique_ptr_destroy_only<RowIterator> subquery_iterator,
    Temp_table_param *temp_table_param, TABLE *table,
    unique_ptr_destroy_only<RowIterator> table_iterator, JOIN *join,
    int ref_slice) {
  if (thd->lex->is_explain_analyze) {
    RowIterator *const table_iter_ptr = table_iterator.get();

    auto iter =
        new (thd->mem_root) GPUTemptableAggregateIterator<IteratorProfilerImpl>(
            thd, std::move(subquery_iterator), temp_table_param, table,
            std::move(table_iterator), join, ref_slice);

    /*
      Provide timing data for the iterator that iterates over the temporary
      table. This should include the time spent both materializing the table
      and iterating over it.
    */
    table_iter_ptr->SetOverrideProfiler(iter->GetTableIterProfiler());
    return iter;
  } else {
    return new (thd->mem_root)
        GPUTemptableAggregateIterator<DummyIteratorProfiler>(
            thd, std::move(subquery_iterator), temp_table_param, table,
            std::move(table_iterator), join, ref_slice);
  }
}

GPUHashJoinIterator::GPUHashJoinIterator(
    THD* thd,
    unique_ptr_destroy_only<RowIterator> build_input,
    const Prealloced_array<TABLE*, 4>& build_input_tables,
    double estimated_build_rows,
    unique_ptr_destroy_only<RowIterator> probe_input,
    const Prealloced_array<TABLE*, 4>& probe_input_tables,
    bool store_rowids,
    table_map tables_to_get_rowid_for,
    size_t max_memory_available,
    const std::vector<HashJoinCondition>& join_conditions,
    bool allow_spill_to_disk,
    JoinType join_type,
    const Mem_root_array<Item*>& extra_conditions,
    bool probe_input_batch_mode,
    uint64_t* hash_table_generation)
    : RowIterator(thd),
      m_build_input(std::move(build_input)),
      m_probe_input(std::move(probe_input)),
      m_build_input_tables(build_input_tables, store_rowids,
                           tables_to_get_rowid_for,
                           /*tables_to_store_contents_of_null_rows_for=*/0),
      m_probe_input_tables(probe_input_tables, store_rowids,
                           tables_to_get_rowid_for,
                           /*tables_to_store_contents_of_null_rows_for=*/0),
      m_tables_to_get_rowid_for(tables_to_get_rowid_for),
      m_join_conditions(PSI_NOT_INSTRUMENTED, join_conditions.data(),
                        join_conditions.data() + join_conditions.size()),
      m_allow_spill_to_disk(allow_spill_to_disk),
      m_join_type(join_type),
      m_estimated_build_rows(estimated_build_rows),
      m_probe_input_batch_mode(probe_input_batch_mode),
      m_hash_table_generation(hash_table_generation),
      // Initialize buffer manager with memory and helper name
      m_buffer_manager(max_memory_available, estimated_build_rows, "GPUHashJoinHelper")  
{
  assert(m_build_input != nullptr);
  assert(m_probe_input != nullptr);

  if (extra_conditions.size() == 1) {
    m_extra_condition = extra_conditions[0];
  } else if (extra_conditions.size() > 1) {
    List<Item> items;
    for (Item* cond : extra_conditions) {
      items.push_back(cond);
    }
    m_extra_condition = new Item_cond_and(items);
    m_extra_condition->quick_fix_field();
    m_extra_condition->update_used_tables();
    m_extra_condition->apply_is_true();
  }
}

// Returns extracted raw row buffer or empty vector on failure
std::vector<uint8_t> GPUHashJoinIterator::store_row_to_buffer(const pack_rows::TableCollection& tables) {
  size_t row_size_upper_bound = m_row_size;
  if (tables.has_blob_column()) {
    row_size_upper_bound = ComputeRowSizeUpperBound(tables);
  }

  // Allocate buffer to hold the raw row bytes
  std::vector<uint8_t> row_buffer(row_size_upper_bound);

  // Copy raw row bytes from table buffers into row_buffer
  uchar* dest = row_buffer.data();
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

bool GPUHashJoinIterator::extract_join_key_for_row(THD* thd, const pack_rows::TableCollection& tables) {
  m_buffer.length(0);
  for (const HashJoinCondition &cond : m_join_conditions) {
    bool null_found = cond.join_condition()->append_join_key_for_hash_join(
        thd, tables.tables_bitmap(), cond, m_join_conditions.size() > 1,
        &m_buffer);
    if (null_found) {
      return false;
    }
  }
  return true;
}

bool GPUHashJoinIterator::Init() {
  // 1. Initialize build and probe input iterators
  PrepareForRequestRowId(m_build_input_tables.tables(),
                         m_tables_to_get_rowid_for);
  if (m_build_input->Init()) {
    return true;
  }
  m_probe_input->EndPSIBatchModeIfStarted();

  int idx = 0;
  m_buffer_manager.SetStatus("BUILD");
  m_row_size = ComputeRowSizeUpperBound(m_build_input_tables);
  m_build_input->SetNullRowFlag(/*is_null_row=*/false);
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

    // Extract join key from build row
    if (!extract_join_key_for_row(thd(), m_build_input_tables)) {
      // Skip this row since join key contains NULL
      continue;
    }

    auto row_buf = store_row_to_buffer(m_build_input_tables);
    if (row_buf.empty()) {
      // Handle error: failed to store row buffer
      return true;
    }

    // Now caller pushes row_buf into the appropriate buffer vector
    build_rows_buffer.push_back(std::move(row_buf));

    // Create key-index pair
    std::string key_copy(m_buffer.ptr(), m_buffer.length());
    KeyIndexPair pair{key_copy, static_cast<uint32_t>(idx)};
    if (m_buffer_manager.PushTuple(pair)) {
      return true;  // error pushing or kernel launch
    }

    idx += 1;
  }

  // 4. Flush remaining batched build keys to GPU
  if (m_buffer_manager.FlushBatch()) {
    return true;  // error flushing last batch
  }

  // 5. Switch to probe phase.
  m_buffer_manager.PopResult(); // sync build stream
  m_buffer_manager.SetStatus("PROBE");

  if (m_probe_input->Init()) {
    return true;
  }
  PrepareForRequestRowId(m_probe_input_tables.tables(),
                         m_tables_to_get_rowid_for);
  m_row_size = ComputeRowSizeUpperBound(m_probe_input_tables);
  if (m_probe_input_batch_mode) {
    m_probe_input->StartPSIBatchMode();
  }
  return false;
}

int GPUHashJoinIterator::Read() {
  for (;;) {
    // Always try to read one probe row each Read() call
    int ret = m_probe_input->Read();
    if (ret == 1) {
      return 1;  // error
    }
    thd()->check_yield();

    if (ret == 0) {
      RequestRowId(m_probe_input_tables.tables(), m_tables_to_get_rowid_for);
      if (!extract_join_key_for_row(thd(), m_probe_input_tables)) {
        // Skip probe row with NULL join key, continue loop
        continue;
      }

      auto probe_row_buf = store_row_to_buffer(m_probe_input_tables);
      if (probe_row_buf.empty()) {
        return 1;  // error
      }

      m_probe_rows_queue.push(std::move(probe_row_buf));

      // Push the key-index pair to GPU buffer manager
      std::string key_copy(m_buffer.ptr(), m_buffer.length());
      uint32_t probe_idx = m_probe_rows_queue.size() - 1;
      KeyIndexPair pair{key_copy, probe_idx};
      if (m_buffer_manager.PushTuple(pair)) {
        return 1;  // error pushing or launching kernel
      }
    } else if (ret == -1) {
      // Probe input exhausted, flush any remaining probe keys
      if (m_buffer_manager.FlushBatch()) {
        return 1;  // error flushing last batch
      }
    }

    // Now try to get one matched result
    auto matched_build_idx_ptr = m_buffer_manager.PopResult();

    if (!matched_build_idx_ptr) {
      // If probe iterator exhausted and no results are available,
      // then this is end of stream
      if (ret == -1) {
        return -1;  // no more rows
      }

      // Otherwise no result yet, wait for GPU batch to be running
      if (!m_buffer_manager.IsExternalCallRunning()) {
        // Not running yet, continue reading more probe rows (loop)
        continue;
      }

      // GPU batch is running, but no results available yet — return 0 indicating no row ready now
      return 0;
    }

    // Got a matched build index, skip if NOT_FOUND
    if (*matched_build_idx_ptr == gpuhashjoinhelpers::NOT_FOUND) {
      // Ignore and continue loop to get next match
      m_probe_rows_queue.pop();
      continue;
    }

    // Load matched build row to build tables
    LoadIntoTableBuffers(m_build_input_tables, build_rows_buffer[*matched_build_idx_ptr].data());

    // Load corresponding probe row to probe tables
    if (m_probe_rows_queue.empty()) {
      log_to_file("Error: probe row buffer queue empty but matched build index found");
      return 1;
    }
    auto probe_row_buf = std::move(m_probe_rows_queue.front());
    m_probe_rows_queue.pop();

    LoadIntoTableBuffers(m_probe_input_tables, probe_row_buf.data());

    // Return success, one joined row ready
    return 0;
  }
}
