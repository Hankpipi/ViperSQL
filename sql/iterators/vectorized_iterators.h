#ifndef SQL_ITERATORS_GPU_ITERATORS_H_
#define SQL_ITERATORS_GPU_ITERATORS_H_

#include "sql/iterators/composite_iterators.h"
#include "sql/iterators/helpers/gpu_agg.h"
#include "sql/iterators/helpers/gpu_hash_join.h"
#include "sql/iterators/external_helper_buffer.h"

namespace gpu_temptable_aggregate_iterator {
/**
   Create an iterator that aggregates the output rows from another iterator
   into a temporary table and then sets up a (pre-existing) iterator to
   access the temporary table.
   @see GPUTemptableAggregateIterator.

   @param thd Thread handler.
   @param subquery_iterator input to aggregation.
   @param temp_table_param temporary table settings.
   @param table_iterator Iterator used for scanning the temporary table
    after materialization.
   @param table the temporary table.
   @param join the JOIN in which we aggregate.
   @param ref_slice the slice to set when accessing temporary table;
    used if anything upstream  wants to evaluate values based on its contents.
   @return the iterator.
*/
RowIterator *CreateIterator(
    THD *thd, unique_ptr_destroy_only<RowIterator> subquery_iterator,
    Temp_table_param *temp_table_param, TABLE *table,
    unique_ptr_destroy_only<RowIterator> table_iterator, JOIN *join,
    int ref_slice);

}

class GPUHashJoinIterator : public RowIterator {
 public:
  GPUHashJoinIterator(
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
    uint64_t* hash_table_generation);

  bool Init() override;
  int Read() override;

  void SetNullRowFlag(bool is_null_row) override {
    m_build_input->SetNullRowFlag(is_null_row);
    m_probe_input->SetNullRowFlag(is_null_row);
  }

  void EndPSIBatchModeIfStarted() override {
    m_build_input->EndPSIBatchModeIfStarted();
    m_probe_input->EndPSIBatchModeIfStarted();
  }

  void UnlockRow() override {
    // Since both inputs may have been materialized to disk, we cannot unlock
    // them.
  }

 private:
  // Underlying build and probe iterators
  const unique_ptr_destroy_only<RowIterator> m_build_input;
  const unique_ptr_destroy_only<RowIterator> m_probe_input;

  // Table collections for build and probe inputs
  pack_rows::TableCollection m_build_input_tables;
  pack_rows::TableCollection m_probe_input_tables;

  table_map m_tables_to_get_rowid_for;

  // Join conditions
  Prealloced_array<HashJoinCondition, 4> m_join_conditions;

  // Combined extra conditions
  Item* m_extra_condition{nullptr};

  JoinType m_join_type;
  bool m_allow_spill_to_disk;
  double m_estimated_build_rows;
  bool m_probe_input_batch_mode;
  uint64_t* m_hash_table_generation;
  std::vector<std::vector<uint8_t>> build_rows_buffer;
  std::queue<std::vector<uint8_t>> m_probe_rows_queue;

  // Buffer manager encapsulating input batch and result queue
  String m_buffer;
  ExternalHelperBufferManager<KeyIndexPair, uint32_t> m_buffer_manager;
  
  size_t m_row_size;

  // Extract join key from the current row of the given tables' buffers into m_buffer
  bool extract_join_key_for_row(THD* thd, const pack_rows::TableCollection& tables);
};


// Store the current row from the given tables' buffers into a CPU memory buffer
std::vector<uint8_t> store_row_to_buffer(const pack_rows::TableCollection& tables, size_t row_size);

class VectorizedFilterIterator final : public RowIterator {
 public:
  VectorizedFilterIterator(THD *thd, unique_ptr_destroy_only<RowIterator> source,
                 pack_rows::TableCollection tables,
                 Item *condition)
    : RowIterator(thd),
      m_source(std::move(source)),
      m_tables(std::move(tables)),
      m_condition(condition),
      m_buffer_manager(64LL * 1024 * 1024,
                       BATCH_SIZE,
                       "LLMFilter") {}

  bool Init() override;

  int Read() override;

  void SetNullRowFlag(bool is_null_row) override {
    m_source->SetNullRowFlag(is_null_row);
  }

  void StartPSIBatchMode() override { m_source->StartPSIBatchMode(); }
  void EndPSIBatchModeIfStarted() override {
    m_source->EndPSIBatchModeIfStarted();
  }
  void UnlockRow() override { m_source->UnlockRow(); }

 private:
  unique_ptr_destroy_only<RowIterator> m_source;
  pack_rows::TableCollection m_tables;
  Item *m_condition;

  size_t m_row_size;
  std::queue<std::vector<uint8_t>> m_rows_queue;
  ExternalHelperBufferManager<std::string, uint8_t> m_buffer_manager;
};

#endif