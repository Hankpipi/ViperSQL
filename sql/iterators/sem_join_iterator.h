#ifndef SQL_ITERATORS_EXTERNAL_HELPERS_SEMHELPERS_SEM_JOIN_ITERATOR_H_
#define SQL_ITERATORS_EXTERNAL_HELPERS_SEMHELPERS_SEM_JOIN_ITERATOR_H_

/*
   Copyright (c) 2025, Songsong Mo

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; version 2 of the License.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program; if not, write to the
   Free Software Foundation, Inc., 59 Temple Place, Suite 330,
   Boston, MA  02111-1307  USA
*/

#include "sql/iterators/composite_iterators.h"
#include "sql/iterators/helpers/sem_join_helper.h"
#include "sql/iterators/external_helper_buffer.h"
#include "sql/join_optimizer/access_path.h"
#include "sql/iterators/hash_join_iterator.h" 
#include "sql/iterators/vectorized_iterators.h"
#include "sql/item_func_semantic.h"
#include <vector>


namespace sem_join_template_iterator {
/**
   Create an iterator that aggregates the output rows from another iterator
   into a temporary table and then sets up a (pre-existing) iterator to
   access the temporary table.

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

class SemJoinIterator : public RowIterator {
 public:
  SemJoinIterator(
    THD* thd,
    unique_ptr_destroy_only<RowIterator> build_input,
    const Prealloced_array<TABLE*, 4>& build_input_tables,
    double estimated_build_rows,
    unique_ptr_destroy_only<RowIterator> probe_input,
    const Prealloced_array<TABLE*, 4>& probe_input_tables,
    bool store_rowids,
    table_map tables_to_get_rowid_for,
    size_t max_memory_available,
    const std::vector<Item_func_sem_join*>& sem_conditions,
    bool allow_spill_to_disk,
    JoinType join_type,
    bool probe_input_batch_mode,
    uint64_t* hash_table_generation,
    AccessPath::Type impl_type);

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
  std::vector<Item_func_sem_join*> m_sem_conditions;

  JoinType m_join_type;
  bool m_allow_spill_to_disk;
  double m_estimated_build_rows;
  bool m_probe_input_batch_mode;
  uint64_t* m_hash_table_generation;
  std::vector<std::vector<uint8_t>> build_rows_buffer;
  std::queue<std::vector<uint8_t>> m_probe_rows_queue;

  // Buffer manager encapsulating input batch and result queue
  String m_buffer;
  ViperFlow<KeyIndexPair, std::pair<size_t, size_t>> m_buffer_manager;
  
  size_t m_row_size;

  AccessPath::Type m_impl_type;

  // Optimization: Track the index of the probe row currently residing in m_probe_input_tables
  // Initialize to a sentinel value (e.g., NOT_FOUND or max size_t) in constructor.
  size_t m_current_loaded_probe_idx;

  // Track the global index corresponding to m_probe_rows_queue.front(). Starts at 0.
  size_t m_queue_front_global_idx;

  // Holds the data for the currently active probe row
  // to ensure pointers in table->record[0] remain valid after Read() returns.
  std::vector<uchar> m_active_probe_row;

  bool extract_join_key_for_row(bool is_probe_phase);
};

#endif  // SQL_ITERATORS_EXTERNAL_HELPERS_SEMHELPERS_SEM_JOIN_ITERATOR_H_
