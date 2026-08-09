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

#include <cstddef>
#include <cstdint>
#include <queue>
#include <string>
#include <vector>

#include "sql/item_func_semantic.h"
#include "sql/iterators/composite_iterators.h"
#include "sql/iterators/external_helper_buffer.h"
#include "sql/iterators/hash_join_iterator.h"
#include "sql/iterators/helpers/sem_join_helper.h"
#include "sql/join_optimizer/access_path.h"

class SemJoinIterator : public RowIterator {
 public:
  SemJoinIterator(THD *thd, unique_ptr_destroy_only<RowIterator> build_input,
                  const Prealloced_array<TABLE *, 4> &build_input_tables,
                  unique_ptr_destroy_only<RowIterator> probe_input,
                  const Prealloced_array<TABLE *, 4> &probe_input_tables,
                  bool store_rowids, table_map tables_to_get_rowid_for,
                  size_t max_memory_available,
                  const std::vector<Item_func_sem_join *> &sem_conditions,
                  bool probe_input_batch_mode, AccessPath::Type impl_type);

  ~SemJoinIterator() override;

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
  enum class KeyExtractionResult { kOk, kNull, kError };

  const unique_ptr_destroy_only<RowIterator> m_build_input;
  const unique_ptr_destroy_only<RowIterator> m_probe_input;

  pack_rows::TableCollection m_build_input_tables;
  pack_rows::TableCollection m_probe_input_tables;

  table_map m_tables_to_get_rowid_for;
  bool m_probe_input_batch_mode;
  bool m_valid_semantic_condition{false};
  std::vector<std::vector<uint8_t>> m_build_rows;
  std::queue<std::vector<uint8_t>> m_probe_rows_queue;

  String m_buffer;
  std::string m_semantic_prompt;
  ViperFlow<semhelpers::KeyIndexPair, std::pair<size_t, size_t>>
      m_buffer_manager;

  size_t m_row_size{0};
  size_t m_probe_row_index{0};
  size_t m_current_loaded_probe_idx;
  size_t m_queue_front_global_idx;

  // Keep the active snapshot alive while TABLE fields refer to its storage.
  std::vector<uchar> m_active_probe_row;

  bool m_probe_input_exhausted{false};
  bool m_probe_batch_flushed{false};
  // Loading an older returned match can overwrite the paused probe child's
  // TABLE buffers. Restore the latest queued probe before resuming it.
  bool m_probe_frontier_overwritten{false};

  Item *m_probe_item{nullptr};
  Item *m_build_item{nullptr};

  KeyExtractionResult extract_join_key_for_row(bool is_probe_phase);
};

#endif  // SQL_ITERATORS_EXTERNAL_HELPERS_SEMHELPERS_SEM_JOIN_ITERATOR_H_
