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

#include "sql/iterators/sem_join_iterator.h"

#include <limits>
#include <string>
#include <utility>

#include "mysqld_error.h"
#include "sql/iterators/vectorized_iterators.h"
#include "sql/pfs_batch_mode.h"

namespace {

std::string GetSemanticJoinPrompt(
    const std::vector<Item_func_sem_join *> &conditions) {
  if (conditions.size() != 1 || conditions.front() == nullptr) return {};
  Item_func_sem_join *condition = conditions.front();
  if (condition->argument_count() != 3 ||
      condition->arguments()[0] == nullptr ||
      !condition->arguments()[0]->const_for_execution()) {
    return {};
  }
  return condition->prompt();
}

void ReportSemanticJoinError(THD *thd, const char *operation) {
  if (thd != nullptr && !thd->is_error()) {
    my_error(ER_INTERNAL_ERROR, MYF(0), operation);
  }
}

}  // namespace

SemJoinIterator::SemJoinIterator(
    THD *thd, unique_ptr_destroy_only<RowIterator> build_input,
    const Prealloced_array<TABLE *, 4> &build_input_tables,
    unique_ptr_destroy_only<RowIterator> probe_input,
    const Prealloced_array<TABLE *, 4> &probe_input_tables, bool store_rowids,
    table_map tables_to_get_rowid_for, size_t max_memory_available,
    const std::vector<Item_func_sem_join *> &sem_conditions,
    bool probe_input_batch_mode, AccessPath::Type impl_type)
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
      m_probe_input_batch_mode(probe_input_batch_mode),
      m_semantic_prompt(GetSemanticJoinPrompt(sem_conditions)),
      m_buffer_manager(max_memory_available, GetSemImplName(impl_type), 1, {},
                       m_semantic_prompt),
      m_current_loaded_probe_idx(std::numeric_limits<size_t>::max()),
      m_queue_front_global_idx(0) {
  assert(m_build_input != nullptr);
  assert(m_probe_input != nullptr);
  if (sem_conditions.size() == 1 && sem_conditions.front() != nullptr &&
      sem_conditions.front()->argument_count() == 3 &&
      !m_semantic_prompt.empty()) {
    Item_func_sem_join *sem_func = sem_conditions.front();
    Item *left = sem_func->arguments()[1];
    Item *right = sem_func->arguments()[2];
    const table_map build_tables = m_build_input_tables.tables_bitmap();
    const table_map probe_tables = m_probe_input_tables.tables_bitmap();
    const auto belongs_to = [](Item *item, table_map tables) {
      if (item == nullptr || tables == 0) return false;
      const table_map used_tables = item->used_tables();
      return (used_tables & tables) != 0 && (used_tables & ~tables) == 0;
    };
    if ((build_tables & probe_tables) == 0) {
      if (belongs_to(left, build_tables) && belongs_to(right, probe_tables)) {
        m_build_item = left;
        m_probe_item = right;
      } else if (belongs_to(right, build_tables) &&
                 belongs_to(left, probe_tables)) {
        m_build_item = right;
        m_probe_item = left;
      }
    }
    m_valid_semantic_condition =
        m_build_item != nullptr && m_probe_item != nullptr;
  }
}

SemJoinIterator::~SemJoinIterator() {
  (void)m_buffer_manager.FlushControl("RESET");
}

SemJoinIterator::KeyExtractionResult SemJoinIterator::extract_join_key_for_row(
    bool is_probe_phase) {
  Item *item_to_read = is_probe_phase ? m_probe_item : m_build_item;
  if (item_to_read == nullptr) return KeyExtractionResult::kError;

  m_buffer.length(0);
  String *value = item_to_read->val_str(&m_buffer);
  if (thd()->is_error()) return KeyExtractionResult::kError;
  if (item_to_read->null_value) return KeyExtractionResult::kNull;
  if (value == nullptr) return KeyExtractionResult::kError;

  // Keep the key valid after the Item's backing record changes.
  if (value != &m_buffer && m_buffer.copy(*value)) {
    if (!thd()->is_error()) {
      my_error(ER_OUTOFMEMORY, MYF(0), value->length());
    }
    return KeyExtractionResult::kError;
  }
  if (m_buffer.length() != 0 && m_buffer.ptr() == nullptr) {
    return KeyExtractionResult::kError;
  }
  return KeyExtractionResult::kOk;
}

bool SemJoinIterator::Init() {
  if (!m_valid_semantic_condition || thd()->is_error()) {
    ReportSemanticJoinError(thd(), "invalid semantic join condition");
    return true;
  }
  if (m_buffer_manager.FlushControl("RESET")) {
    ReportSemanticJoinError(thd(), "semantic join reset failed");
    return true;
  }
  if (m_buffer_manager.Reset()) {
    ReportSemanticJoinError(thd(), "semantic join buffer reset failed");
    return true;
  }
  m_build_rows.clear();
  while (!m_probe_rows_queue.empty()) m_probe_rows_queue.pop();
  m_active_probe_row.clear();
  m_current_loaded_probe_idx = std::numeric_limits<size_t>::max();
  m_queue_front_global_idx = 0;
  m_probe_input_exhausted = false;
  m_probe_batch_flushed = false;
  m_probe_frontier_overwritten = false;

  PrepareForRequestRowId(m_build_input_tables.tables(),
                         m_tables_to_get_rowid_for);
  if (m_build_input->Init()) {
    ReportSemanticJoinError(thd(),
                            "semantic join build input initialization failed");
    return true;
  }
  m_probe_input->EndPSIBatchModeIfStarted();

  m_buffer_manager.SetStatus("BUILD");
  m_row_size = ComputeRowSizeUpperBound(m_build_input_tables);
  m_build_input->SetNullRowFlag(/*is_null_row=*/false);
  PFSBatchMode batch_mode(m_build_input.get());
  while (true) {
    int ret = m_build_input->Read();
    if (ret == 1) {  // error
      ReportSemanticJoinError(thd(), "semantic join build input read failed");
      return true;
    }
    thd()->check_yield();
    if (ret == -1) {  // EOF
      break;
    }
    assert(ret == 0);
    RequestRowId(m_build_input_tables.tables(), m_tables_to_get_rowid_for);

    const KeyExtractionResult key_result = extract_join_key_for_row(false);
    if (key_result == KeyExtractionResult::kError) {
      ReportSemanticJoinError(thd(),
                              "semantic join build key extraction failed");
      return true;
    }
    if (key_result == KeyExtractionResult::kNull) continue;

    auto row_buf = store_row_to_buffer(m_build_input_tables, m_row_size);
    if (row_buf.empty()) {
      ReportSemanticJoinError(thd(),
                              "semantic join build row buffering failed");
      return true;
    }

    const size_t row_index = m_build_rows.size();
    m_build_rows.push_back(std::move(row_buf));

    std::string key_copy;
    if (m_buffer.length() != 0) {
      key_copy.assign(m_buffer.ptr(), m_buffer.length());
    }
    semhelpers::KeyIndexPair pair{std::move(key_copy), row_index};
    if (m_buffer_manager.PushTuple(pair)) {
      ReportSemanticJoinError(thd(), "semantic join build submission failed");
      return true;
    }
  }

  if (m_buffer_manager.FlushBatch()) {
    ReportSemanticJoinError(thd(), "semantic join build flush failed");
    return true;
  }

  if (m_buffer_manager.FlushControl("BUILD_DONE")) {
    ReportSemanticJoinError(thd(), "semantic join build completion failed");
    return true;
  }

  if (m_buffer_manager.HasError()) {
    ReportSemanticJoinError(thd(), "semantic join build helper failed");
    return true;
  }
  m_buffer_manager.SetStatus("PROBE");
  m_probe_row_index = 0;

  if (m_probe_input->Init()) {
    ReportSemanticJoinError(thd(),
                            "semantic join probe input initialization failed");
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

int SemJoinIterator::Read() {
  // The obligation is latched for this public Read() call and is satisfied by
  // one successful probe-batch submission before a match can be returned.
  bool must_submit_before_pop =
      !m_probe_input_exhausted && m_buffer_manager.ShouldRefillBeforePop();
  bool submitted_before_pop = false;

  for (;;) {
    // Fill-first scheduling: form the next probe batch before waiting for a
    // helper result. Queued output is still exposed one tuple at a time.
    if (!m_probe_input_exhausted) {
      if (m_probe_frontier_overwritten) {
        if (m_probe_rows_queue.empty()) {
          ReportSemanticJoinError(thd(),
                                  "semantic join probe frontier is invalid");
          return 1;
        }
        LoadIntoTableBuffers(m_probe_input_tables,
                             m_probe_rows_queue.back().data());
        m_probe_frontier_overwritten = false;
      }
      const int ret = m_probe_input->Read();
      if (ret == 1) {
        ReportSemanticJoinError(thd(), "semantic join probe input read failed");
        return 1;  // error
      }
      thd()->check_yield();

      if (ret == -1) {
        m_probe_input_exhausted = true;
      } else {
        assert(ret == 0);
        RequestRowId(m_probe_input_tables.tables(), m_tables_to_get_rowid_for);
        const KeyExtractionResult key_result = extract_join_key_for_row(true);
        if (key_result == KeyExtractionResult::kError) {
          ReportSemanticJoinError(thd(),
                                  "semantic join probe key extraction failed");
          return 1;
        }
        if (key_result == KeyExtractionResult::kNull) continue;

        auto probe_row_buf =
            store_row_to_buffer(m_probe_input_tables, m_row_size);
        if (probe_row_buf.empty()) {
          ReportSemanticJoinError(thd(),
                                  "semantic join probe row buffering failed");
          return 1;
        }

        m_probe_rows_queue.push(std::move(probe_row_buf));

        std::string key_copy;
        if (m_buffer.length() != 0) {
          key_copy.assign(m_buffer.ptr(), m_buffer.length());
        }
        semhelpers::KeyIndexPair pair{std::move(key_copy), m_probe_row_index};
        ++m_probe_row_index;
        bool did_submit = false;
        if (m_buffer_manager.PushTuple(pair, &did_submit)) {
          ReportSemanticJoinError(thd(),
                                  "semantic join probe submission failed");
          return 1;
        }
        if (did_submit) submitted_before_pop = true;
      }
    }

    if (m_probe_input_exhausted && !m_probe_batch_flushed) {
      // Flush exactly once; EOF submits a residual below the minimum size.
      if (m_buffer_manager.FlushBatch()) {
        ReportSemanticJoinError(thd(), "semantic join probe flush failed");
        return 1;  // error flushing last batch
      }
      m_probe_batch_flushed = true;
    }

    // At low output supply, first advance the probe child until PushTuple
    // dispatches the next batch. A stocked result queue applies backpressure;
    // EOF bypasses this gate only after the residual flush above.
    if (!m_probe_input_exhausted && !submitted_before_pop &&
        m_buffer_manager.ShouldRefillBeforePop()) {
      must_submit_before_pop = true;
    }
    if (!m_probe_input_exhausted && must_submit_before_pop &&
        !submitted_before_pop && m_buffer_manager.ShouldRefillBeforePop()) {
      continue;
    }

    std::unique_ptr<std::pair<size_t, size_t>> result_pair_ptr;
    if (m_buffer_manager.HasReadyResult()) {
      result_pair_ptr = m_buffer_manager.PopResult();
    } else if (m_probe_input_exhausted) {
      // No relational tuples remain, so it is now correct to wait for the
      // final external request.
      result_pair_ptr = m_buffer_manager.PopResult();
    } else {
      continue;
    }

    if (!result_pair_ptr) {
      if (m_buffer_manager.HasError()) {
        ReportSemanticJoinError(thd(), "semantic join result fetch failed");
        return 1;
      }
      if (m_probe_input_exhausted && !m_buffer_manager.HasPendingWork()) {
        while (!m_probe_rows_queue.empty()) m_probe_rows_queue.pop();
        return -1;
      }
      continue;
    }

    const size_t returned_probe_idx = result_pair_ptr->first;
    const size_t matched_build_idx = result_pair_ptr->second;

    if (matched_build_idx >= m_build_rows.size()) {
      ReportSemanticJoinError(thd(),
                              "semantic join returned an invalid build row");
      return 1;
    }
    LoadIntoTableBuffers(m_build_input_tables,
                         m_build_rows[matched_build_idx].data());

    if (returned_probe_idx == m_current_loaded_probe_idx) {
      // The helper can return multiple build matches for one probe row.
      LoadIntoTableBuffers(m_probe_input_tables, m_active_probe_row.data());
    } else {
      while (!m_probe_rows_queue.empty() &&
             m_queue_front_global_idx < returned_probe_idx) {
        m_probe_rows_queue.pop();
        ++m_queue_front_global_idx;
      }
      if (m_probe_rows_queue.empty() ||
          m_queue_front_global_idx != returned_probe_idx) {
        ReportSemanticJoinError(thd(),
                                "semantic join returned an invalid probe row");
        return 1;
      }

      m_active_probe_row = std::move(m_probe_rows_queue.front());
      m_probe_rows_queue.pop();
      LoadIntoTableBuffers(m_probe_input_tables, m_active_probe_row.data());
      m_current_loaded_probe_idx = returned_probe_idx;
      ++m_queue_front_global_idx;
    }
    m_probe_frontier_overwritten =
        !m_probe_input_exhausted && returned_probe_idx + 1 < m_probe_row_index;
    return 0;
  }
}
