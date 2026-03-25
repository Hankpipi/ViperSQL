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
#include "sql/sql_tmp_table.h"
#include "sql/opt_trace.h"
#include "sql/opt_trace_context.h"
#include "sql/pfs_batch_mode.h"
#include "sql/iterators/timing_iterator.h"
#include "sql/sql_optimizer.h"
#include "scope_guard.h"

static std::string HexDump(const void* data, size_t size) {
    const unsigned char* p = static_cast<const unsigned char*>(data);
    std::ostringstream oss;
    for (size_t i = 0; i < size; ++i) {
        oss << std::hex << std::setw(2) << std::setfill('0') << (int)p[i] << " ";
    }
    return oss.str();
}

SemJoinIterator::SemJoinIterator(
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
    AccessPath::Type impl_type)
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
      m_sem_conditions(sem_conditions),
      m_allow_spill_to_disk(allow_spill_to_disk),
      m_join_type(join_type),
      m_estimated_build_rows(estimated_build_rows),
      m_probe_input_batch_mode(probe_input_batch_mode),
      m_hash_table_generation(hash_table_generation),
      m_impl_type(impl_type),
      // Initialize buffer manager with memory and helper name
      m_buffer_manager(max_memory_available, estimated_build_rows, GetSemImplName(impl_type)),
      m_current_loaded_probe_idx(std::numeric_limits<size_t>::max()),
      m_queue_front_global_idx(0)
{
  assert(m_build_input != nullptr);
  assert(m_probe_input != nullptr);
  // Extract prompt from the first condition (if exists) and set it
  if (!m_sem_conditions.empty()) {
      std::string p = m_sem_conditions[0]->prompt();
      // [TODO] send prompt to ViperFlow
      log_to_file("SemJoinIterator: Prompt set to: " + p);
  }
  if (!m_sem_conditions.empty()) {
    Item_func_sem_join* sem_func = m_sem_conditions[0];

    // 1. Find which argument belongs to the Build Tables
    m_build_item = resolve_item_for_tables(sem_func, m_build_input_tables);
    
    // 2. Find which argument belongs to the Probe Tables
    m_probe_item = resolve_item_for_tables(sem_func, m_probe_input_tables);

    // Fallback/Safety: If resolution failed (e.g. complex expression), 
    // try to deduce: if we found Build, the other is Probe.
    if (m_build_item && !m_probe_item) {
          m_probe_item = (m_build_item == sem_func->arguments()[1]) 
                        ? sem_func->arguments()[2] 
                        : sem_func->arguments()[1];
    } else if (!m_build_item && m_probe_item) {
          m_build_item = (m_probe_item == sem_func->arguments()[1]) 
                        ? sem_func->arguments()[2] 
                        : sem_func->arguments()[1];
    }

    if (!m_build_item || !m_probe_item) {
        log_to_file("SemJoinIterator Error: Could not map arguments to Build/Probe tables!");
    }
  }
}

Item* SemJoinIterator::resolve_item_for_tables(Item_func_sem_join* sem_func, const pack_rows::TableCollection& tables) {
    // We only care about args[1] and args[2]. Arg[0] is the prompt.
    Item* candidates[] = { sem_func->arguments()[1], sem_func->arguments()[2] };

    for (Item* item : candidates) {
      if (item->type() == Item::FIELD_ITEM) {
          Item_field* field_item = static_cast<Item_field*>(item);
          for (size_t i = 0; i < tables.tables().size(); ++i) {
            if (tables.tables()[i].table == field_item->field->table) {
              return item;
            }
          }
      }
    }
    return nullptr;
}

SemJoinIterator::~SemJoinIterator() {
  log_to_file("SemJoinIterator::~SemJoinIterator");

  (void)m_buffer_manager.FlushControl("RESET");
}

bool SemJoinIterator::extract_join_key_for_row(bool is_probe_phase) {
  // 1. Safety Checks
  if (m_sem_conditions.empty()) {
      log_to_file("extract_join_key_for_row: Error - m_sem_conditions is empty!");
      return false;
  }

  Item_func_sem_join* sem_func = m_sem_conditions[0];
  Item* item_to_read = is_probe_phase ? m_probe_item : m_build_item;

  if (item_to_read == nullptr) return false;

  // 2. OPTIMIZED EXTRACT: Pass m_buffer as the scratch space
  //    MySQL will write here if it needs temp storage.
  String* s = item_to_read->val_str(&m_buffer);
  
  if (!s || item_to_read->null_value) {
      return false; // NULL handling
  }

  // 3. NORMALIZE: Ensure the data is actually in m_buffer
  if (s != &m_buffer) {
      // The Item returned a pointer to internal memory (e.g. Record Buffer).
      // We must copy it into m_buffer so we own the data.
      if (m_buffer.copy(*s)) {
          return false; // OOM error
      }
  }
  return true;
}

bool SemJoinIterator::Init() {
  log_to_file("SemJoinIterator::Init");
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
    if (!extract_join_key_for_row(false)) {
      // Skip this row since join key contains NULL
      continue;
    }

    auto row_buf = store_row_to_buffer(m_build_input_tables, m_row_size);
    if (row_buf.empty()) {
      // Handle error: failed to store row buffer
      log_to_file("Init (Build): store_row_to_buffer returned EMPTY!");
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

  // 4. Flush remaining batched build keys to python server
  if (m_buffer_manager.FlushBatch()) {
    return true;  // error flushing last batch
  }

  if (m_buffer_manager.FlushControl("BUILD_DONE")) {
    return true; 
  }

  // 5. Switch to probe phase.
  m_buffer_manager.PopResult(); // sync build stream
  m_buffer_manager.SetStatus("PROBE");
  probe_idx = 0;

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

int SemJoinIterator::Read() {
  for (;;) {
    // Always try to read one probe row each Read() call
    int ret = m_probe_input->Read();
    if (ret == 1) {
      return 1;  // error
    }
    thd()->check_yield();

    if (ret == 0) {
      RequestRowId(m_probe_input_tables.tables(), m_tables_to_get_rowid_for);
      if (!extract_join_key_for_row(true)) {
        // Skip probe row with NULL join key, continue loop
        continue;
      }

      auto probe_row_buf = store_row_to_buffer(m_probe_input_tables, m_row_size);
      if (probe_row_buf.empty()) {
        return 1;  // error
      }

      m_probe_rows_queue.push(std::move(probe_row_buf));

      // Push the key-index pair to GPU buffer manager
      std::string key_copy(m_buffer.ptr(), m_buffer.length());
      KeyIndexPair pair{key_copy, probe_idx};
      probe_idx += 1;
      if (m_buffer_manager.PushTuple(pair)) {
        return 1;  // error pushing or launching kernel
      }
    } else if (ret == -1) {
      // Probe input exhausted, flush any remaining probe keys
      if (m_buffer_manager.FlushBatch()) {
        return 1;  // error flushing last batch
      }
    }

    // -------------------------------------------------------
    // Processing Results
    // -------------------------------------------------------
    auto result_pair_ptr = m_buffer_manager.PopResult();

    if (!result_pair_ptr) {
      if (ret == -1) return -1; // EOF
      continue; // Wait for results
    }

    size_t returned_probe_idx = result_pair_ptr->first;
    size_t matched_build_idx  = result_pair_ptr->second;

    // 1. Load Build Row (Must always be loaded as it likely changes)
    if (matched_build_idx >= build_rows_buffer.size()) {
        std::string err_msg = "Error: Returned build index out of bounds!";
        log_to_file(err_msg);
        return 1;
    }
    LoadIntoTableBuffers(m_build_input_tables, build_rows_buffer[matched_build_idx].data());

    // 2. Load Probe Row
    if (returned_probe_idx == m_current_loaded_probe_idx) {
        // HIT: Already loaded.
        // The data is safe because it lives in 'm_active_probe_row' from the previous call.
    } else {
        // MISS: Need to load new row.
        
        while (!m_probe_rows_queue.empty() && m_queue_front_global_idx < returned_probe_idx) {
            m_probe_rows_queue.pop();
            m_queue_front_global_idx++;
        }

        if (m_probe_rows_queue.empty() || m_queue_front_global_idx != returned_probe_idx) {
             // ... error handling ...
             return 1;
        }

        // FIX: Move into the CLASS MEMBER, not a local variable.
        // This destroys the previous active row (which is fine, we are done with it)
        // and persists the new one.
        m_active_probe_row = std::move(m_probe_rows_queue.front());
        m_probe_rows_queue.pop();
        
        // Load from the member variable
        LoadIntoTableBuffers(m_probe_input_tables, m_active_probe_row.data());
        
        m_current_loaded_probe_idx = returned_probe_idx;
        
        // m_queue_front_global_idx must be incremented because we popped one item
        m_queue_front_global_idx++;
    }
    return 0;
  }
}