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

#include "sem_filter_iterator.h"
#include "sql/item_func_semantic.h"

#include <utility>


// Returns extracted raw row buffer or empty vector on failure
// std::vector<uint8_t> store_row_to_buffer(const pack_rows::TableCollection& tables, size_t row_size) {
//   size_t row_size_upper_bound = row_size;
//   if (tables.has_blob_column()) {
//     row_size_upper_bound = ComputeRowSizeUpperBound(tables);
//   }

//   // Allocate buffer to hold the raw row bytes
//   std::vector<uint8_t> row_buffer(row_size_upper_bound);

//   // Copy raw row bytes from table buffers into row_buffer
//   uchar* dest = row_buffer.data();
//   dest = StoreFromTableBuffersRaw(tables, dest);

//   if (dest == nullptr) {
//     // Copy failed (e.g. OOM), return empty vector
//     return std::vector<uint8_t>();
//   }

//   // Resize to actual copied size
//   size_t actual_size = dest - row_buffer.data();
//   row_buffer.resize(actual_size);

//   return row_buffer;
// }

// External helpers you already provide elsewhere:
// size_t ComputeRowSizeUpperBound(const pack_rows::TableCollection&);
// std::vector<uint8_t> store_row_to_buffer(const pack_rows::TableCollection&, size_t);
// void LoadIntoTableBuffers(const pack_rows::TableCollection&, const uint8_t*);
// void log_to_file(const std::string&);

SemFilterIterator::SemFilterIterator(THD *thd,
                                     unique_ptr_destroy_only<RowIterator> source,
                                     pack_rows::TableCollection tables,
                                     Item *condition,
                                     size_t num_rows_estimate,
                                     AccessPath::Type impl_type)
  : RowIterator(thd),
    m_source(std::move(source)),
    m_tables(std::move(tables)),
    m_condition(condition),
    m_impl_type(impl_type),
    m_buffer_manager(64LL * 1024 * 1024,  // 64 MB staging
                     num_rows_estimate,
                     GetSemImplName(impl_type)) {}      // tag used by helper factory

bool SemFilterIterator::Init() {
  m_row_size = ComputeRowSizeUpperBound(m_tables);
  return m_source->Init();
}

int SemFilterIterator::Read() {
  for (;;) {
    int ret = m_source->Read();
    if (ret == 1)    // upstream error
      return 1;

    thd()->check_yield();

    if (ret == 0) {
      // pack current row into an in-memory buffer
      auto row_buf = store_row_to_buffer(m_tables, m_row_size);
      if (row_buf.empty()) {
        log_to_file("SemFilterIterator: failed to pack row");
        return 1;
      }
      m_rows_queue.push(std::move(row_buf));

      // get per-row value and current predicate
      auto *sf = static_cast<Item_func_sem_filter*>(m_condition);
      const std::string value = sf->compute_value();
      const std::string predicate = sf->compute_predicate(); // may be empty

      if (!predicate.empty() && predicate != m_last_predicate) {
        m_buffer_manager.SetStatus(predicate);
        m_last_predicate = predicate;
      }

      // submit the value (batched internally)
      if (m_buffer_manager.PushTuple(value)) {
        log_to_file("SemFilterIterator: PushTuple failed");
        return 1;
      }
    }
    else if (ret == -1) {
      if (m_buffer_manager.FlushBatch()) {
        log_to_file("SemFilterIterator: FlushBatch failed");
        return 1;
      }
    }

    // try to fetch one boolean result for the head of the queue
    auto res_ptr = m_buffer_manager.PopResult();
    if (!res_ptr) {
      // no result yet
      if (ret == -1)
        return -1; // EOF
      if (!m_buffer_manager.IsExternalCallRunning())
        continue;  // keep pulling upstream
      return 0;     
    }

    const bool matched = (*res_ptr != 0);
    auto row_buf = std::move(m_rows_queue.front());
    m_rows_queue.pop();

    if (!matched) {
      m_source->UnlockRow();
      continue;
    }

    LoadIntoTableBuffers(m_tables, row_buf.data());
    return 0;
  }
}
