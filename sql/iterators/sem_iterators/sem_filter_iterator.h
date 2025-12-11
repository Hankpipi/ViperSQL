#ifndef SQL_ITERATORS_EXTERNAL_HELPERS_SEMHELPERS_SEM_FILTER_ITERATOR_H_
#define SQL_ITERATORS_EXTERNAL_HELPERS_SEMHELPERS_SEM_FILTER_ITERATOR_H_

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
#include <queue>
#include <string>
#include <vector>

#include "sql/iterators/row_iterator.h"
#include "sql/iterators/external_helper_buffer.h"
#include "sql/join_optimizer/access_path.h"
#include "sql/iterators/vectorized_iterators.h"
#include "sql/pack_rows.h"

// Forward decl of expression node used to compute value/predicate.
class Item_func_semantic_filter;

class SemFilterIterator final : public RowIterator {
public:
  SemFilterIterator(THD *thd,
                    unique_ptr_destroy_only<RowIterator> source,
                    pack_rows::TableCollection tables,
                    Item *condition,
                    size_t num_rows_estimate,
                    AccessPath::Type impl_type);

  bool Init() override;
  int  Read() override;

  void SetNullRowFlag(bool is_null_row) override {
    m_source->SetNullRowFlag(is_null_row);
  }
  void StartPSIBatchMode() override { m_source->StartPSIBatchMode(); }
  void EndPSIBatchModeIfStarted() override { m_source->EndPSIBatchModeIfStarted(); }
  void UnlockRow() override { m_source->UnlockRow(); }

private:
  AccessPath::Type m_impl_type;
  
  unique_ptr_destroy_only<RowIterator> m_source;
  pack_rows::TableCollection           m_tables;
  Item*                                m_condition;

  size_t                               m_row_size{0};
  std::queue<std::vector<uint8_t>>     m_rows_queue;

  // Batches std::string inputs and returns uint8_t (0/1) outputs.
  ExternalHelperBufferManager<std::string, uint8_t> m_buffer_manager;

  // Optional: last broadcast predicate (if buffer manager forwards SetStatus).
  std::string                          m_last_predicate;
};

#endif  // SQL_ITERATORS_EXTERNAL_HELPERS_SEMHELPERS_SEM_FILTER_ITERATOR_H_
