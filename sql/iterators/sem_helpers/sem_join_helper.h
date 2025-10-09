#ifndef SQL_ITERATORS_EXTERNAL_HELPERS_SEMHELPERS_JOIN_HELPER_H_
#define SQL_ITERATORS_EXTERNAL_HELPERS_SEMHELPERS_JOIN_HELPER_H_

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

#include "sql/iterators/external_helper_interface.h"
#include <string>
#include <vector>
#include <future>
#include <cstddef>
#include <unordered_map>

namespace semhelpers {

/**
 * SemJoinHelper
 * A configurable semantic join helper that calls different backends
 * by switching the model name in the request JSON.
 */
class SemJoinHelper : public ExternalHelperInterface {
public:
  explicit SemJoinHelper(std::string model_name);
  ~SemJoinHelper() override;

  bool Init(size_t capacity) override;
  bool SubmitBatch(const void* host_data, size_t n_rows) override;
  bool FetchResults(void* out_buffer, size_t* out_result_count) override;
  bool Synchronize() override;
  void Destroy() override;
  void SetStatus(const std::string& status) override;

  void SetPredicate(std::string predicate) override;
  void SetModelName(std::string model_name) override;
  const std::string& GetModelName() override;

private:

  bool SubmitBuildBatch(const void* host_data, size_t n_rows);
  bool SubmitProbeBatch(const void* host_data, size_t n_rows);

  std::string          m_model_name;     // backend model name
  std::string          m_predicate;      // predicate string
  size_t               m_capacity{0};    // max batch size
  size_t               m_expected_count{0};
  std::string          m_raw_response;   // raw JSON (stringified) from server
  std::unordered_map<size_t, std::vector<size_t>> m_results;        // parsed int pair results
  std::future<void>    m_future;
  std::string          m_status;
};

}  // namespace semhelpers

#endif  // SQL_ITERATORS_EXTERNAL_HELPERS_SEMHELPERS_JOIN_HELPER_H_
