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

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <future>
#include <string>
#include <utility>
#include <vector>

#include "sql/iterators/external_helper_interface.h"

namespace semhelpers {

struct KeyIndexPair {
  std::string key;
  size_t index;
};

class SemJoinHelper : public ExternalHelperInterface {
 public:
  SemJoinHelper(std::string model_name, std::string predicate);
  ~SemJoinHelper() override;

  bool Init() override;
  bool SubmitBatch(const void *host_data, size_t n_rows) override;
  bool FetchResults(void *out_buffer, size_t *out_result_count) override;
  size_t ResultBufferCapacity(size_t submitted_rows) const override {
    return std::max(m_results.size(), submitted_rows);
  }
  bool Synchronize() override;
  bool IsIdle() const override {
    return !m_future.valid() || m_future.wait_for(std::chrono::seconds(0)) ==
                                    std::future_status::ready;
  }
  void Destroy() override;
  void SetStatus(const std::string &status) override;

 private:
  bool SubmitBuildBatch(const void *host_data, size_t n_rows);
  bool SubmitProbeBatch(const void *host_data, size_t n_rows);
  bool SubmitBuildDone();
  bool SubmitReset();

  const std::string m_model_name;
  const std::string m_predicate;
  std::vector<std::pair<size_t, size_t>> m_results;
  std::future<void> m_future;
  bool m_failed{false};
  std::string m_status;
  const std::string m_join_id;
};

}  // namespace semhelpers

#endif  // SQL_ITERATORS_EXTERNAL_HELPERS_SEMHELPERS_JOIN_HELPER_H_
