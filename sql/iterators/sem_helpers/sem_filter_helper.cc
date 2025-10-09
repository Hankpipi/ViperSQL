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

#include "sem_filter_helper.h"
#include "zmq_rpc_api.h"                // semantic_filter_zmq_rpc_call
#include <nlohmann/json.hpp>
#include <unordered_set>
#include <utility>

namespace semhelpers {

SemFilterHelper::SemFilterHelper(std::string model_name)
    : m_model_name(std::move(model_name)) {
  m_capacity = 0;
  m_expected_count = 0;
}

SemFilterHelper::~SemFilterHelper() {
  Destroy();
}

bool SemFilterHelper::Init(size_t capacity) {
  m_capacity = capacity;
  m_results.clear();
  m_raw_response.clear();
  return false;
}

bool SemFilterHelper::SubmitBatch(const void* host_data, size_t n_rows) {
  // Save expected count and copy inputs.
  m_expected_count = n_rows;
  const std::string* src = static_cast<const std::string*>(host_data);
  std::vector<std::string> values(src, src + n_rows);

  const std::string name = m_model_name;
  const std::string predicate = m_predicate;  // explicit predicate

  // Fire async RPC and compute boolean mask.
  m_future = std::async(std::launch::async, [this, name, values = std::move(values), predicate]() {
    // Call JSON RPC (endpoint is fixed inside rpc_client).
    nlohmann::json resp = semantic_filter_zmq_rpc_call(name, values, predicate);

    // Keep raw JSON for diagnostics.
    m_raw_response = resp.dump();

    // Response schema: { "name": <...>, "values": [filtered subset] }
    std::vector<std::string> filtered;
    try {
      if (resp.is_object() && resp.contains("values") && resp["values"].is_array()) {
        filtered = resp["values"].get<std::vector<std::string>>();
      }
    } catch (...) {
      filtered.clear();
    }

    // Build boolean results by membership test against the filtered subset.
    std::unordered_set<std::string> keep(filtered.begin(), filtered.end());
    m_results.clear();
    m_results.reserve(m_expected_count);
    for (size_t i = 0; i < m_expected_count; ++i) {
      m_results.push_back(keep.count(values[i]) ? 1u : 0u);
    }
  });

  // Return false on success.
  return false;
}

bool SemFilterHelper::Synchronize() {
  if (m_future.valid()) m_future.wait();
  return false;
}

bool SemFilterHelper::FetchResults(void* out_buffer, size_t* out_result_count) {
  if (m_future.valid()) m_future.wait();
  if (!out_buffer || !out_result_count) return true;  // indicate error

  // Ensure size matches expectation (pad or trim).
  if (m_results.size() < m_expected_count) m_results.resize(m_expected_count, 0);
  if (m_results.size() > m_expected_count) m_results.resize(m_expected_count);

  auto* out = static_cast<uint8_t*>(out_buffer);
  for (size_t i = 0; i < m_expected_count; ++i) out[i] = m_results[i];
  *out_result_count = m_expected_count;
  return false;
}

void SemFilterHelper::Destroy() {
  if (m_future.valid()) m_future.wait();
  m_results.clear();
  m_raw_response.clear();
  m_expected_count = 0;
}

void SemFilterHelper::SetStatus(const std::string& status) {
  log_to_file("SemFilterHelper: " + status);
}

void SemFilterHelper::SetPredicate(std::string predicate) { 
    m_predicate = std::move(predicate); 
}
  
void SemFilterHelper::SetModelName(std::string model_name) { 
    m_model_name = std::move(model_name); 
}
const std::string& SemFilterHelper::GetModelName() const { 
    return m_model_name; 
}


} // namespace semhelpers
