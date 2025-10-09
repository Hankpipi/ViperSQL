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


#include "sql/iterators/sem_helpers/join_helper.h"

#include <algorithm>
#include <utility>
#include <nlohmann/json.hpp>

#include "zmq_rpc_api.h"   // nlohmann::json sem_join_zmq_rpc_call(const nlohmann::json& req);
#include "utils/base64.h"  // std::string base64_encode(const std::string&)

#include "sql_string.h"
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>


#ifndef SEMJOIN_NOT_FOUND_SENTINEL
#define SEMJOIN_NOT_FOUND_SENTINEL 0xFFFFFFFFu
#endif



namespace semhelpers {


struct KeyIndexPair {
  std::string key; // Join key
  size_t index;  // Index of the full row in CPU build buffer
};

inline void to_json(nlohmann::json& j, const KeyIndexPair& p) {
  j = nlohmann::json::array({ static_cast<uint32_t>(p.index), base64_encode(p.key) });
}

static constexpr int MAX_KEY_SIZE = 32;
static constexpr size_t MIN_TABLE_CAPACITY = 1 << 20;
static constexpr size_t MAX_TABLE_CAPACITY = 1 << 27;
static constexpr uint32_t NOT_FOUND = 0xFFFFFFFF;

struct PackedKey {
  uint8_t data[MAX_KEY_SIZE];
};

struct HashEntry {
  PackedKey key;
  uint32_t index;
};


SemJoinHelper::SemJoinHelper(std::string model_name)
    : m_model_name(std::move(model_name)) {
  m_capacity = 0;
  m_expected_count = 0;
}

SemJoinHelper::~SemJoinHelper() {
  Destroy();
}

bool SemJoinHelper::Init(size_t capacity) {
  m_capacity = capacity;
  m_expected_count = 0;
  m_raw_response.clear();
  m_results.clear();
  m_status.clear();
  return false;
}

bool SemJoinHelper::SubmitBatch(const void* host_data, size_t n_rows) {
  if (m_status == "BUILD") {
    return SubmitBuildBatch(host_data, n_rows);
  } else if (m_status == "PROBE") {
    return SubmitProbeBatch(host_data, n_rows);
  } else {
    log_to_file("SemJoinHelper::SubmitBatch unknown status: " + m_status);
    return true;
  }
}

bool SemJoinHelper::SubmitBuildBatch(const void* host_data, size_t n_rows) {

  const KeyIndexPair* pairs = static_cast<const KeyIndexPair*>(host_data);

  std::vector<KeyIndexPair> values(pairs, pairs + n_rows);

  const std::string name = m_model_name;
  const std::string predicate = m_predicate; 
  const std::string type = "build";

  // build 阶段不返回匹配，仅作为 barrier；异步发起便于不阻塞
  m_expected_count = 0;
  m_future = std::async(std::launch::async, [this, name, values = std::move(values), predicate, type]() {
    try {
      nlohmann::json resp = sem_join_zmq_rpc_call(m_model_name, values, m_predicate, type);
      m_raw_response = resp.dump();
      // 如果服务端需要返回 build_id，可在此缓存（扩展字段）
      // if (resp.contains("build_id")) m_build_id = resp["build_id"].get<std::string>();
    } catch (...) {
      m_raw_response = "{}";
    }
  });

  return false;
}

bool SemJoinHelper::SubmitProbeBatch(const void* host_data, size_t n_rows) {

  const KeyIndexPair* pairs = static_cast<const KeyIndexPair*>(host_data);

  std::vector<KeyIndexPair> values(pairs, pairs + n_rows);

  const std::string name = m_model_name;
  const std::string predicate = m_predicate; 
  const std::string type = "probe";

  m_expected_count = n_rows;

  // 异步调用并把结果解析为 m_results（长度 = n_rows）
  m_future = std::async(std::launch::async, [this, name, values = std::move(values), predicate, type]() {
    nlohmann::json resp;
    try {
      resp = sem_join_zmq_rpc_call(m_model_name, values, m_predicate, type);;
      m_raw_response = resp.dump();
    } catch (...) {
      m_raw_response = "{}";
    }

    std::unordered_map<size_t, std::vector<size_t>> results(n_rows);
    if (resp.is_object() && resp.contains("values") && resp["values"].is_object()) {
      const auto& mp = resp["values"];
      for (auto it = mp.begin(); it != mp.end(); ++it) {
        size_t pid = 0;
        try {
          pid = static_cast<size_t>(std::stoull(it.key()));
        } catch (...) {
          continue; // 键非数字，跳过
        }
        if (!it.value().is_array()) continue; // 值不是列表，跳过

        auto& bids = results[pid];

        for (const auto& b : it.value()) {

          size_t bid = 0; try { bid = b.get<size_t>(); } catch (...) {}
          if (bid > 0) bids.push_back(bid);
        }
      }
    }
    m_results.swap(results);
  });

  return false;
}

bool SemJoinHelper::FetchResults(void* out_buffer, size_t* out_result_count) {
  if (!out_buffer || !out_result_count) return true;

  if (m_future.valid()) m_future.wait();

  if (m_status == "BUILD") {
    // 作为 barrier：不返回匹配
    *out_result_count = 0;
    return false;
  }

  // PROBE
  static_cast<std::unordered_map<size_t, std::vector<size_t>>*>(out_buffer).swap(m_results);
  *out_result_count = m_results.size();
  return false;
}

bool SemJoinHelper::Synchronize() {
  if (m_future.valid()) m_future.wait();
  return false;
}

void SemJoinHelper::Destroy() {
  if (m_future.valid()) m_future.wait();
  m_results.clear();
  m_raw_response.clear();
  m_expected_count = 0;
  m_status.clear();
}

void SemJoinHelper::SetStatus(const std::string& status) {
  m_status = status;
  log_to_file("SemJoinHelper: status=" + status);
}

void SemJoinHelper::SetPredicate(std::string predicate) {
  m_predicate = std::move(predicate);
}

void SemJoinHelper::SetModelName(std::string model_name) {
  m_model_name = std::move(model_name);
}

const std::string& SemJoinHelper::GetModelName() {
  return m_model_name;
}

} // namespace semhelpers

