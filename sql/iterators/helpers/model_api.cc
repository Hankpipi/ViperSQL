#include "sql/iterators/helpers/model_api.h"

#include <algorithm>
#include <chrono>
#include <future>
#include <nlohmann/json.hpp>
#include <string>
#include <utility>
#include <vector>

#include "zmq_rpc_api.h"

namespace llmhelpers {
namespace {

bool ResponseSucceeded(const nlohmann::json &response) {
  return response.is_object() && response.contains("ok") &&
         response["ok"].is_boolean() && response["ok"].get<bool>();
}

bool ParseFilterResult(const nlohmann::json &value, uint8_t *result) {
  if (value.is_boolean()) {
    *result = value.get<bool>() ? 1 : 0;
    return true;
  }
  if (value.is_number_unsigned()) {
    const uint64_t integer = value.get<uint64_t>();
    if (integer > 1) return false;
    *result = static_cast<uint8_t>(integer);
    return true;
  }
  if (value.is_number_integer()) {
    const int64_t integer = value.get<int64_t>();
    if (integer < 0 || integer > 1) return false;
    *result = static_cast<uint8_t>(integer);
    return true;
  }
  return false;
}

}  // namespace

LLMFilterHelper::~LLMFilterHelper() { Destroy(); }

bool LLMFilterHelper::Init() {
  m_results.clear();
  m_expected_count = 0;
  m_failed = false;
  m_last_service_time_ns.store(0);
  m_server_epoch.clear();
  return false;
}

bool LLMFilterHelper::SubmitBatch(const void *host_data, size_t n_rows) {
  if (n_rows != 0 && host_data == nullptr) {
    m_expected_count = n_rows;
    m_results.clear();
    m_failed = true;
    return true;
  }

  m_expected_count = n_rows;
  m_failed = false;
  m_results.clear();
  m_last_service_time_ns.store(0);
  m_server_epoch.clear();
  std::vector<std::string> values;
  if (n_rows != 0) {
    const auto *host_prompts = static_cast<const std::string *>(host_data);
    values.assign(host_prompts, host_prompts + n_rows);
  }

  try {
    m_future = std::async(std::launch::async, [this,
                                               values = std::move(values)]() {
      const auto service_start = std::chrono::steady_clock::now();
      try {
        nlohmann::json resp = semhelpers::semantic_task_zmq_rpc_call(
            "sem_lite_llm_filter", values, "predicate", "");
        if (resp.contains("server_epoch") && resp["server_epoch"].is_string()) {
          m_server_epoch = resp["server_epoch"].get<std::string>();
        }

        if (!ResponseSucceeded(resp) || !resp.contains("values") ||
            !resp["values"].is_array() ||
            resp["values"].size() != values.size()) {
          m_failed = true;
          m_results.clear();
        } else {
          std::vector<uint8_t> temp_results;
          temp_results.reserve(values.size());
          bool valid_results = true;
          for (const auto &val : resp["values"]) {
            uint8_t result = 0;
            if (!ParseFilterResult(val, &result)) {
              valid_results = false;
              break;
            }
            temp_results.push_back(result);
          }
          if (valid_results) {
            m_results = std::move(temp_results);
          } else {
            m_failed = true;
            m_results.clear();
          }
        }
      } catch (...) {
        m_failed = true;
        m_results.clear();
      }
      const auto service_end = std::chrono::steady_clock::now();
      m_last_service_time_ns.store(static_cast<uint64_t>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(service_end -
                                                               service_start)
              .count()));
    });
  } catch (...) {
    m_failed = true;
    m_results.clear();
    return true;
  }

  return false;
}

bool LLMFilterHelper::Synchronize() {
  if (m_future.valid()) m_future.wait();
  return m_failed;
}

bool LLMFilterHelper::FetchResults(void *out_buffer, size_t *out_result_count) {
  if (m_future.valid()) m_future.wait();

  if (out_result_count == nullptr ||
      (m_expected_count != 0 && out_buffer == nullptr) || m_failed ||
      m_results.size() != m_expected_count) {
    if (out_result_count != nullptr) *out_result_count = 0;
    return true;
  }

  if (!m_results.empty()) {
    auto *out = static_cast<uint8_t *>(out_buffer);
    std::copy(m_results.begin(), m_results.end(), out);
  }

  *out_result_count = m_expected_count;
  return false;
}

void LLMFilterHelper::Destroy() {
  if (m_future.valid()) m_future.wait();
  m_results.clear();
  m_expected_count = 0;
  m_failed = false;
  m_last_service_time_ns.store(0);
  m_server_epoch.clear();
}

}  // namespace llmhelpers
