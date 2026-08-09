#include "sql/iterators/helpers/sem_join_helper.h"

#include <algorithm>
#include <cstdint>
#include <future>
#include <limits>
#include <nlohmann/json.hpp>
#include <random>
#include <sstream>
#include <unordered_set>
#include <utility>
#include <vector>

#include "zmq_rpc_api.h"

namespace semhelpers {
namespace {

using ResultPair = std::pair<size_t, size_t>;

std::string GenRandomJoinId() {
  static thread_local std::mt19937_64 rng{std::random_device{}()};
  std::uniform_int_distribution<uint64_t> distribution;

  const uint64_t first = distribution(rng);
  const uint64_t second = distribution(rng);

  std::ostringstream stream;
  stream << std::hex << first << second;
  return stream.str();
}

bool ResponseSucceeded(const nlohmann::json &response) {
  return response.is_object() && response.contains("ok") &&
         response["ok"].is_boolean() && response["ok"].get<bool>();
}

bool ParseIndex(const nlohmann::json &value, size_t *index) {
  uint64_t parsed = 0;
  if (value.is_number_unsigned()) {
    parsed = value.get<uint64_t>();
  } else if (value.is_number_integer()) {
    const int64_t signed_value = value.get<int64_t>();
    if (signed_value < 0) return false;
    parsed = static_cast<uint64_t>(signed_value);
  } else {
    return false;
  }
  if (parsed > std::numeric_limits<size_t>::max()) return false;
  *index = static_cast<size_t>(parsed);
  return true;
}

}  // namespace

SemJoinHelper::SemJoinHelper(std::string model_name, std::string predicate)
    : m_model_name(std::move(model_name)),
      m_predicate(std::move(predicate)),
      m_join_id(GenRandomJoinId()) {}

SemJoinHelper::~SemJoinHelper() { Destroy(); }

bool SemJoinHelper::Init() {
  if (m_future.valid()) m_future.wait();
  m_results.clear();
  m_failed = false;
  m_status.clear();
  return false;
}

bool SemJoinHelper::SubmitBatch(const void *host_data, size_t n_rows) {
  m_results.clear();
  m_failed = false;

  if (m_status == "BUILD") return SubmitBuildBatch(host_data, n_rows);
  if (m_status == "BUILD_DONE") return SubmitBuildDone();
  if (m_status == "PROBE") return SubmitProbeBatch(host_data, n_rows);
  if (m_status == "RESET") return SubmitReset();

  m_failed = true;
  return true;
}

bool SemJoinHelper::SubmitBuildBatch(const void *host_data, size_t n_rows) {
  if (n_rows != 0 && host_data == nullptr) {
    m_failed = true;
    return true;
  }

  std::vector<KeyIndexPair> values;
  if (n_rows != 0) {
    const auto *pairs = static_cast<const KeyIndexPair *>(host_data);
    values.assign(pairs, pairs + n_rows);
  }

  const std::string name = m_model_name;
  const std::string predicate = m_predicate;
  const std::string join_id = m_join_id;
  try {
    m_future = std::async(
        std::launch::async,
        [this, name, predicate, values = std::move(values), join_id]() {
          try {
            const nlohmann::json response = semantic_join_zmq_rpc_call(
                name, values, predicate, "build", join_id);
            if (!ResponseSucceeded(response)) m_failed = true;
          } catch (...) {
            m_failed = true;
          }
        });
  } catch (...) {
    m_failed = true;
    return true;
  }
  return false;
}

bool SemJoinHelper::SubmitProbeBatch(const void *host_data, size_t n_rows) {
  if (n_rows != 0 && host_data == nullptr) {
    m_failed = true;
    return true;
  }

  std::vector<KeyIndexPair> values;
  if (n_rows != 0) {
    const auto *pairs = static_cast<const KeyIndexPair *>(host_data);
    values.assign(pairs, pairs + n_rows);
  }

  const std::string name = m_model_name;
  const std::string predicate = m_predicate;
  const std::string join_id = m_join_id;
  try {
    m_future = std::async(
        std::launch::async,
        [this, name, predicate, values = std::move(values), join_id]() {
          try {
            const nlohmann::json response = semantic_join_zmq_rpc_call(
                name, values, predicate, "probe", join_id);
            if (!ResponseSucceeded(response) || !response.contains("values") ||
                !response["values"].is_array()) {
              m_failed = true;
              return;
            }

            std::unordered_set<size_t> submitted_probe_indices;
            submitted_probe_indices.reserve(values.size());
            for (const KeyIndexPair &value : values) {
              submitted_probe_indices.insert(value.index);
            }

            std::vector<ResultPair> results;
            results.reserve(response["values"].size());
            for (const auto &entry : response["values"]) {
              if (!entry.is_array() || entry.size() != 2) {
                m_failed = true;
                return;
              }

              size_t build_index;
              size_t probe_index;
              if (!ParseIndex(entry[0], &build_index) ||
                  !ParseIndex(entry[1], &probe_index) ||
                  submitted_probe_indices.find(probe_index) ==
                      submitted_probe_indices.end()) {
                m_failed = true;
                return;
              }
              results.emplace_back(probe_index, build_index);
            }

            std::sort(results.begin(), results.end());
            if (std::adjacent_find(results.begin(), results.end()) !=
                results.end()) {
              m_failed = true;
              return;
            }
            m_results.swap(results);
          } catch (...) {
            m_failed = true;
          }
        });
  } catch (...) {
    m_failed = true;
    return true;
  }
  return false;
}

bool SemJoinHelper::SubmitBuildDone() {
  const std::string name = m_model_name;
  const std::string predicate = m_predicate;
  const std::string join_id = m_join_id;
  try {
    m_future =
        std::async(std::launch::async, [this, name, predicate, join_id]() {
          try {
            const std::vector<KeyIndexPair> empty;
            const nlohmann::json response = semantic_join_zmq_rpc_call(
                name, empty, predicate, "build_done", join_id);
            if (!ResponseSucceeded(response)) m_failed = true;
          } catch (...) {
            m_failed = true;
          }
        });
  } catch (...) {
    m_failed = true;
    return true;
  }
  return false;
}

bool SemJoinHelper::SubmitReset() {
  const std::string name = m_model_name;
  const std::string predicate = m_predicate;
  const std::string join_id = m_join_id;
  try {
    m_future =
        std::async(std::launch::async, [this, name, predicate, join_id]() {
          try {
            const std::vector<KeyIndexPair> empty;
            const nlohmann::json response = semantic_join_zmq_rpc_call(
                name, empty, predicate, "reset", join_id);
            if (!ResponseSucceeded(response)) m_failed = true;
          } catch (...) {
            m_failed = true;
          }
        });
  } catch (...) {
    m_failed = true;
    return true;
  }
  return false;
}

bool SemJoinHelper::FetchResults(void *out_buffer, size_t *out_result_count) {
  if (out_result_count == nullptr) {
    m_failed = true;
    return true;
  }
  *out_result_count = 0;

  if (Synchronize()) return true;
  if (m_status == "BUILD" || m_status == "BUILD_DONE" || m_status == "RESET") {
    return false;
  }
  if (m_status != "PROBE") {
    m_failed = true;
    return true;
  }
  if (m_results.empty()) return false;
  if (out_buffer == nullptr) {
    m_failed = true;
    return true;
  }

  auto *results = static_cast<ResultPair *>(out_buffer);
  std::copy(m_results.begin(), m_results.end(), results);
  *out_result_count = m_results.size();
  m_results.clear();
  return false;
}

bool SemJoinHelper::Synchronize() {
  if (m_future.valid()) m_future.wait();
  return m_failed;
}

void SemJoinHelper::Destroy() {
  if (m_future.valid()) m_future.wait();
  m_results.clear();
  m_failed = false;
  m_status.clear();
}

void SemJoinHelper::SetStatus(const std::string &status) { m_status = status; }

}  // namespace semhelpers
