#include "sql/iterators/helpers/sem_join_helper.h"

#include <algorithm>
#include <utility>
#include <vector>
#include <future>
#include <nlohmann/json.hpp>
#include <random>
#include <sstream>

#include "zmq_rpc_api.h"
#include "sql_string.h"

namespace semhelpers {

// Ensure this matches the definition in your Iterator/BufferManager
using ResultPair = std::pair<size_t, size_t>;

static std::string GenRandomJoinId() {
  static thread_local std::mt19937_64 rng{std::random_device{}()};
  std::uniform_int_distribution<uint64_t> dist;

  uint64_t a = dist(rng);
  uint64_t b = dist(rng);

  std::ostringstream oss;
  oss << std::hex << a << b;
  return oss.str();
}

SemJoinHelper::SemJoinHelper(std::string model_name)
    : m_model_name(std::move(model_name)),
      m_join_id(GenRandomJoinId()) {
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
  m_results.clear(); // This is now vector<pair<size_t, size_t>>
  m_status.clear();
  return false;
}

bool SemJoinHelper::SubmitBatch(const void* host_data, size_t n_rows) {
  if (m_status == "BUILD") {
    return SubmitBuildBatch(host_data, n_rows);
  } else if (m_status == "BUILD_DONE") {
    return SubmitBuildDone();
  } else if (m_status == "PROBE") {
    return SubmitProbeBatch(host_data, n_rows);
  } else if (m_status == "RESET") {
    return SubmitReset();
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
  const std::string join_id = m_join_id;

  // Build phase acts as an upload barrier. No results returned immediately.
  m_expected_count = 0;
  
  m_future = std::async(std::launch::async, [this, name, values = std::move(values), predicate, type, join_id]() {
    try {
      nlohmann::json resp = semantic_join_zmq_rpc_call(name, values, predicate, type, join_id);
      m_raw_response = resp.dump();
    } catch (const std::exception& e) {
      m_raw_response = "{}";
      log_to_file("SemJoinHelper::SubmitBuildBatch Exception: " + std::string(e.what()));
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
  const std::string join_id = m_join_id;

  // In Probe phase, we expect results corresponding to these rows
  m_expected_count = n_rows;

  m_future = std::async(std::launch::async, [this, name, values = std::move(values), predicate, type, join_id]() {
    nlohmann::json resp;
    try {
      resp = semantic_join_zmq_rpc_call(name, values, predicate, type, join_id);
      // Optional: Store raw response for debug
      // m_raw_response = resp.dump(); 
    } catch (const std::exception& e) {
        log_to_file("SemJoinHelper::SubmitProbeBatch RPC Exception: " + std::string(e.what()));
        return;
    } catch (...) {
        return;
    }

    // Temporary vector to hold flattened results
    std::vector<ResultPair> temp_results;

    if (resp.is_object() && resp.contains("values") && resp["values"].is_array()) {
      const auto& arr = resp["values"];
    
      for (const auto& e : arr) {
        if (!e.is_array() || e.size() < 2) continue;
      
        size_t pid = 0, bid = 0;
        try {
          bid = e.at(0).get<size_t>();
          pid = e.at(1).get<size_t>();
        } catch (...) {
          continue;
        }
      
        temp_results.emplace_back(pid, bid);
      }
    }

    // CRITICAL: Sort results by Probe Index (first) then Build Index (second).
    std::sort(temp_results.begin(), temp_results.end());

    // Swap into the member variable for FetchResults to pick up
    // Note: We use a mutex if FetchResults can be called concurrently, 
    // but BufferManager usually waits for Synchronize() first.
    m_results.swap(temp_results);

  });

  return false;
}

bool SemJoinHelper::SubmitBuildDone() {
  const std::string name = m_model_name;
  const std::string predicate = m_predicate;
  const std::string type = "build_done";
  const std::string join_id = m_join_id;

  m_expected_count = 0;

  m_future = std::async(std::launch::async, [this, name, predicate, type, join_id]() {
    try {
      std::vector<KeyIndexPair> empty;
      nlohmann::json resp = semantic_join_zmq_rpc_call(name, empty, predicate, type, join_id);
      m_raw_response = resp.dump();
    } catch (const std::exception& e) {
      m_raw_response = "{}";
      log_to_file("SemJoinHelper::SubmitBuildDone Exception: " + std::string(e.what()));
    } catch (...) {
      m_raw_response = "{}";
    }
  });

  return false;
}

bool SemJoinHelper::SubmitReset() {
  const std::string name = m_model_name;
  const std::string predicate = m_predicate;
  const std::string type = "reset";
  const std::string join_id = m_join_id;

  m_expected_count = 0;

  m_future = std::async(std::launch::async, [this, name, predicate, type, join_id]() {
    try {
      std::vector<KeyIndexPair> empty;
      nlohmann::json resp = semantic_join_zmq_rpc_call(name, empty, predicate, type, join_id);
      m_raw_response = resp.dump();
    } catch (const std::exception& e) {
      m_raw_response = "{}";
      log_to_file("SemJoinHelper::SubmitReset Exception: " + std::string(e.what()));
    } catch (...) {
      m_raw_response = "{}";
    }
  });

  return false;
}

bool SemJoinHelper::FetchResults(void* out_buffer, size_t* out_result_count) {
  if (!out_buffer || !out_result_count) return true;

  // Ensure async task is done
  if (m_future.valid()) m_future.wait();

  if (m_status == "BUILD" || m_status == "BUILD_DONE" || m_status == "RESET") {
    // Build phase returns nothing
    *out_result_count = 0;
    return false;
  }

  // PROBE Phase
  if (m_results.empty()) {
      *out_result_count = 0;
      return false;
  }

  // 1. Cast output buffer to the expected Pair type
  ResultPair* buffer_ptr = static_cast<ResultPair*>(out_buffer);

  // 2. Copy data (Flat copy)
  // Since std::vector stores data contiguously, we can technically use memcpy,
  // but std::copy is safer for types.
  std::copy(m_results.begin(), m_results.end(), buffer_ptr);

  *out_result_count = m_results.size();
  
  // 3. Clear results to avoid re-reading
  m_results.clear();

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

void SemJoinHelper::SetJoinId(std::string join_id) { 
  m_join_id = std::move(join_id); 
}

const std::string& SemJoinHelper::GetJoinId() const { 
  return m_join_id; 
}

} // namespace semhelpers