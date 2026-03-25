#include "sql/iterators/helpers/model_api.h"
#include "zmq_rpc_api.h"
#include <nlohmann/json.hpp>
#include <future>
#include <algorithm>
#include <string>

using json = nlohmann::json;

namespace llmhelpers {

// ------------------------------------------------------------------
// LLMFilterHelper IMPLEMENTATION
// ------------------------------------------------------------------
LLMFilterHelper::LLMFilterHelper() : m_capacity(0) {}
LLMFilterHelper::~LLMFilterHelper() { Destroy(); }

bool LLMFilterHelper::Init(size_t capacity) {
  m_capacity = capacity;
  m_prompts.clear();
  m_results.clear();
  return false;
}

void LLMFilterHelper::SetPredicate(const std::string& predicate) { m_predicate = predicate; }
void LLMFilterHelper::SetModelName(const std::string& model_name) { m_model_name = model_name; }

bool LLMFilterHelper::SubmitBatch(const void* host_data, size_t n_rows) {
  m_expected_count = n_rows;
  const std::string* host_prompts = static_cast<const std::string*>(host_data);
  m_prompts.assign(host_prompts, host_prompts + n_rows);

  std::string name = m_model_name.empty() ? "sem_lite_llm_filter" : m_model_name;
  std::string predicate = m_predicate;

  m_future = std::async(std::launch::async, [this, name, predicate, values = m_prompts]() {
    try {
      nlohmann::json resp = semhelpers::semantic_task_zmq_rpc_call(name, values, "predicate", predicate);
      m_raw_response = resp.dump();

      if (resp.contains("ok") && resp["ok"].get<bool>()) {
        std::vector<uint8_t> temp_results;
        temp_results.reserve(values.size());
        
        if (resp.contains("values") && resp["values"].is_array()) {
          for (const auto& val : resp["values"]) {
            temp_results.push_back(val.get<int>() > 0 ? 1 : 0);
          }
        }
        m_results = std::move(temp_results);
      } else {
        m_results.assign(values.size(), 0);
      }
    } catch (...) {
      m_results.assign(values.size(), 0);
    }
  });

  return false;
}

bool LLMFilterHelper::Synchronize() {
  if (m_future.valid()) m_future.wait();
  return false;
}

bool LLMFilterHelper::FetchResults(void* out_buffer, size_t* out_result_count) {
  if (m_future.valid()) m_future.wait();

  if (m_results.size() < m_expected_count) {
    m_results.resize(m_expected_count, 0);
  }

  uint8_t* out = static_cast<uint8_t*>(out_buffer);
  std::copy(m_results.begin(), m_results.end(), out);
  
  *out_result_count = m_expected_count;
  return false;
}

void LLMFilterHelper::Destroy() {
  if (m_future.valid()) m_future.wait();
  m_prompts.clear();
  m_results.clear();
  m_raw_response.clear();
}

void LLMFilterHelper::SetStatus(const std::string& status) {}

// ------------------------------------------------------------------
// LLMTwoColFilterHelper IMPLEMENTATION
// ------------------------------------------------------------------
bool LLMTwoColFilterHelper::SubmitBatch(const void* host_data, size_t n_rows) {
  return LLMFilterHelper::SubmitBatch(host_data, n_rows);
}

// ------------------------------------------------------------------
// LLMGenerateHelper IMPLEMENTATION
// ------------------------------------------------------------------
LLMGenerateHelper::LLMGenerateHelper() : m_capacity(0), m_expected_count(0) {}
LLMGenerateHelper::~LLMGenerateHelper() { Destroy(); }

bool LLMGenerateHelper::Init(size_t capacity) {
  m_capacity = capacity;
  m_prompts.clear();
  m_results.clear();
  m_raw_response.clear();
  m_expected_count = 0;
  return false;
}

void LLMGenerateHelper::SetInstruction(const std::string& instruction) { m_instruction = instruction; }
void LLMGenerateHelper::SetModelName(const std::string& model_name) { m_model_name = model_name; }

bool LLMGenerateHelper::SubmitBatch(const void* host_data, size_t n_rows) {
  m_expected_count = n_rows;
  const std::string* host_prompts = static_cast<const std::string*>(host_data);
  m_prompts.assign(host_prompts, host_prompts + n_rows);

  std::string name = m_model_name.empty() ? "sem_generate" : m_model_name;
  std::string instruction = m_instruction;

  m_future = std::async(std::launch::async, [this, name, instruction, values = m_prompts]() {
    try {
      nlohmann::json resp = semhelpers::semantic_task_zmq_rpc_call(name, values, "instruction", instruction);
      m_raw_response = resp.dump();

      std::vector<std::string> temp_results;
      if (resp.contains("ok") && resp["ok"].get<bool>() && resp.contains("values") && resp["values"].is_array()) {
        for (const auto& val : resp["values"]) {
          temp_results.push_back(val.get<std::string>());
        }
      }
      m_results = std::move(temp_results);
    } catch (...) {
      m_results.assign(values.size(), "");
    }
  });

  return false;
}

bool LLMGenerateHelper::Synchronize() {
  if (m_future.valid()) m_future.wait();
  return false;
}

bool LLMGenerateHelper::FetchResults(void* out_buffer, size_t* out_result_count) {
  if (m_future.valid()) m_future.wait();

  auto *out = static_cast<std::string*>(out_buffer);
  const size_t N = m_expected_count;

  if (m_results.size() < N) {
      m_results.resize(N, "");
  } else if (m_results.size() > N) {
      m_results.resize(N);
  }

  if (out) {
    for (size_t i = 0; i < N; ++i) {
        out[i] = std::move(m_results[i]);
    }
  }
  
  if (out_result_count) {
      *out_result_count = N;
  }

  return false;
}

void LLMGenerateHelper::Destroy() {
  if (m_future.valid()) m_future.wait();
  m_prompts.clear();
  m_results.clear();
  m_raw_response.clear();
  m_expected_count = 0;
}

void LLMGenerateHelper::SetStatus(const std::string& status) {}

} // namespace llmhelpers