#include "sql/iterators/helpers/model_api.h"

#include <curl/curl.h>
#include <nlohmann/json.hpp>
#include <future>
#include <cstdlib>
#include <algorithm>
#include <cctype>
#include <string>
#include <sstream>

using json = nlohmann::json;

namespace llmhelpers {

// Helper to read environment variable
static std::string get_openai_api_key() {
  const char* key = std::getenv("OPENAI_API_KEY");
  return key ? std::string(key) : std::string();
}

// cURL write callback
static size_t WriteCallback(void* contents, size_t size, size_t nmemb, void* userp) {
  ((std::string*)userp)->append((char*)contents, size * nmemb);
  return size * nmemb;
}

// Single async call helper
static std::string call_openai_api(const std::string& prompt, const std::string& api_key) {
  CURL* curl = curl_easy_init();
  std::string readBuffer;
  if (!curl) return readBuffer;

  json payload = {
    {"model", "openai/gpt-4o-mini"},
    {"messages", {{{"role", "user"}, {"content", prompt}}}}
  };
  std::string payload_str = payload.dump();

  struct curl_slist* headers = nullptr;
  headers = curl_slist_append(headers, ("Authorization: Bearer " + api_key).c_str());
  headers = curl_slist_append(headers, "Content-Type: application/json");

  curl_easy_setopt(curl, CURLOPT_URL, "https://openrouter.ai/api/v1/chat/completions");
  curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
  curl_easy_setopt(curl, CURLOPT_POSTFIELDS, payload_str.c_str());
  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
  curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readBuffer);
  curl_easy_perform(curl);

  curl_slist_free_all(headers);
  curl_easy_cleanup(curl);

  // parse JSON
  try {
    auto resp = json::parse(readBuffer);
    return resp["choices"][0]["message"]["content"].get<std::string>();
  } catch (...) {
    return readBuffer;
  }
}

LLMFilterHelper::LLMFilterHelper() : m_capacity(0) {}
LLMFilterHelper::~LLMFilterHelper() { Destroy(); }

bool LLMFilterHelper::Init(size_t capacity) {
  m_capacity = capacity;
  m_prompts.clear();
  m_results.clear();
  return false;
}

bool LLMFilterHelper::SubmitBatch(const void* host_data, size_t n_rows) {
  // 1) Gather the N prompts
  const std::string* host_prompts = static_cast<const std::string*>(host_data);
  m_prompts.assign(host_prompts, host_prompts + n_rows);

  // 2) Build a JSON‐output prompt
  std::ostringstream oss;
  oss << "You will be given " << n_rows << " questions.  "
      << "Answer with a JSON array of booleans (true/false) of length "
      << n_rows << ", in plain text—*no* markdown or code fences.\nQuestions:\n";
  for (size_t i = 0; i < n_rows; ++i) {
    oss << i+1 << ". " << m_prompts[i] << "\n";
  }
  std::string combined = oss.str();

  // 3) Fire off the async LLM call
  std::string api_key = get_openai_api_key();
  if (api_key.empty()) {
    log_to_file("LLMFilterHelper: missing OPENAI_API_KEY");
    return true;
  }
  m_future = std::async(std::launch::async, [this, combined, api_key]() {
    m_raw_response = call_openai_api(combined, api_key);
  });

  return false;
}

bool LLMFilterHelper::Synchronize() {
  if (m_future.valid()) {
    m_future.wait();
  }
  return false;
}

bool LLMFilterHelper::FetchResults(void* out_buffer, size_t* out_result_count) {
  m_results.clear();
  try {
    auto j = json::parse(m_raw_response);
    if (j.is_array()) {
      // Copy up to N values
      for (size_t i = 0; i < j.size() && i < m_prompts.size(); ++i) {
        m_results.push_back(j[i].get<bool>() ? 1 : 0);
      }
    } else {
      log_to_file("LLMFilterHelper: JSON was not an array, cannot parse directly.");
    }
  } catch (const json::exception &e) {
    log_to_file(std::string("LLMFilterHelper: JSON parse error: ") + e.what());
  }

  // 4) Pad or truncate so we always have exactly N entries
  if (m_results.size() < m_prompts.size()) {
    m_results.resize(m_prompts.size(), 0);
  } else if (m_results.size() > m_prompts.size()) {
    m_results.resize(m_prompts.size());
  }

  // 5) Copy out
  uint8_t* out = static_cast<uint8_t*>(out_buffer);
  for (size_t i = 0; i < m_results.size(); ++i) {
    out[i] = m_results[i];
  }
  *out_result_count = m_results.size();
  return false;
}

void LLMFilterHelper::Destroy() {
  m_prompts.clear();
  m_results.clear();
  m_raw_response.clear();
}

void LLMFilterHelper::SetStatus(const std::string& status) {
  log_to_file("LLMFilterHelper: " + status);
}

} // namespace llmhelpers
