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

// Picks model dynamically and issues the HTTP request.
// - If approx_tokens(prompt + expected output) ≤ 16000: use gpt-4o-mini
// - Otherwise: gpt-4.1-nano
static std::string call_openai_api(const std::string& prompt,
                                   const std::string& api_key,
                                   size_t expected_output_tokens) {
  // approx tokens = chars/4
  size_t approx_in = prompt.size() / 4;
  size_t total = approx_in + expected_output_tokens;
  const char* model =
      (total <= 12000) ? "openai/gpt-4o-mini" : "openai/gpt-4.1-nano";

  CURL* curl = curl_easy_init();
  std::string readBuffer;
  if (!curl) return readBuffer;

  // Build payload, include max_tokens
  json payload = {
    {"model", model},
    {"max_tokens", (int)expected_output_tokens},
    {"messages", {{{"role", "user"}, {"content", prompt}}}}
  };
  std::string payload_str = payload.dump();

  struct curl_slist* headers = nullptr;
  headers = curl_slist_append(headers, ("Authorization: Bearer " + api_key).c_str());
  headers = curl_slist_append(headers, "Content-Type: application/json");

  curl_easy_setopt(curl, CURLOPT_URL, "https://openrouter.ai/api/v1/chat/completions");
  curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
  curl_easy_setopt(curl, CURLOPT_POSTFIELDS, payload_str.c_str());
  curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, payload_str.size());
  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
  curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readBuffer);
  curl_easy_perform(curl);

  curl_slist_free_all(headers);
  curl_easy_cleanup(curl);

  // extract the assistant's content field
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
  // Record expected count
  m_expected_count = n_rows;
  size_t expected_output_tokens = n_rows;

  // Copy raw prompts
  const std::string* host_prompts = static_cast<const std::string*>(host_data);
  m_prompts.assign(host_prompts, host_prompts + n_rows);

  // Build the combined prompt with filtered text
  std::ostringstream oss;
  oss << "For each of the following " << n_rows
      << " questions, decide true (1) or false (0). "
      << "Output ONLY a single concatenated string of length "
      << n_rows << " consisting of the digits '0' and '1'—"
      << "no quotes, no spaces, no newlines, no commentary.\nQuestions:\n";

  for (size_t i = 0; i < n_rows; ++i) {
    // Filter out unwanted characters
    const std::string &raw = m_prompts[i];
    std::string filtered;
    filtered.reserve(raw.size());
    for (char c : raw) {
      if (std::isalpha(static_cast<unsigned char>(c)) ||
          c == ',' ||
          c == '.' ||
          c == ' ') {
        filtered.push_back(c);
      }
    }
    // Trim trailing spaces
    while (!filtered.empty() && filtered.back() == ' ') {
      filtered.pop_back();
    }
    oss << "Question(" << (i + 1) << "): " << filtered << "\n";
  }

  std::string combined = oss.str();

  // Launch async LLM call
  const char* key = std::getenv("OPENAI_API_KEY");
  if (!key) {
    log_to_file("LLMFilterHelper: missing OPENAI_API_KEY");
    return true;
  }
  std::string api_key(key);
  m_future = std::async(std::launch::async, [this, combined, api_key, expected_output_tokens]() {
    m_raw_response = call_openai_api(combined, api_key, expected_output_tokens);
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
  // wait for the async call if needed
  if (m_future.valid()) m_future.wait();

  // Parse m_raw_response for the first m_expected_count characters '0'/'1'
  m_results.clear();
  m_results.reserve(m_expected_count);
  for (char c : m_raw_response) {
    if (c == '0' || c == '1') {
      m_results.push_back(static_cast<uint8_t>(c - '0'));
      if (m_results.size() == m_expected_count) break;
    }
  }
  // Pad with 0 if too short
  if (m_results.size() < m_expected_count) {
    m_results.resize(m_expected_count, 0);
  }

  // Copy out
  uint8_t* out = static_cast<uint8_t*>(out_buffer);
  for (size_t i = 0; i < m_expected_count; ++i) out[i] = m_results[i];
  *out_result_count = m_expected_count;
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
