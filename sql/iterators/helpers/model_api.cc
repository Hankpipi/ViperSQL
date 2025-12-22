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

static inline std::string sanitize_for_batch(const std::string &raw) {
  std::string out; out.reserve(raw.size());
  for (char c : raw) {
    unsigned char uc = static_cast<unsigned char>(c);
    // keep ':' so the model sees the identifier/text boundary
    if (std::isalpha(uc) || std::isdigit(uc) ||
        c == ',' || c == '.' || c == ' ' || c == '-' || c == '_' || c == ':') {
      out.push_back(c);
    }
  }
  // rtrim spaces
  while (!out.empty() && out.back() == ' ') out.pop_back();
  return out;
}

static inline std::string to_single_line(const std::string &s) {
  std::string out; out.reserve(s.size());
  for (char c : s) out.push_back((c=='\n' || c=='\r') ? ' ' : c);
  return out;
}

static void extract_array_of_strings(const std::string &text,
                                     std::vector<std::string> &out,
                                     size_t expected_count) {
  auto try_parse = [&](const std::string &payload) -> bool {
    try {
      json j = json::parse(payload);
      if (j.is_array()) {
        out.clear();
        out.reserve(j.size());
        for (const auto &e : j) {
          if (e.is_string()) out.push_back(e.get<std::string>());
          else if (e.is_number() || e.is_boolean()) out.push_back(e.dump());
          else if (e.is_null()) out.emplace_back("");
          else out.push_back(e.dump());
        }
        return true;
      }
    } catch (...) {}
    return false;
  };

  // First attempt: whole content
  if (try_parse(text)) return;

  // Second: bracket slice
  size_t l = text.find('[');
  if (l != std::string::npos) {
    // Find matching closing bracket naively (take last ']')
    size_t r = text.rfind(']');
    if (r != std::string::npos && r > l) {
      std::string slice = text.substr(l, r - l + 1);
      if (try_parse(slice)) return;
    }
  }

  // Fallback: line-based
  out.clear();
  std::istringstream iss(text);
  std::string line;
  while (std::getline(iss, line)) {
    // strip leading/trailing spaces and remove wrapping quotes if any
    auto beg = line.find_first_not_of(" \t\r\n");
    auto end = line.find_last_not_of(" \t\r\n");
    if (beg == std::string::npos) continue;
    std::string t = line.substr(beg, end - beg + 1);
    if (t.size() >= 2 && ((t.front()=='"' && t.back()=='"') || (t.front()=='\'' && t.back()=='\'')))
      t = t.substr(1, t.size()-2);
    if (!t.empty()) out.push_back(t);
    if (out.size() == expected_count) break;
  }
}

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
  const std::string* host_prompts = static_cast<const std::string*>(host_data);
  m_prompts.assign(host_prompts, host_prompts + n_rows);

  // Token budget: we only need n_rows chars of output (0/1), add a small cushion.
  size_t expected_output_tokens = std::max<size_t>(n_rows, 64);

  std::ostringstream oss;
  oss << "You are a strict classifier.\n"
      << "You will receive " << n_rows << " INPUTs labeled Input(1.." << n_rows << "). "
      << "For each input, decide True=1 or False=0.\n"
      << "OUTPUT FORMAT:\n"
      << "Return ONLY one concatenated string of exactly " << n_rows
      << " characters consisting solely of digits '0' and '1' in order, "
      << "no spaces, no quotes, no newlines, no comments.\n\n"
      << "INPUTS:\n";

  for (size_t i = 0; i < n_rows; ++i) {
    std::string filtered = sanitize_for_batch(m_prompts[i]);
    oss << "Input(" << (i + 1) << "): " << filtered << "\n";
  }
  oss << "\nReturn the concatenated 0/1 string now.";

  std::string combined = oss.str();

  // Launch async LLM call
  const std::string api_key = get_openai_api_key();
  if (api_key.empty()) {
    log_to_file("LLMFilterHelper: missing OPENAI_API_KEY");
    return true;
  }

  m_future = std::async(std::launch::async,
                        [this, combined, api_key, expected_output_tokens]() {
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

bool LLMGenerateHelper::SubmitBatch(const void* host_data, size_t n_rows) {
  m_expected_count = n_rows;
  const std::string* host_prompts = static_cast<const std::string*>(host_data);
  m_prompts.assign(host_prompts, host_prompts + n_rows);

  // Design a strict batch prompt → JSON array of strings, length = n_rows.
  std::ostringstream oss;
  oss << "You are a precise text transformer and generator.\n"
      << "You will receive " << n_rows << " independent INPUTs labeled Input(1.." << n_rows << ").\n"
      << "Each Input(i) contains an Instruction followed by one or more Columns in the form "
        "\"ColumnName: value.\" Execute the Instruction using ONLY the provided column values "
        "(e.g., rewrite, translate, grammar check, change casing, summarize, or other text transforms). "
        "Do not echo column names in the output unless the Instruction explicitly requests it.\n"
      << "OUTPUT FORMAT (MANDATORY):\n"
      << "1) Return ONLY a valid JSON array of " << n_rows << " strings, in order.\n"
      << "2) Do not include any keys, labels, comments, or code fences.\n"
      << "3) Each output must be a single line (no newline characters; use spaces instead).\n"
      << "4) If an input is malformed or cannot be satisfied, output an empty string \"\" in that position.\n\n"
      << "INPUTS:\n";
  for (size_t i = 0; i < n_rows; ++i) {
    // Keep the input small & consistent
    std::string filtered = sanitize_for_batch(m_prompts[i]);
    oss << "Input(" << (i+1) << "): " << filtered << "\n";
  }
  oss << "\nReturn the JSON array now.";

  std::string combined = oss.str();

  // Token budget: ~128 tokens per output item (adjustable).
  size_t expected_output_tokens = std::min<size_t>(n_rows * 128, 8192);

  const std::string api_key = get_openai_api_key();
  if (api_key.empty()) {
    log_to_file("LLMGenerateHelper: missing OPENAI_API_KEY");
    return true;
  }

  m_future = std::async(std::launch::async, [this, combined, api_key, expected_output_tokens]() {
    m_raw_response = call_openai_api(combined, api_key, expected_output_tokens);
  });

  return false;
}

bool LLMGenerateHelper::Synchronize() {
  if (m_future.valid()) m_future.wait();
  return false;
}

bool LLMGenerateHelper::FetchResults(void* out_buffer, size_t* out_result_count) {
  if (m_future.valid()) m_future.wait();

  // out_buffer points to an array of std::string with capacity >= m_batch_size
  auto *out = static_cast<std::string*>(out_buffer);
  const size_t N = m_expected_count;

  auto emit_empties = [&](void) {
    if (out) {
      for (size_t i = 0; i < N; ++i) out[i].clear();
    }
    if (out_result_count) *out_result_count = N;
    return false;
  };

  // SIMPLE ERROR CHECK: e.g. {"error":{"message":"User not found.","code":401}}
  if (m_raw_response.find("\"error\":{\"message\":") != std::string::npos) {
    log_to_file("LLMGenerateHelper: provider error payload detected; emitting empties");
    return emit_empties();
  }

  // Parse array-of-strings
  m_results.clear();
  m_results.reserve(N);
  extract_array_of_strings(m_raw_response, m_results, N);

  // Fallback to empties if nothing parsed
  if (m_results.empty()) {
    log_to_file("LLMGenerateHelper: no parsable items; emitting empties");
    return emit_empties();
  }

  // Normalize to single line + cap length
  constexpr size_t kMaxOutChars = 2048;
  for (std::string &s : m_results) {
    s = to_single_line(s);
    if (s.size() > kMaxOutChars) s.resize(kMaxOutChars);
  }

  // Enforce exactly N outputs
  if (m_results.size() < N) m_results.resize(N, "");
  else if (m_results.size() > N) m_results.resize(N);

  // Copy/move into the caller's array
  if (out) {
    for (size_t i = 0; i < N; ++i) out[i] = std::move(m_results[i]);
  }
  if (out_result_count) *out_result_count = N;

  return false;
}

void LLMGenerateHelper::Destroy() {
  if (m_future.valid()) {
    // best-effort wait to avoid dangling
    m_future.wait();
  }
  m_prompts.clear();
  m_results.clear();
  m_raw_response.clear();
  m_expected_count = 0;
}

void LLMGenerateHelper::SetStatus(const std::string& status) {
  log_to_file("LLMGenerateHelper: " + status);
}

} // namespace llmhelpers