// model_api.h
#ifndef SQL_ITERATORS_EXTERNAL_HELPERS_LLMHELPERS_API_H_
#define SQL_ITERATORS_EXTERNAL_HELPERS_LLMHELPERS_API_H_

#include "sql/iterators/external_helper_interface.h"
#include <string>
#include <vector>
#include <cstddef>
#include <future>

namespace llmhelpers {

/**
  LLMFilterHelper
  Implements a semantic-filter helper that sends batches of prompts
  to an LLM backend and parses boolean responses.
*/
class LLMFilterHelper : public ExternalHelperInterface {
public:
  LLMFilterHelper();
  ~LLMFilterHelper() override;

  /// Initialize with the max number of prompts per batch
  bool Init(size_t capacity) override;

  /// Submit an array of std::string prompts as one combined batch
  bool SubmitBatch(const void* host_data, size_t n_rows) override;

  /// Fetch parsed boolean results for that batch
  bool FetchResults(void* out_buffer, size_t* out_result_count) override;

  /// Wait for the async LLM call to complete
  bool Synchronize() override;

  /// Clean up internal buffers
  void Destroy() override;

  /// Log status messages
  void SetStatus(const std::string& status) override;

protected:
  size_t                    m_capacity;       ///< batch size
  size_t                    m_expected_count; ///< last submit question count
  std::vector<std::string>  m_prompts;        ///< stored prompts
  std::string               m_raw_response;   ///< full LLM output
  std::vector<uint8_t>      m_results;        ///< parsed true/false per prompt
  std::future<void>         m_future;         ///< async handle for the LLM call
};

/**
  LLMTwoColFilterHelper
  Dedicated helper for SEMANTIC_FILTER_TWO_COL operations.
  Focuses on comparing two text segments (Consistency, Relevance, Equality).
  Inherits generic fetch/sync logic from LLMFilterHelper.
*/
class LLMTwoColFilterHelper : public LLMFilterHelper {
public:
  using LLMFilterHelper::LLMFilterHelper; // Inherit constructor
  
  /// Specialized batch submission for two-column comparison tasks
  bool SubmitBatch(const void* host_data, size_t n_rows) override;
};

/**
 * LLMGenerateHelper
 * - Submits a batch of N prompts (std::string each)
 * - Asynchronously calls the LLM
 * - Fetches a batch of N generated strings (1:1 with inputs)
 */
class LLMGenerateHelper : public ExternalHelperInterface {
public:
  LLMGenerateHelper();
  ~LLMGenerateHelper();

  // Prepare internal buffers. 'capacity' is kept for symmetry; not strictly required.
  bool Init(size_t capacity);

  // host_data must point to an array of std::string of length n_rows.
  bool SubmitBatch(const void* host_data, size_t n_rows);

  // Wait for the async request to finish (no-op if already done).
  bool Synchronize();

  // Writes results into *(std::vector<std::string>*)out_buffer.
  // On success, *out_result_count == number of strings produced (== n_rows).
  bool FetchResults(void* out_buffer, size_t* out_result_count);

  // Release internal buffers and join any outstanding async work.
  void Destroy();

  // Optional status hook for logging/diagnostics.
  void SetStatus(const std::string& status);

private:
  size_t m_capacity{0};
  size_t m_expected_count{0};

  std::vector<std::string> m_prompts;
  std::vector<std::string> m_results;

  std::string m_raw_response;

  // Async task that fills m_raw_response.
  std::future<void> m_future;
};

}  // namespace llmhelpers

#endif  // SQL_ITERATORS_EXTERNAL_HELPERS_LLMHELPERS_API_H_