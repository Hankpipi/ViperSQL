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

private:
  size_t                    m_capacity;       ///< batch size
  size_t                    m_expected_count; ///< last submit question count
  std::vector<std::string>  m_prompts;        ///< stored prompts
  std::string               m_raw_response;   ///< full LLM output
  std::vector<uint8_t>      m_results;        ///< parsed true/false per prompt
  std::future<void>         m_future;         ///< async handle for the LLM call
};

}  // namespace llmhelpers

#endif  // SQL_ITERATORS_EXTERNAL_HELPERS_LLMHELPERS_API_H_
