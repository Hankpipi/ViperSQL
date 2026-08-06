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
  Implements a semantic-filter helper that sends batches of values
  and a predicate to the ZMQ Python Server.
*/
class LLMFilterHelper : public ExternalHelperInterface {
public:
  LLMFilterHelper();
  ~LLMFilterHelper() override;

  bool Init(size_t capacity) override;
  bool SubmitBatch(const void* host_data, size_t n_rows) override;
  bool FetchResults(void* out_buffer, size_t* out_result_count) override;
  size_t ResultBufferCapacity(size_t submitted_rows) const override {
    return m_results.size() > submitted_rows ? m_results.size() : submitted_rows;
  }
  bool Synchronize() override;
  bool IsIdle() const override {
    return !m_future.valid() ||
           m_future.wait_for(std::chrono::seconds(0)) ==
               std::future_status::ready;
  }
  void Destroy() override;
  void SetStatus(const std::string& status) override;

  void SetPredicate(const std::string& predicate);
  void SetModelName(const std::string& model_name);

protected:
  size_t                    m_capacity;       
  size_t                    m_expected_count; 
  std::vector<std::string>  m_prompts;        
  std::string               m_raw_response;   
  std::vector<uint8_t>      m_results;        
  std::future<void>         m_future;         
  
  std::string               m_predicate;
  std::string               m_model_name;
};

/**
  LLMTwoColFilterHelper
  Dedicated helper for SEMANTIC_FILTER_TWO_COL operations.
  Inherits generic fetch logic from LLMFilterHelper.
*/
class LLMTwoColFilterHelper : public LLMFilterHelper {
public:
  using LLMFilterHelper::LLMFilterHelper;
  bool SubmitBatch(const void* host_data, size_t n_rows) override;
};

/**
 * LLMGenerateHelper
 * Submits generation requests to the Python Server via ZMQ.
 */
class LLMGenerateHelper : public ExternalHelperInterface {
public:
  LLMGenerateHelper();
  ~LLMGenerateHelper();

  bool Init(size_t capacity);
  bool SubmitBatch(const void* host_data, size_t n_rows);
  bool Synchronize();
  bool IsIdle() const override {
    return !m_future.valid() ||
           m_future.wait_for(std::chrono::seconds(0)) ==
               std::future_status::ready;
  }
  bool FetchResults(void* out_buffer, size_t* out_result_count);
  void Destroy();
  void SetStatus(const std::string& status);

  void SetInstruction(const std::string& instruction);
  void SetModelName(const std::string& model_name);

private:
  size_t m_capacity{0};
  size_t m_expected_count{0};

  std::vector<std::string> m_prompts;
  std::vector<std::string> m_results;
  std::string m_raw_response;
  std::future<void> m_future;

  std::string m_instruction;
  std::string m_model_name;
};

}  // namespace llmhelpers

#endif  // SQL_ITERATORS_EXTERNAL_HELPERS_LLMHELPERS_API_H_
