#ifndef SQL_ITERATORS_EXTERNAL_HELPERS_LLMHELPERS_API_H_
#define SQL_ITERATORS_EXTERNAL_HELPERS_LLMHELPERS_API_H_

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <future>
#include <string>
#include <vector>

#include "sql/iterators/external_helper_interface.h"

namespace llmhelpers {

/** Sends semantic-filter batches to the remote model service. */
class LLMFilterHelper : public ExternalHelperInterface {
 public:
  LLMFilterHelper() = default;
  ~LLMFilterHelper() override;

  bool Init() override;
  bool SubmitBatch(const void *host_data, size_t n_rows) override;
  bool FetchResults(void *out_buffer, size_t *out_result_count) override;
  double LastBatchServiceSeconds() const override {
    return static_cast<double>(m_last_service_time_ns.load()) / 1.0e9;
  }
  std::string LastServerEpoch() const override { return m_server_epoch; }
  bool Synchronize() override;
  bool IsIdle() const override {
    return !m_future.valid() || m_future.wait_for(std::chrono::seconds(0)) ==
                                    std::future_status::ready;
  }
  void Destroy() override;

 private:
  size_t m_expected_count{0};
  std::vector<uint8_t> m_results;
  std::future<void> m_future;
  bool m_failed{false};
  std::atomic<uint64_t> m_last_service_time_ns{0};
  std::string m_server_epoch;
};

}  // namespace llmhelpers

#endif  // SQL_ITERATORS_EXTERNAL_HELPERS_LLMHELPERS_API_H_
