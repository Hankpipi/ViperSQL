#ifndef SQL_ITERATORS_EXTERNAL_HELPER_BUFFER_H_
#define SQL_ITERATORS_EXTERNAL_HELPER_BUFFER_H_

#include <algorithm>
#include <vector>
#include <queue>
#include <string>
#include <memory>
#include "sql/iterators/helpers/gpu_hash_join.h"
#include "sql/iterators/helpers/model_api.h"
#include "sql/iterators/helpers/sem_join_helper.h"

// Forward declaration of the helper interface
class ExternalHelperInterface;

/// Template class managing CPU buffers, helper calls, and result queue.
/// TupleType: type of tuples sent to external helper (e.g., tuples)
/// ResultType: type of results returned from helper (e.g., matched indices)
template <typename TupleType, typename ResultType>
class ViperFlow {
public:
  ViperFlow(
      size_t max_memory_available, size_t estimated_rows, 
      const std::string& helper_name);

  ~ViperFlow();

  bool PushTuple(const TupleType& tuple);

  /// Push a tuple and report whether this call submitted a new helper batch.
  /// Iterators use this acknowledgement to enforce fill-before-pop scheduling.
  bool PushTuple(const TupleType& tuple, bool* did_submit);

  bool FlushBatch();

  bool FlushControl(const std::string& status);

  /// Drain any outstanding request and discard state from a prior execution.
  bool Reset();

  std::unique_ptr<ResultType> PopResult();

  /// Return whether PopResult() can complete without waiting for the helper.
  bool HasReadyResult() const;

  /// Return whether buffered input, an external call, or queued output remains.
  bool HasPendingWork() const;

  bool HasError() const;

  /// The paper's O(1) helper-idle test. One ViperFlow owns at most one call.
  bool IsIdle() const;

  /// Return whether an idle helper should be refilled before exposing output.
  /// A B_min output watermark prevents nested iterators from building an
  /// unbounded ready-result backlog while preserving fill-first at low supply.
  bool ShouldRefillBeforePop() const;

  void SetStatus(const std::string& status);

  bool IsExternalCallRunning() const;

private:
  bool FetchAndQueueResults();

  bool SubmitInputBuffer();

  std::vector<TupleType> m_input_buffer;
  std::queue<std::unique_ptr<ResultType>> m_result_queue;

  size_t m_batch_size;
  size_t m_min_batch_size;
  size_t m_max_memory_bytes;
  size_t m_estimated_rows;
  std::string m_helper_name;

  std::unique_ptr<ExternalHelperInterface> m_helper;

  bool m_external_call_running = false;
  bool m_failed = false;
  size_t m_in_flight_batch_size = 0;
};

// Implementation

template <typename TupleType, typename ResultType>
ViperFlow<TupleType, ResultType>::ViperFlow(
    size_t max_memory_available, size_t estimated_rows, const std::string& helper_name)
    : m_max_memory_bytes(max_memory_available),
      m_estimated_rows(estimated_rows),
      m_helper_name(helper_name),
      m_external_call_running(false) {
  m_batch_size = BATCH_SIZE;

  if (helper_name == "GPUHashJoinHelper") {
    m_batch_size = 1000;
    m_helper = std::make_unique<gpuhashjoinhelpers::GPUHashJoinHelper>(m_batch_size);
  }
  else if (helper_name == "semantic_filter") {
    m_helper = std::make_unique<llmhelpers::LLMFilterHelper>();
  }
  else if (helper_name == "semantic_filter_two_col") {
    m_helper = std::make_unique<llmhelpers::LLMTwoColFilterHelper>();
  }
  else if (helper_name == "semantic_generate") {
    m_helper = std::make_unique<llmhelpers::LLMGenerateHelper>();
  }
  else if (helper_name == "sem_llm_join") {
    m_helper = std::make_unique<semhelpers::SemJoinHelper>("sem_cascade_join");
  }
  else {
    log_to_file("Unknown helper: " + helper_name);
    m_helper = nullptr;
  }

  // B_min is the profiling threshold in the paper. Until profiling supplies
  // it, use the requested half-capacity policy and keep it valid for B_max=1.
  m_min_batch_size = std::max<size_t>(1, m_batch_size / 2);

  if (m_helper) {
    if (m_helper->Init(m_estimated_rows)) {
      log_to_file("Failed to initialize helper: " + helper_name);
      m_helper.reset();
    }
  }
}

template <typename TupleType, typename ResultType>
ViperFlow<TupleType, ResultType>::~ViperFlow() {
  if (m_helper) {
    m_helper->Destroy();
  }
}

template <typename TupleType, typename ResultType>
bool ViperFlow<TupleType, ResultType>::FetchAndQueueResults() {
  if (!m_helper) {
    log_to_file("helper not initialized in FetchAndQueueResults");
    return true;
  }

  if (!m_external_call_running) {
    return false;
  }

  if (m_helper->Synchronize()) {
    log_to_file("Failed to synchronize helper in FetchAndQueueResults");
    m_failed = true;
    m_external_call_running = false;
    m_in_flight_batch_size = 0;
    return true;
  }

  const size_t result_capacity = std::max(
      m_batch_size, m_helper->ResultBufferCapacity(m_in_flight_batch_size));
  std::vector<ResultType> results_buffer(result_capacity);
  size_t results_count = 0;

  if (m_helper->FetchResults(results_buffer.data(), &results_count)) {
    log_to_file("Failed to fetch results from helper in FetchAndQueueResults");
    m_failed = true;
    m_external_call_running = false;
    m_in_flight_batch_size = 0;
    return true;
  }

  if (results_count > results_buffer.size()) {
    m_failed = true;
    m_external_call_running = false;
    m_in_flight_batch_size = 0;
    return true;
  }

  for (size_t i = 0; i < results_count; i++) {
    m_result_queue.push(std::make_unique<ResultType>(std::move(results_buffer[i])));
  }

  m_external_call_running = false;
  m_in_flight_batch_size = 0;

  return false;
}

template <typename TupleType, typename ResultType>
bool ViperFlow<TupleType, ResultType>::SubmitInputBuffer() {
  const size_t actual_batch_size = m_input_buffer.size();
  if (m_helper->SubmitBatch(m_input_buffer.data(), actual_batch_size)) {
    m_failed = true;
    return true;
  }

  m_input_buffer.clear();
  m_external_call_running = true;
  m_in_flight_batch_size = actual_batch_size;
  return false;
}

template <typename TupleType, typename ResultType>
bool ViperFlow<TupleType, ResultType>::PushTuple(const TupleType& tuple) {
  return PushTuple(tuple, nullptr);
}

template <typename TupleType, typename ResultType>
bool ViperFlow<TupleType, ResultType>::PushTuple(const TupleType& tuple,
                                                bool* did_submit) {
  if (did_submit != nullptr) *did_submit = false;

  if (!m_helper || m_failed) {
    log_to_file("helper not initialized in PushTuple");
    return true;
  }

  m_input_buffer.push_back(tuple);

  const bool reached_max = m_input_buffer.size() >= m_batch_size;
  // Avoid polling the external helper for the first B_min - 1 tuples.
  const bool idle_at_min =
      m_input_buffer.size() >= m_min_batch_size && IsIdle();

  if (reached_max || idle_at_min) {
    if (FetchAndQueueResults()) {
      return true;
    }
    if (SubmitInputBuffer()) {
      return true;
    }
    if (did_submit != nullptr) *did_submit = true;
  }

  return false;
}

template <typename TupleType, typename ResultType>
bool ViperFlow<TupleType, ResultType>::FlushBatch() {
  if (!m_helper || m_failed) {
    log_to_file("helper not initialized in FlushBatch");
    return true;
  }

  if (FetchAndQueueResults()) {
    return true;
  }

  if (m_input_buffer.empty()) {
    return false;
  }

  // EOF is a correctness boundary, so residual input is submitted even when
  // it is smaller than B_min.
  return SubmitInputBuffer();
}

template <typename TupleType, typename ResultType>
bool ViperFlow<TupleType, ResultType>::FlushControl(const std::string& status) {
  if (!m_helper) {
    log_to_file("helper not initialized in FlushControl");
    return true;
  }

  if (m_external_call_running) {
    if (m_helper->Synchronize()) return true;
    if (FetchAndQueueResults()) return true;
    m_external_call_running = false;
  }

  m_helper->SetStatus(status);
  if (m_helper->SubmitBatch(nullptr, 0)) {
    log_to_file("Failed to submit control batch, status=" + status);
    return true;
  }
  m_external_call_running = true;
  m_in_flight_batch_size = 0;

  if (m_helper->Synchronize()) return true;
  if (FetchAndQueueResults()) return true;
  m_external_call_running = false;

  return false;
}

template <typename TupleType, typename ResultType>
bool ViperFlow<TupleType, ResultType>::Reset() {
  if (!m_helper) {
    return true;
  }

  if (m_external_call_running && FetchAndQueueResults()) return true;

  m_input_buffer.clear();
  while (!m_result_queue.empty()) m_result_queue.pop();
  m_external_call_running = false;
  m_failed = false;
  m_in_flight_batch_size = 0;
  return false;
}

template <typename TupleType, typename ResultType>
std::unique_ptr<ResultType> ViperFlow<TupleType, ResultType>::PopResult() {
  if (m_result_queue.empty()) {
    if (FetchAndQueueResults()) {
      log_to_file("FetchAndQueueResults fail");
      return nullptr;
    }
    if (m_result_queue.empty()) {
      return nullptr;
    }
  }

  auto res = std::move(m_result_queue.front());
  m_result_queue.pop();
  return res;
}

template <typename TupleType, typename ResultType>
bool ViperFlow<TupleType, ResultType>::HasReadyResult() const {
  return !m_result_queue.empty();
}

template <typename TupleType, typename ResultType>
bool ViperFlow<TupleType, ResultType>::HasPendingWork() const {
  return !m_input_buffer.empty() || m_external_call_running ||
         !m_result_queue.empty();
}

template <typename TupleType, typename ResultType>
bool ViperFlow<TupleType, ResultType>::HasError() const {
  return m_failed;
}

template <typename TupleType, typename ResultType>
bool ViperFlow<TupleType, ResultType>::IsIdle() const {
  return !m_external_call_running || (m_helper && m_helper->IsIdle());
}

template <typename TupleType, typename ResultType>
bool ViperFlow<TupleType, ResultType>::ShouldRefillBeforePop() const {
  // Check backpressure first so a sufficiently stocked output queue avoids an
  // external readiness probe altogether.
  return m_result_queue.size() < m_min_batch_size && IsIdle();
}

template <typename TupleType, typename ResultType>
void ViperFlow<TupleType, ResultType>::SetStatus(const std::string& status) {
  if (m_helper) {
    m_helper->SetStatus(status);
  }
}

template <typename TupleType, typename ResultType>
bool ViperFlow<TupleType, ResultType>::IsExternalCallRunning() const {
  return m_external_call_running;
}

#endif  // SQL_ITERATORS_EXTERNAL_HELPER_BUFFER_H_
