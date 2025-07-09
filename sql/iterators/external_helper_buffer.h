#ifndef SQL_ITERATORS_EXTERNAL_HELPER_BUFFER_H_
#define SQL_ITERATORS_EXTERNAL_HELPER_BUFFER_H_

#include <vector>
#include <queue>
#include <string>
#include <memory>
#include "sql/iterators/gpu_helpers/gpu_hash_join.h"

// Forward declaration of the GPU helper interface
class ExternalHelperInterface;

/// Template class managing CPU buffers, GPU helper calls, and result queue.
/// TupleType: type of tuples sent to GPU (e.g., build/probe rows)
/// ResultType: type of results returned from GPU (e.g., matched indices)
template <typename TupleType, typename ResultType>
class ExternalHelperBufferManager {
public:
  ExternalHelperBufferManager(
      size_t max_memory_available, size_t estimated_rows, 
      const std::string& helper_name);

  ~ExternalHelperBufferManager();

  bool PushTuple(const TupleType& tuple);

  bool FlushBatch();

  std::unique_ptr<ResultType> PopResult();

  void SetStatus(const std::string& status);

  bool IsExternalCallRunning() const;

private:
  bool FetchAndQueueResults();

  std::vector<TupleType> m_input_buffer;
  std::queue<std::unique_ptr<ResultType>> m_result_queue;

  size_t m_batch_size;
  size_t m_max_memory_bytes;
  size_t m_estimated_rows;
  std::string m_helper_name;

  std::unique_ptr<ExternalHelperInterface> m_gpu_helper;

  bool m_external_call_running = false;
};

// Implementation

template <typename TupleType, typename ResultType>
ExternalHelperBufferManager<TupleType, ResultType>::ExternalHelperBufferManager(
    size_t max_memory_available, size_t estimated_rows, const std::string& helper_name)
    : m_max_memory_bytes(max_memory_available),
      m_estimated_rows(estimated_rows),
      m_helper_name(helper_name),
      m_external_call_running(false) {
  m_batch_size = BATCH_SIZE;

  if (helper_name == "GPUHashJoinHelper") {
    m_gpu_helper = std::make_unique<gpuhashjoinhelpers::GPUHashJoinHelper>();
  } else {
    log_to_file("Unknown helper: " + helper_name);
    m_gpu_helper = nullptr;
  }

  if (m_gpu_helper) {
    if (m_gpu_helper->Init(m_estimated_rows)) {
      log_to_file("Failed to initialize helper: " + helper_name);
      m_gpu_helper.reset();
    }
  }
}

template <typename TupleType, typename ResultType>
ExternalHelperBufferManager<TupleType, ResultType>::~ExternalHelperBufferManager() {
  if (m_gpu_helper) {
    m_gpu_helper->Destroy();
  }
}

template <typename TupleType, typename ResultType>
bool ExternalHelperBufferManager<TupleType, ResultType>::FetchAndQueueResults() {
  if (!m_gpu_helper) {
    log_to_file("GPU helper not initialized in FetchAndQueueResults");
    return true;
  }

  if (!m_external_call_running) {
    return false;
  }

  if (m_gpu_helper->Synchronize()) {
    log_to_file("Failed to synchronize GPU in FetchAndQueueResults");
    return true;
  }

  std::vector<ResultType> results_buffer(m_batch_size);
  size_t results_count = 0;

  if (m_gpu_helper->FetchResults(results_buffer.data(), &results_count)) {
    log_to_file("Failed to fetch results from GPU in FetchAndQueueResults");
    return true;
  }

  for (size_t i = 0; i < results_count; i++) {
    m_result_queue.push(std::make_unique<ResultType>(std::move(results_buffer[i])));
  }

  m_external_call_running = false;

  return false;
}

template <typename TupleType, typename ResultType>
bool ExternalHelperBufferManager<TupleType, ResultType>::PushTuple(const TupleType& tuple) {
  if (!m_gpu_helper) {
    log_to_file("GPU helper not initialized in PushTuple");
    return true;
  }

  m_input_buffer.push_back(tuple);

  if (m_input_buffer.size() >= m_batch_size) {
    if (FetchAndQueueResults()) {
      return true;
    }
    if (m_gpu_helper->SubmitBatch(m_input_buffer.data(), m_input_buffer.size())) {
      log_to_file("Failed to submit batch to GPU in PushTuple");
      return true;
    }
    m_input_buffer.clear();
    m_external_call_running = true;
  }

  return false;
}

template <typename TupleType, typename ResultType>
bool ExternalHelperBufferManager<TupleType, ResultType>::FlushBatch() {
  if (!m_gpu_helper) {
    log_to_file("GPU helper not initialized in FlushBatch");
    return true;
  }

  if (FetchAndQueueResults()) {
    return true;
  }

  if (m_input_buffer.empty()) {
    return false;
  }

  if (m_gpu_helper->SubmitBatch(m_input_buffer.data(), m_input_buffer.size())) {
    log_to_file("Failed to submit batch in FlushBatch");
    return true;
  }

  m_input_buffer.clear();
  m_external_call_running = true;

  return false;
}

template <typename TupleType, typename ResultType>
std::unique_ptr<ResultType> ExternalHelperBufferManager<TupleType, ResultType>::PopResult() {
  if (m_result_queue.empty()) {
    if (FetchAndQueueResults()) {
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
void ExternalHelperBufferManager<TupleType, ResultType>::SetStatus(const std::string& status) {
  if (m_gpu_helper) {
    m_gpu_helper->SetStatus(status);
  }
}

template <typename TupleType, typename ResultType>
bool ExternalHelperBufferManager<TupleType, ResultType>::IsExternalCallRunning() const {
  return m_external_call_running;
}

#endif  // SQL_ITERATORS_EXTERNAL_HELPER_BUFFER_H_
