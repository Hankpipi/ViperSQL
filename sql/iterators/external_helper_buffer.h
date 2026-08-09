#ifndef SQL_ITERATORS_EXTERNAL_HELPER_BUFFER_H_
#define SQL_ITERATORS_EXTERNAL_HELPER_BUFFER_H_

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <queue>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "sql/iterators/helpers/gpu_hash_join.h"
#include "sql/iterators/helpers/model_api.h"
#include "sql/iterators/helpers/sem_join_helper.h"
#include "sql/semantic_batch_controller.h"
#include "sql/semantic_profile.h"

namespace vipersql_semantic_fingerprint {

constexpr uint64_t kOffsetBasis = 14695981039346656037ULL;

inline void AddByte(uint64_t *fingerprint, uint8_t byte) {
  *fingerprint ^= byte;
  *fingerprint *= 1099511628211ULL;
}

inline void AddSize(uint64_t *fingerprint, size_t value) {
  for (size_t index = 0; index < sizeof(value); ++index) {
    AddByte(fingerprint, static_cast<uint8_t>((value >> (index * 8)) & 0xffU));
  }
}

inline void AddString(uint64_t *fingerprint, const std::string &value) {
  AddSize(fingerprint, value.size());
  for (unsigned char byte : value) AddByte(fingerprint, byte);
}

}  // namespace vipersql_semantic_fingerprint

// Manages input buffering, helper calls, and result delivery.
template <typename TupleType, typename ResultType>
class ViperFlow {
 public:
  ViperFlow(size_t max_memory_available, const std::string &helper_name,
            size_t semantic_predicate_count = 1,
            std::string semantic_feedback_key = {},
            std::string semantic_join_predicate = {});

  ~ViperFlow();

  bool PushTuple(const TupleType &tuple);

  // Push a tuple and report whether this call submitted a new helper batch.
  // Iterators use this acknowledgement to enforce fill-before-pop scheduling.
  bool PushTuple(const TupleType &tuple, bool *did_submit);

  bool FlushBatch();

  bool FlushControl(const std::string &status);

  // Drain any outstanding request and discard state from a prior execution.
  bool Reset();

  std::unique_ptr<ResultType> PopResult();

  bool PopResult(ResultType *result);

  // Return whether PopResult() can complete without waiting for the helper.
  bool HasReadyResult() const;

  // Return whether buffered input, an external call, or queued output remains.
  bool HasPendingWork() const;

  bool HasError() const;

  // One ViperFlow owns at most one helper call.
  bool IsIdle() const;

  // Return whether an idle helper should be refilled before exposing output.
  // The output watermark bounds the ready-result backlog.
  bool ShouldRefillBeforePop() const;

  void SetStatus(const std::string &status);

  // Account only active relational production work. Iterator callers measure
  // child Read(), packing, and prompt rendering, excluding PushTuple() waits.
  void ObserveProducerActiveSeconds(double seconds);

  // Avoid clock reads entirely for fixed profiles and non-semantic helpers.
  bool ShouldMeasureProducerTime() const;

 private:
  bool FetchAndQueueResults();

  bool SubmitInputBuffer();

  void EnsureSemanticBatchController();

  void MaybeCompleteSemanticFeedback();

  std::vector<TupleType> m_input_buffer;
  std::queue<ResultType> m_result_queue;

  size_t m_batch_size;
  size_t m_min_batch_size;
  size_t m_configured_min_batch_size;
  size_t m_configured_max_batch_size;
  size_t m_configured_result_refill_watermark;
  size_t m_result_refill_watermark;
  size_t m_max_memory_bytes;
  std::string m_helper_name;
  size_t m_semantic_predicate_count;
  vipersql::SemanticBatchOperatorKind m_operator_kind{
      vipersql::SemanticBatchOperatorKind::kOther};
  // The startup profile is copied into the iterator. Runtime profile reloads
  // are unsupported, and retaining a raw pointer here would allow mixed policy
  // epochs if that ever changed.
  vipersql::SemanticOperatorProfile m_semantic_profile;
  bool m_has_semantic_profile{false};
  std::string m_semantic_feedback_key;
  std::unique_ptr<vipersql::SemanticBatchController>
      m_semantic_batch_controller;
  bool m_adaptive_launch_selected{true};
  size_t m_max_observed_input_bytes{0};
  size_t m_buffered_input_bytes{0};
  size_t m_total_semantic_input_rows{0};
  double m_producer_active_seconds{0.0};
  bool m_semantic_input_exhausted{false};
  bool m_semantic_feedback_completed{false};
  bool m_semantic_memory_forced_batch{false};
  uint64_t m_semantic_input_fingerprint{
      vipersql_semantic_fingerprint::kOffsetBasis};
  uint64_t m_semantic_output_fingerprint{
      vipersql_semantic_fingerprint::kOffsetBasis};
  size_t m_semantic_output_rows{0};

  std::unique_ptr<ExternalHelperInterface> m_helper;

  bool m_external_call_running = false;
  bool m_failed = false;
  size_t m_in_flight_batch_size = 0;
  // Bind result cardinality validation to the request that was submitted.
  // m_gpu_probe_mode is mutable phase state and is not itself an ownership tag
  // for an already in-flight request.
  bool m_in_flight_gpu_probe = false;
  // GPU BUILD and PROBE share one ViperFlow. Track the phase so result-count
  // validation applies only to probe requests.
  bool m_gpu_probe_mode{false};
};

// Implementation

template <typename TupleType, typename ResultType>
ViperFlow<TupleType, ResultType>::ViperFlow(size_t max_memory_available,
                                            const std::string &helper_name,
                                            size_t semantic_predicate_count,
                                            std::string semantic_feedback_key,
                                            std::string semantic_join_predicate)
    : m_max_memory_bytes(max_memory_available),
      m_helper_name(helper_name),
      m_semantic_predicate_count(std::max<size_t>(1, semantic_predicate_count)),
      m_semantic_feedback_key(std::move(semantic_feedback_key)),
      m_external_call_running(false) {
  m_batch_size = kExternalHelperBatchSize;
  m_min_batch_size = std::max<size_t>(1, m_batch_size / 2);

  if (helper_name == "GPUHashJoinHelper") {
    m_helper =
        std::make_unique<gpuhashjoinhelpers::GPUHashJoinHelper>(m_batch_size);
  } else if (helper_name == "semantic_filter") {
    const vipersql::SemanticOperatorProfile &policy =
        vipersql::GetSemanticProfile().unary_filter;
    m_semantic_profile = policy;
    m_has_semantic_profile = true;
    m_operator_kind = vipersql::SemanticBatchOperatorKind::kUnaryFilter;
    m_batch_size = std::max<size_t>(1, policy.max_batch_size);
    m_min_batch_size =
        std::clamp(policy.min_batch_size, size_t{1}, m_batch_size);
    m_helper = std::make_unique<llmhelpers::LLMFilterHelper>();
  } else if (helper_name == "semantic_filter_two_col") {
    const vipersql::SemanticOperatorProfile &policy =
        vipersql::GetSemanticProfile().unary_filter;
    m_semantic_profile = policy;
    m_has_semantic_profile = true;
    m_operator_kind = vipersql::SemanticBatchOperatorKind::kTwoColumnFilter;
    m_batch_size = std::max<size_t>(1, policy.max_batch_size);
    m_min_batch_size =
        std::clamp(policy.min_batch_size, size_t{1}, m_batch_size);
    m_helper = std::make_unique<llmhelpers::LLMFilterHelper>();
  } else if (helper_name == "sem_llm_join") {
    const vipersql::SemanticOperatorProfile &policy =
        vipersql::GetSemanticProfile().binary_join;
    m_semantic_profile = policy;
    m_has_semantic_profile = true;
    m_operator_kind = vipersql::SemanticBatchOperatorKind::kBinaryJoin;
    m_batch_size = std::max<size_t>(1, policy.max_batch_size);
    m_min_batch_size =
        std::clamp(policy.min_batch_size, size_t{1}, m_batch_size);
    m_helper = std::make_unique<semhelpers::SemJoinHelper>(
        "sem_cascade_join", std::move(semantic_join_predicate));
  } else {
    m_helper = nullptr;
  }

  m_configured_min_batch_size = m_min_batch_size;
  m_configured_max_batch_size = m_batch_size;
  m_result_refill_watermark = m_min_batch_size;
  m_configured_result_refill_watermark = m_result_refill_watermark;
  m_adaptive_launch_selected =
      !m_has_semantic_profile || !m_semantic_profile.adaptive_batching ||
      m_operator_kind == vipersql::SemanticBatchOperatorKind::kBinaryJoin;

  m_input_buffer.reserve(m_configured_max_batch_size);

  if (m_helper) {
    if (m_helper->Init()) {
      m_helper.reset();
    }
  }
}

template <typename TupleType, typename ResultType>
ViperFlow<TupleType, ResultType>::~ViperFlow() {
  if (m_semantic_batch_controller != nullptr &&
      !m_semantic_feedback_completed) {
    m_semantic_batch_controller->Complete(/*had_error=*/true);
  }
  if (m_helper) {
    m_helper->Destroy();
  }
}

template <typename TupleType, typename ResultType>
bool ViperFlow<TupleType, ResultType>::FetchAndQueueResults() {
  if (!m_helper) {
    return true;
  }

  if (!m_external_call_running) {
    return false;
  }

  const size_t completed_batch_size = m_in_flight_batch_size;
  const bool completed_gpu_probe = m_in_flight_gpu_probe;
  if (m_helper->Synchronize()) {
    m_failed = true;
    m_external_call_running = false;
    m_in_flight_batch_size = 0;
    m_in_flight_gpu_probe = false;
    return true;
  }

  // ResultBufferCapacity() is the helper contract after Synchronize().
  const size_t result_capacity =
      m_helper->ResultBufferCapacity(m_in_flight_batch_size);
  std::vector<ResultType> results_buffer(result_capacity);
  size_t results_count = 0;

  if (m_helper->FetchResults(results_buffer.data(), &results_count)) {
    m_failed = true;
    m_external_call_running = false;
    m_in_flight_batch_size = 0;
    m_in_flight_gpu_probe = false;
    return true;
  }

  if (results_count > results_buffer.size()) {
    m_failed = true;
    m_external_call_running = false;
    m_in_flight_batch_size = 0;
    m_in_flight_gpu_probe = false;
    return true;
  }

  // GPU PROBE has an exact one-result-per-input contract. Reject a short (or
  // otherwise malformed) batch before any later result batch can shift onto
  // the wrong iterator-owned probe snapshots. BUILD legitimately returns zero
  // results and semantic helpers keep their existing variable-output behavior.
  if (completed_gpu_probe && results_count != completed_batch_size) {
    m_failed = true;
    m_external_call_running = false;
    m_in_flight_batch_size = 0;
    m_in_flight_gpu_probe = false;
    return true;
  }

  if (results_count != 0) {
    if constexpr (std::is_same_v<std::decay_t<ResultType>, uint8_t>) {
      if (m_has_semantic_profile &&
          m_operator_kind != vipersql::SemanticBatchOperatorKind::kBinaryJoin) {
        for (size_t index = 0; index < results_count; ++index) {
          vipersql_semantic_fingerprint::AddByte(&m_semantic_output_fingerprint,
                                                 results_buffer[index]);
        }
        m_semantic_output_rows += results_count;
      }
    } else if constexpr (std::is_same_v<std::decay_t<ResultType>,
                                        std::string>) {
      if (m_has_semantic_profile &&
          m_operator_kind != vipersql::SemanticBatchOperatorKind::kBinaryJoin) {
        for (size_t index = 0; index < results_count; ++index) {
          vipersql_semantic_fingerprint::AddString(
              &m_semantic_output_fingerprint, results_buffer[index]);
        }
        m_semantic_output_rows += results_count;
      }
    }
    for (size_t index = 0; index < results_count; ++index) {
      m_result_queue.push(std::move(results_buffer[index]));
    }
  }

  if (m_semantic_batch_controller != nullptr && completed_batch_size != 0) {
    m_semantic_batch_controller->RecordProviderEpoch(
        m_helper->LastServerEpoch());
    const double service_seconds = m_helper->LastBatchServiceSeconds();
    if (service_seconds > 0.0) {
      m_semantic_batch_controller->RecordSuccessfulBatch(completed_batch_size,
                                                         service_seconds);
    }
  }

  m_external_call_running = false;
  m_in_flight_batch_size = 0;
  m_in_flight_gpu_probe = false;
  MaybeCompleteSemanticFeedback();

  return false;
}

template <typename TupleType, typename ResultType>
bool ViperFlow<TupleType, ResultType>::SubmitInputBuffer() {
  const size_t actual_batch_size = m_input_buffer.size();
  if (m_helper->SubmitBatch(m_input_buffer.data(), actual_batch_size)) {
    m_failed = true;
    m_in_flight_gpu_probe = false;
    return true;
  }

  m_input_buffer.clear();
  m_buffered_input_bytes = 0;
  m_external_call_running = true;
  m_in_flight_batch_size = actual_batch_size;
  m_in_flight_gpu_probe = m_gpu_probe_mode && actual_batch_size != 0;
  return false;
}

template <typename TupleType, typename ResultType>
void ViperFlow<TupleType, ResultType>::EnsureSemanticBatchController() {
  if (m_semantic_batch_controller != nullptr || !m_has_semantic_profile ||
      !m_semantic_profile.adaptive_batching ||
      m_operator_kind == vipersql::SemanticBatchOperatorKind::kBinaryJoin) {
    return;
  }

  m_semantic_batch_controller =
      std::make_unique<vipersql::SemanticBatchController>(
          m_semantic_profile, m_operator_kind, m_semantic_predicate_count,
          m_max_observed_input_bytes, m_semantic_feedback_key,
          m_configured_max_batch_size);
  const vipersql::SemanticBatchPolicy &policy =
      m_semantic_batch_controller->policy();
  m_min_batch_size =
      std::clamp(policy.launch_rows, size_t{1}, m_configured_max_batch_size);
  m_batch_size = std::clamp(policy.target_rows, m_min_batch_size,
                            m_configured_max_batch_size);
  m_result_refill_watermark = std::clamp(
      policy.refill_watermark_rows, size_t{1}, m_configured_max_batch_size);
  m_semantic_batch_controller->RecordInputRows(m_total_semantic_input_rows);
  m_semantic_batch_controller->RecordProducerActiveSeconds(
      m_producer_active_seconds);
  m_adaptive_launch_selected = true;
}

template <typename TupleType, typename ResultType>
void ViperFlow<TupleType, ResultType>::MaybeCompleteSemanticFeedback() {
  if (m_semantic_batch_controller == nullptr || m_semantic_feedback_completed ||
      !m_semantic_input_exhausted || !m_input_buffer.empty() ||
      m_external_call_running || !m_result_queue.empty()) {
    return;
  }
  m_semantic_batch_controller->RecordResultFingerprint(
      m_semantic_input_fingerprint, m_semantic_output_fingerprint,
      m_semantic_output_rows);
  m_semantic_batch_controller->RecordProviderEpoch(m_helper->LastServerEpoch());
  if (m_semantic_memory_forced_batch) {
    m_semantic_batch_controller->RecordMemoryForcedBatch();
  }
  m_semantic_batch_controller->Complete(m_failed);
  m_semantic_feedback_completed = true;
}

template <typename TupleType, typename ResultType>
bool ViperFlow<TupleType, ResultType>::PushTuple(const TupleType &tuple) {
  return PushTuple(tuple, nullptr);
}

template <typename TupleType, typename ResultType>
bool ViperFlow<TupleType, ResultType>::PushTuple(const TupleType &tuple,
                                                 bool *did_submit) {
  if (did_submit != nullptr) *did_submit = false;

  if (!m_helper || m_failed) {
    return true;
  }

  m_input_buffer.push_back(tuple);
  if (m_has_semantic_profile) {
    ++m_total_semantic_input_rows;
    if (m_semantic_batch_controller != nullptr) {
      m_semantic_batch_controller->RecordInputRows(1);
    }
  }

  if constexpr (std::is_same_v<std::decay_t<TupleType>, std::string>) {
    if (m_has_semantic_profile &&
        m_operator_kind != vipersql::SemanticBatchOperatorKind::kBinaryJoin) {
      vipersql_semantic_fingerprint::AddString(&m_semantic_input_fingerprint,
                                               tuple);
    }
    m_max_observed_input_bytes =
        std::max(m_max_observed_input_bytes, tuple.size());
    if (tuple.size() >
        std::numeric_limits<size_t>::max() - m_buffered_input_bytes) {
      m_buffered_input_bytes = std::numeric_limits<size_t>::max();
    } else {
      m_buffered_input_bytes += tuple.size();
    }
  }

  // Resolve the launch watermark only after the baseline sample has arrived,
  // so long/compound classification observes a small set of real inputs. The
  // optimizer's estimated cardinality is deliberately not consulted here.
  if (!m_adaptive_launch_selected &&
      m_input_buffer.size() >= m_configured_min_batch_size) {
    EnsureSemanticBatchController();
  }

  const bool reached_max = m_input_buffer.size() >= m_batch_size;
  const bool reached_memory =
      m_max_memory_bytes > 0 && m_buffered_input_bytes >= m_max_memory_bytes;
  const bool at_or_above_min = m_input_buffer.size() >= m_min_batch_size;
  // The maximum size is an unconditional submission boundary. Below it,
  // submit after reaching the minimum size when the helper is idle.
  const bool idle_at_min = at_or_above_min && !reached_max && IsIdle();

  if (reached_memory && !reached_max) {
    m_semantic_memory_forced_batch = true;
  }

  if (reached_max || reached_memory || idle_at_min) {
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
    return true;
  }

  if (m_has_semantic_profile &&
      m_operator_kind != vipersql::SemanticBatchOperatorKind::kBinaryJoin &&
      !m_semantic_input_exhausted) {
    m_semantic_input_exhausted = true;
    EnsureSemanticBatchController();
    if (m_semantic_batch_controller != nullptr) {
      m_semantic_batch_controller->RecordEof();
    }
  }

  if (FetchAndQueueResults()) {
    return true;
  }

  if (m_input_buffer.empty()) {
    MaybeCompleteSemanticFeedback();
    return false;
  }

  // EOF submits residual input even below the minimum batch size.
  return SubmitInputBuffer();
}

template <typename TupleType, typename ResultType>
bool ViperFlow<TupleType, ResultType>::FlushControl(const std::string &status) {
  if (!m_helper) {
    return true;
  }

  if (m_external_call_running) {
    if (FetchAndQueueResults()) return true;
  }

  m_helper->SetStatus(status);
  if (m_helper->SubmitBatch(nullptr, 0)) {
    return true;
  }
  m_external_call_running = true;
  m_in_flight_batch_size = 0;
  m_in_flight_gpu_probe = false;

  if (FetchAndQueueResults()) return true;
  return false;
}

template <typename TupleType, typename ResultType>
bool ViperFlow<TupleType, ResultType>::Reset() {
  if (!m_helper) {
    return true;
  }

  const bool is_gpu_hash_join = m_helper_name == "GPUHashJoinHelper";
  bool reinitialize_gpu_helper = is_gpu_hash_join && m_failed;
  if (m_external_call_running && FetchAndQueueResults()) {
    // A CUDA error is recoverable at an execution boundary. Fetching the
    // outstanding request has already marked the old request as no longer
    // running, so discard its resources below and create clean buffers for the
    // next Init(). Other helpers retain their historical error behavior.
    if (!is_gpu_hash_join) return true;
    reinitialize_gpu_helper = true;
  }
  if (m_semantic_batch_controller != nullptr &&
      !m_semantic_feedback_completed) {
    m_semantic_batch_controller->Complete(/*had_error=*/true);
  }

  m_input_buffer.clear();
  m_result_queue = {};
  m_external_call_running = false;
  m_failed = false;
  m_in_flight_batch_size = 0;
  m_in_flight_gpu_probe = false;
  m_gpu_probe_mode = false;
  m_min_batch_size = m_configured_min_batch_size;
  m_batch_size = m_configured_max_batch_size;
  m_result_refill_watermark = m_configured_result_refill_watermark;
  m_adaptive_launch_selected =
      !m_has_semantic_profile || !m_semantic_profile.adaptive_batching ||
      m_operator_kind == vipersql::SemanticBatchOperatorKind::kBinaryJoin;
  m_semantic_batch_controller.reset();
  m_max_observed_input_bytes = 0;
  m_buffered_input_bytes = 0;
  m_total_semantic_input_rows = 0;
  m_producer_active_seconds = 0.0;
  m_semantic_input_exhausted = false;
  m_semantic_feedback_completed = false;
  m_semantic_memory_forced_batch = false;
  m_semantic_input_fingerprint = vipersql_semantic_fingerprint::kOffsetBasis;
  m_semantic_output_fingerprint = vipersql_semantic_fingerprint::kOffsetBasis;
  m_semantic_output_rows = 0;

  if (reinitialize_gpu_helper) {
    m_helper->Destroy();
    if (m_helper->Init()) {
      // Leave a clean logical ViperFlow state but remember the failed recovery
      // so a later execution boundary can retry initialization.
      m_failed = true;
      return true;
    }
  }
  return false;
}

template <typename TupleType, typename ResultType>
std::unique_ptr<ResultType> ViperFlow<TupleType, ResultType>::PopResult() {
  ResultType result;
  if (!PopResult(&result)) return nullptr;
  return std::make_unique<ResultType>(std::move(result));
}

template <typename TupleType, typename ResultType>
bool ViperFlow<TupleType, ResultType>::PopResult(ResultType *result) {
  if (result == nullptr) return false;
  if (m_result_queue.empty()) {
    if (FetchAndQueueResults()) {
      return false;
    }
    if (m_result_queue.empty()) {
      return false;
    }
  }

  *result = std::move(m_result_queue.front());
  m_result_queue.pop();
  if (m_semantic_batch_controller != nullptr) {
    MaybeCompleteSemanticFeedback();
  }
  return true;
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
  return m_result_queue.size() < m_result_refill_watermark && IsIdle();
}

template <typename TupleType, typename ResultType>
void ViperFlow<TupleType, ResultType>::SetStatus(const std::string &status) {
  if (m_helper_name == "GPUHashJoinHelper") {
    m_gpu_probe_mode = status == "PROBE";
  }
  if (m_helper) {
    m_helper->SetStatus(status);
  }
}

template <typename TupleType, typename ResultType>
void ViperFlow<TupleType, ResultType>::ObserveProducerActiveSeconds(
    double seconds) {
  if (!(seconds > 0.0)) return;
  m_producer_active_seconds += seconds;
  if (m_semantic_batch_controller != nullptr) {
    m_semantic_batch_controller->RecordProducerActiveSeconds(seconds);
  }
}

template <typename TupleType, typename ResultType>
bool ViperFlow<TupleType, ResultType>::ShouldMeasureProducerTime() const {
  return m_has_semantic_profile && m_semantic_profile.adaptive_batching &&
         m_operator_kind != vipersql::SemanticBatchOperatorKind::kBinaryJoin;
}

#endif  // SQL_ITERATORS_EXTERNAL_HELPER_BUFFER_H_
