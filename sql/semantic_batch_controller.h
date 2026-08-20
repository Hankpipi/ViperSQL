/* Copyright (c) 2026, Zihao Yu.

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License, version 2.0.
*/

#ifndef SQL_SEMANTIC_BATCH_CONTROLLER_H_
#define SQL_SEMANTIC_BATCH_CONTROLLER_H_

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "sql/semantic_profile.h"

namespace vipersql {

/**
  The three independent watermarks used by a semantic ViperFlow execution.

  launch_rows is the minimum occupancy at which an observed-idle helper may
  be launched. target_rows is the hard input-buffer target/cap. The refill
  watermark applies to ready output, not input, and therefore must not be
  overloaded as either input watermark.
*/
struct SemanticBatchPolicy {
  size_t launch_rows{1};
  size_t target_rows{1};
  size_t refill_watermark_rows{1};
  size_t safe_cap_rows{1};
};

/** Policy selection phase. */
enum class SemanticBatchSelectionPhase {
  kColdStart,
  kPrioritizedExploration,
  kConfidenceRace,
  kIncumbentExploit,
  kEmpiricalBest,
  kSparseReprobe,
  kIncumbentWarmup,
  kGuardedProbe,
  kIncumbentRecovery,
  kRegretBoundedExploit
};

/** Selection state needed when publishing feedback for the next execution. */
struct SemanticBatchSelectionDiagnostics {
  SemanticBatchSelectionPhase phase{SemanticBatchSelectionPhase::kColdStart};
  std::vector<SemanticBatchPolicy> safe_shortlist;
};

/** Per-execution observations retained by SemanticBatchController. */
struct SemanticBatchExecutionStats {
  size_t total_input_rows{0};
  double producer_active_seconds{0.0};
  uint64_t successful_batch_count{0};
  size_t successful_batch_rows{0};
  double successful_batch_service_seconds{0.0};
  double profiled_batch_service_seconds{0.0};
  double execution_elapsed_seconds{0.0};
  bool saw_eof{false};
  bool completed{false};
  bool had_error{false};
  bool has_result_fingerprint{false};
  uint64_t input_fingerprint{0};
  uint64_t output_fingerprint{0};
  size_t fingerprint_output_rows{0};
  std::string provider_epoch;
  bool provider_epoch_observed{false};
  bool provider_epoch_changed{false};
  bool had_memory_forced_batch{false};
};

/**
  One semantic operator execution's fixed batch policy and observations.

  The profile is copied into a const member and policy selection happens once
  in the constructor. Record*() never changes the active policy: publishing
  feedback can affect only a subsequently constructed controller (normally a
  later iterator Reset/execution epoch), so buffered and in-flight rows never
  observe a mid-execution watermark change.
*/
class SemanticBatchController {
 public:
  SemanticBatchController(const SemanticOperatorProfile &profile,
                          SemanticBatchOperatorKind operator_kind,
                          size_t predicate_count,
                          size_t maximum_observed_input_bytes,
                          std::string feedback_key,
                          size_t hard_safe_cap_rows = 0);

  SemanticBatchController(const SemanticBatchController &) = delete;
  SemanticBatchController &operator=(const SemanticBatchController &) = delete;

  const SemanticBatchPolicy &policy() const { return m_policy; }

  void RecordInputRows(size_t rows);
  void RecordProducerActiveSeconds(double seconds);
  void RecordSuccessfulBatch(size_t rows, double service_seconds);
  /**
    Record an ordered, opaque execution digest maintained by ViperFlow. No
    input text or model output is retained or exposed by the controller.
  */
  void RecordResultFingerprint(uint64_t input_fingerprint,
                               uint64_t output_fingerprint, size_t output_rows);
  void RecordProviderEpoch(std::string provider_epoch);
  void RecordMemoryForcedBatch();
  void RecordEof();

  /**
    Mark the epoch complete and, when valid, publish feedback for the next one.
    Calls are idempotent. Empty keys, errors, missing EOF, incomplete samples,
    and binary semantic joins intentionally never update the cache.
  */
  void Complete(bool had_error);

 private:
  const SemanticOperatorProfile m_profile;
  const SemanticBatchOperatorKind m_operator_kind;
  const size_t m_predicate_count;
  const size_t m_maximum_observed_input_bytes;
  const std::string m_feedback_key;
  const size_t m_hard_safe_cap_rows;
  const std::string m_cache_key;
  SemanticBatchPolicy m_policy;
  SemanticBatchSelectionDiagnostics m_selection_diagnostics;
  SemanticBatchExecutionStats m_stats;
  std::chrono::steady_clock::time_point m_execution_started_at;
};

}  // namespace vipersql

#endif  // SQL_SEMANTIC_BATCH_CONTROLLER_H_
