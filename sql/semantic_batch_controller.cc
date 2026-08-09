/* Copyright (c) 2026, ViperSQL contributors.

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License, version 2.0.
*/

#include "sql/semantic_batch_controller.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iterator>
#include <limits>
#include <list>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace vipersql {
namespace {

constexpr size_t kMaximumFeedbackEntries = 256;
constexpr size_t kMaximumCandidateValues = 32;
constexpr size_t kMaximumStoredKeyBytes = 512;
constexpr double kFeedbackEwmaWeight = 0.25;
constexpr double kMinimumLatencyScale = 0.10;
constexpr double kMaximumLatencyScale = 10.0;
constexpr size_t kMaximumShortlistArms = 8;
constexpr size_t kMaximumArmsPerFeedbackEntry = kMaximumShortlistArms;
constexpr size_t kRobustRecentSampleCount = 7;
constexpr uint64_t kShortlistMinimumSamples = 3;
constexpr double kIndifferenceRatio = 1.05;
constexpr double kConfidenceNoiseFloorRatio = kIndifferenceRatio - 1.0;
constexpr uint64_t kConfidenceRaceInterval = 4;
constexpr uint64_t kSparseReprobeInterval = 64;
constexpr uint64_t kIncumbentWarmupSamples = 2;
constexpr uint64_t kIncumbentExecutionsBetweenProbes = 2;
constexpr uint64_t kPromotionPairedWins = 2;
constexpr double kMaximumPredictedProbeRatio = 1.10;
constexpr double kPairedPromotionRatio = 0.95;
constexpr double kPairedTieRatio = 1.05;
constexpr double kPairedSuspensionRatio = 1.10;
constexpr double kMaximumBracketEnvironmentRatio = 1.25;
constexpr auto kFeedbackTimeToLive = std::chrono::minutes(30);

struct SemanticFillBeforePopSimulationInput {
  size_t total_input_rows{0};
  double producer_seconds_per_row{0.0};
  double helper_latency_scale{1.0};
  SemanticBatchPolicy policy;
};

struct SemanticFillBeforePopSimulationResult {
  double elapsed_seconds{0.0};
  double first_output_seconds{0.0};
  double helper_busy_seconds{0.0};
  double producer_stall_seconds{0.0};
  uint64_t submitted_batches{0};
  size_t largest_submitted_batch_rows{0};
};

struct SemanticBatchArmFeedback {
  SemanticBatchArmFeedback() = default;
  SemanticBatchArmFeedback(
      size_t launch_rows_arg, size_t target_rows_arg,
      uint64_t completed_executions_arg = 0,
      double mean_elapsed_seconds_arg = 0.0,
      double ewma_elapsed_seconds_arg = 0.0,
      double elapsed_m2_seconds_squared_arg = 0.0,
      const std::vector<double> &recent_elapsed_seconds_arg = {})
      : launch_rows(launch_rows_arg),
        target_rows(target_rows_arg),
        completed_executions(completed_executions_arg),
        mean_elapsed_seconds(mean_elapsed_seconds_arg),
        ewma_elapsed_seconds(ewma_elapsed_seconds_arg),
        elapsed_m2_seconds_squared(elapsed_m2_seconds_squared_arg),
        recent_elapsed_seconds(recent_elapsed_seconds_arg) {}

  size_t launch_rows{1};
  size_t target_rows{1};
  uint64_t completed_executions{0};
  double mean_elapsed_seconds{0.0};
  double ewma_elapsed_seconds{0.0};
  double elapsed_m2_seconds_squared{0.0};
  std::vector<double> recent_elapsed_seconds;
  uint64_t paired_comparisons{0};
  uint64_t paired_wins{0};
  uint64_t paired_losses{0};
  uint64_t suspension_count{0};
  double last_paired_elapsed_ratio{1.0};
  bool suspended{false};
  bool quarantined{false};
  bool indifferent{false};
};

/** Cached observations for one query context. */
struct SemanticBatchLearnedFeedback {
  uint64_t completed_executions{0};
  double expected_input_rows{0.0};
  double producer_seconds_per_row{0.0};
  double helper_latency_scale{1.0};
  std::vector<SemanticBatchArmFeedback> arms;
  std::vector<SemanticBatchPolicy> frozen_shortlist;
  bool has_incumbent{false};
  SemanticBatchPolicy incumbent_policy;
  uint64_t incumbent_executions_since_probe{0};
  bool recovery_required{false};
  SemanticBatchPolicy pending_probe_policy;
  double pending_probe_elapsed_seconds{0.0};
  double pending_probe_helper_latency_scale{0.0};
  double pending_probe_producer_seconds_per_row{0.0};
  double pending_before_elapsed_seconds{0.0};
  double pending_before_helper_latency_scale{0.0};
  double pending_before_producer_seconds_per_row{0.0};
  double last_incumbent_elapsed_seconds{0.0};
  double last_incumbent_helper_latency_scale{0.0};
  double last_incumbent_producer_seconds_per_row{0.0};
  uint64_t last_incumbent_batch_count{0};
  SemanticBatchSelectionPhase observation_phase{
      SemanticBatchSelectionPhase::kColdStart};
  SemanticBatchPolicy observation_policy;
  uint64_t observation_batch_count{0};
  bool has_result_fingerprint{false};
  uint64_t input_fingerprint{0};
  uint64_t output_fingerprint{0};
  size_t fingerprint_input_rows{0};
  size_t fingerprint_output_rows{0};
  std::string provider_epoch;
  bool result_feedback_stable{true};
};

/** Inputs used by the policy selector. */
struct SemanticBatchSelectionContext {
  SemanticBatchOperatorKind operator_kind{SemanticBatchOperatorKind::kOther};
  size_t predicate_count{1};
  size_t maximum_observed_input_bytes{0};
  size_t hard_safe_cap_rows{0};
  SemanticBatchLearnedFeedback learned;
};

size_t SemanticProfiledBatchSafeCap(const SemanticOperatorProfile &profile,
                                    size_t hard_safe_cap_rows = 0);

bool UpdateSemanticBatchArmFeedback(SemanticBatchArmFeedback *feedback,
                                    double elapsed_seconds);

bool IsNonnegativeFinite(double value) {
  return std::isfinite(value) && value >= 0.0;
}

bool IsPositiveFinite(double value) {
  return std::isfinite(value) && value > 0.0;
}

size_t SaturatingAdd(size_t left, size_t right) {
  if (right > std::numeric_limits<size_t>::max() - left) {
    return std::numeric_limits<size_t>::max();
  }
  return left + right;
}

uint64_t SaturatingIncrement(uint64_t value) {
  return value == std::numeric_limits<uint64_t>::max() ? value : value + 1;
}

uint64_t SaturatingAddUint64(uint64_t left, uint64_t right) {
  if (right > std::numeric_limits<uint64_t>::max() - left) {
    return std::numeric_limits<uint64_t>::max();
  }
  return left + right;
}

uint64_t SaturatingAddProduct(uint64_t value, uint64_t delta,
                              size_t multiplier) {
  if (delta == 0 || multiplier == 0) return value;
  if (multiplier > (std::numeric_limits<uint64_t>::max() - value) / delta) {
    return std::numeric_limits<uint64_t>::max();
  }
  return value + delta * static_cast<uint64_t>(multiplier);
}

double SaturatingAddSeconds(double left, double right) {
  if (!IsNonnegativeFinite(right)) return left;
  if (right > std::numeric_limits<double>::max() - left) {
    return std::numeric_limits<double>::max();
  }
  return left + right;
}

double MeasurementLatency(const SemanticBatchMeasurement &measurement) {
  if (IsPositiveFinite(measurement.latency_seconds)) {
    return measurement.latency_seconds;
  }
  if (IsPositiveFinite(measurement.p90_latency_seconds)) {
    return measurement.p90_latency_seconds;
  }
  return 0.0;
}

double ProfileBatchLatencySeconds(const SemanticOperatorProfile &profile,
                                  size_t rows) {
  if (rows == 0) return 0.0;

  const SemanticBatchMeasurement *lower = nullptr;
  const SemanticBatchMeasurement *upper = nullptr;
  for (const SemanticBatchMeasurement &point : profile.latency_curve) {
    if (point.batch_size == 0 || MeasurementLatency(point) == 0.0) continue;
    if (point.batch_size <= rows &&
        (lower == nullptr || point.batch_size > lower->batch_size)) {
      lower = &point;
    }
    if (point.batch_size >= rows &&
        (upper == nullptr || point.batch_size < upper->batch_size)) {
      upper = &point;
    }
  }

  if (lower != nullptr && upper != nullptr) {
    if (lower->batch_size == upper->batch_size) {
      return MeasurementLatency(*lower);
    }
    const double fraction =
        static_cast<double>(rows - lower->batch_size) /
        static_cast<double>(upper->batch_size - lower->batch_size);
    return MeasurementLatency(*lower) +
           fraction * (MeasurementLatency(*upper) - MeasurementLatency(*lower));
  }
  if (upper != nullptr) return MeasurementLatency(*upper);

  // Do not extrapolate beyond measured batch sizes.
  if (lower != nullptr) return MeasurementLatency(*lower);

  const double transfer_bandwidth =
      IsPositiveFinite(profile.transfer_bandwidth_bytes_per_second)
          ? profile.transfer_bandwidth_bytes_per_second
          : 1.0;
  const double processing_throughput =
      IsPositiveFinite(profile.processing_throughput_rows_per_second)
          ? profile.processing_throughput_rows_per_second
          : 1.0;
  const double bytes_per_row = std::max(0.0, profile.input_bytes_per_row) +
                               std::max(0.0, profile.result_bytes_per_input);
  const double invocation =
      IsNonnegativeFinite(profile.invocation_overhead_seconds)
          ? profile.invocation_overhead_seconds
          : 0.0;
  const double row_count = static_cast<double>(rows);
  return invocation + row_count / processing_throughput +
         row_count * bytes_per_row / transfer_bandwidth;
}

void StableHashInteger(uint64_t *signature, uint64_t value) {
  for (size_t index = 0; index < sizeof(value); ++index) {
    *signature ^= static_cast<uint8_t>((value >> (index * 8)) & 0xffU);
    *signature *= 1099511628211ULL;
  }
}

void StableHashDouble(uint64_t *signature, double value) {
  static_assert(sizeof(value) == sizeof(uint64_t));
  uint64_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  StableHashInteger(signature, bits);
}

uint64_t StableHashString(const std::string &value) {
  uint64_t signature = 14695981039346656037ULL;
  for (const unsigned char byte : value) {
    signature ^= byte;
    signature *= 1099511628211ULL;
  }
  return signature;
}

uint64_t ProfileSignature(const SemanticOperatorProfile &profile) {
  uint64_t signature = 14695981039346656037ULL;
  StableHashInteger(&signature, profile.min_batch_size);
  StableHashInteger(&signature, profile.max_batch_size);
  StableHashInteger(&signature, profile.helper_parallelism);
  StableHashDouble(&signature, profile.input_bytes_per_row);
  for (const SemanticBatchMeasurement &point : profile.latency_curve) {
    StableHashInteger(&signature, point.batch_size);
    StableHashDouble(&signature, point.latency_seconds);
    StableHashDouble(&signature, point.p90_latency_seconds);
  }
  return signature;
}

std::string MakeCacheKey(const std::string &feedback_key,
                         SemanticBatchOperatorKind operator_kind,
                         size_t predicate_count,
                         size_t maximum_observed_input_bytes,
                         size_t hard_safe_cap_rows,
                         const SemanticOperatorProfile &profile) {
  if (feedback_key.empty()) return {};

  std::string key = std::to_string(static_cast<int>(operator_kind));
  key.push_back(':');
  key += std::to_string(std::max<size_t>(1, predicate_count));
  key.push_back(':');
  // Bucket prompt width to bound cache cardinality.
  size_t width_bucket = 0;
  for (size_t width = maximum_observed_input_bytes; width > 1; width >>= 1) {
    ++width_bucket;
  }
  const double profiled_bytes = std::max(1.0, profile.input_bytes_per_row);
  const double threshold_as_double =
      std::ceil(std::max(512.0, 8.0 * profiled_bytes));
  const size_t long_input_threshold =
      threshold_as_double >=
              static_cast<double>(std::numeric_limits<size_t>::max())
          ? std::numeric_limits<size_t>::max()
          : static_cast<size_t>(threshold_as_double);
  key += "width2=";
  key += std::to_string(width_bucket);
  key += ":long=";
  key.push_back(maximum_observed_input_bytes >= long_input_threshold ? '1'
                                                                     : '0');
  key.push_back(':');
  key += feedback_key;
  key += ":profile=";
  key += std::to_string(ProfileSignature(profile));
  key += ":cap=";
  key +=
      std::to_string(SemanticProfiledBatchSafeCap(profile, hard_safe_cap_rows));
  if (key.size() <= kMaximumStoredKeyBytes) return key;

  // The cache is intentionally bounded in key bytes as well as entry count.
  // Retain a deterministic digest of the omitted suffix. A collision can at
  // worst select another safe policy.
  const uint64_t digest = StableHashString(key);
  key.resize(kMaximumStoredKeyBytes / 2);
  key += ":#";
  key += std::to_string(digest);
  return key;
}

struct FeedbackCacheEntry {
  std::string key;
  SemanticBatchLearnedFeedback feedback;
  std::chrono::steady_clock::time_point updated_at;
};

struct FeedbackCache {
  std::mutex mutex;
  // Front is most recently used. The small fixed bound keeps linear lookup
  // simple and deterministic while avoiding an unbounded process-wide map.
  std::list<FeedbackCacheEntry> entries;
};

FeedbackCache &GetFeedbackCache() {
  static FeedbackCache cache;
  return cache;
}

bool ReadFeedback(const std::string &key,
                  SemanticBatchLearnedFeedback *feedback) {
  if (key.empty() || feedback == nullptr) return false;
  FeedbackCache &cache = GetFeedbackCache();
  std::lock_guard<std::mutex> lock(cache.mutex);
  const auto found =
      std::find_if(cache.entries.begin(), cache.entries.end(),
                   [&](const auto &entry) { return entry.key == key; });
  if (found == cache.entries.end()) return false;
  const auto now = std::chrono::steady_clock::now();
  if (now - found->updated_at > kFeedbackTimeToLive) {
    cache.entries.erase(found);
    return false;
  }
  *feedback = found->feedback;
  cache.entries.splice(cache.entries.begin(), cache.entries, found);
  return true;
}

double Ewma(double old_value, double sample) {
  return old_value + kFeedbackEwmaWeight * (sample - old_value);
}

double RobustArmElapsedSeconds(const SemanticBatchArmFeedback &feedback) {
  std::vector<double> samples;
  samples.reserve(feedback.recent_elapsed_seconds.size());
  for (double sample : feedback.recent_elapsed_seconds) {
    if (IsPositiveFinite(sample)) samples.push_back(sample);
  }
  if (samples.size() < kShortlistMinimumSamples)
    return feedback.mean_elapsed_seconds;

  std::sort(samples.begin(), samples.end());
  const size_t middle = samples.size() / 2;
  if (samples.size() % 2 != 0) return samples[middle];
  return samples[middle - 1] + (samples[middle] - samples[middle - 1]) / 2.0;
}

bool SamePolicyArm(const SemanticBatchPolicy &left,
                   const SemanticBatchPolicy &right) {
  return left.launch_rows == right.launch_rows &&
         left.target_rows == right.target_rows;
}

bool ShortlistContainsArm(const std::vector<SemanticBatchPolicy> &shortlist,
                          size_t launch_rows, size_t target_rows) {
  return std::any_of(shortlist.begin(), shortlist.end(),
                     [&](const SemanticBatchPolicy &candidate) {
                       return candidate.launch_rows == launch_rows &&
                              candidate.target_rows == target_rows;
                     });
}

SemanticBatchArmFeedback *FindArmFeedback(
    SemanticBatchLearnedFeedback *feedback, const SemanticBatchPolicy &policy) {
  if (feedback == nullptr) return nullptr;
  const auto found = std::find_if(
      feedback->arms.begin(), feedback->arms.end(), [&](const auto &arm) {
        return arm.launch_rows == policy.launch_rows &&
               arm.target_rows == policy.target_rows;
      });
  return found == feedback->arms.end() ? nullptr : &*found;
}

double ObservationElapsedSeconds(const SemanticBatchLearnedFeedback &sample) {
  const auto found = std::find_if(
      sample.arms.begin(), sample.arms.end(), [&](const auto &arm) {
        return arm.launch_rows == sample.observation_policy.launch_rows &&
               arm.target_rows == sample.observation_policy.target_rows &&
               arm.completed_executions != 0 &&
               IsPositiveFinite(arm.mean_elapsed_seconds);
      });
  return found == sample.arms.end() ? 0.0 : found->mean_elapsed_seconds;
}

bool EnvironmentRatioWithin(double before, double after, double limit) {
  if (!IsPositiveFinite(before) || !IsPositiveFinite(after) ||
      !IsPositiveFinite(limit)) {
    return false;
  }
  return std::max(before, after) / std::min(before, after) <= limit;
}

bool EnvironmentBracketComparable(double before, double probe, double after,
                                  double limit) {
  return EnvironmentRatioWithin(before, after, limit) &&
         EnvironmentRatioWithin(before, probe, limit) &&
         EnvironmentRatioWithin(probe, after, limit);
}

double GeometricMean(double left, double right) {
  if (!IsPositiveFinite(left) || !IsPositiveFinite(right)) return 0.0;
  const double result = std::sqrt(left) * std::sqrt(right);
  return IsPositiveFinite(result) ? result : 0.0;
}

void ClearPendingProbe(SemanticBatchLearnedFeedback *feedback) {
  feedback->recovery_required = false;
  feedback->pending_probe_policy = {};
  feedback->pending_probe_elapsed_seconds = 0.0;
  feedback->pending_probe_helper_latency_scale = 0.0;
  feedback->pending_probe_producer_seconds_per_row = 0.0;
  feedback->pending_before_elapsed_seconds = 0.0;
  feedback->pending_before_helper_latency_scale = 0.0;
  feedback->pending_before_producer_seconds_per_row = 0.0;
}

void RecordLastIncumbentObservation(SemanticBatchLearnedFeedback *feedback,
                                    const SemanticBatchLearnedFeedback &sample,
                                    double elapsed_seconds) {
  feedback->last_incumbent_elapsed_seconds = elapsed_seconds;
  feedback->last_incumbent_helper_latency_scale = sample.helper_latency_scale;
  feedback->last_incumbent_producer_seconds_per_row =
      sample.producer_seconds_per_row;
  feedback->last_incumbent_batch_count = sample.observation_batch_count;
}

/** Update learned batch-selection state after a completed execution. */
void ApplyRegretBoundedObservation(SemanticBatchLearnedFeedback *feedback,
                                   const SemanticBatchLearnedFeedback &sample) {
  if (feedback == nullptr || feedback->completed_executions == 0) return;

  const double elapsed_seconds = ObservationElapsedSeconds(sample);
  if (!IsPositiveFinite(elapsed_seconds)) return;

  if (!feedback->has_incumbent) {
    feedback->has_incumbent = true;
    feedback->incumbent_policy = sample.observation_policy;
    feedback->incumbent_executions_since_probe = 1;
    ClearPendingProbe(feedback);
    RecordLastIncumbentObservation(feedback, sample, elapsed_seconds);
    return;
  }

  const bool is_probe =
      sample.observation_phase == SemanticBatchSelectionPhase::kGuardedProbe ||
      sample.observation_phase == SemanticBatchSelectionPhase::kSparseReprobe;
  if (is_probe &&
      !SamePolicyArm(sample.observation_policy, feedback->incumbent_policy)) {
    feedback->pending_probe_policy = sample.observation_policy;
    feedback->pending_probe_elapsed_seconds = elapsed_seconds;
    feedback->pending_probe_helper_latency_scale = sample.helper_latency_scale;
    feedback->pending_probe_producer_seconds_per_row =
        sample.producer_seconds_per_row;
    feedback->pending_before_elapsed_seconds =
        feedback->last_incumbent_elapsed_seconds;
    feedback->pending_before_helper_latency_scale =
        feedback->last_incumbent_helper_latency_scale;
    feedback->pending_before_producer_seconds_per_row =
        feedback->last_incumbent_producer_seconds_per_row;
    feedback->recovery_required = true;
    return;
  }

  if (sample.observation_phase ==
          SemanticBatchSelectionPhase::kIncumbentRecovery &&
      feedback->recovery_required &&
      SamePolicyArm(sample.observation_policy, feedback->incumbent_policy)) {
    bool promoted = false;
    const bool comparable_environment =
        EnvironmentBracketComparable(
            feedback->pending_before_helper_latency_scale,
            feedback->pending_probe_helper_latency_scale,
            sample.helper_latency_scale, kMaximumBracketEnvironmentRatio) &&
        EnvironmentBracketComparable(
            feedback->pending_before_producer_seconds_per_row,
            feedback->pending_probe_producer_seconds_per_row,
            sample.producer_seconds_per_row, kMaximumBracketEnvironmentRatio);
    const double incumbent_bracket_seconds = GeometricMean(
        feedback->pending_before_elapsed_seconds, elapsed_seconds);
    if (comparable_environment &&
        IsPositiveFinite(feedback->pending_probe_elapsed_seconds) &&
        IsPositiveFinite(incumbent_bracket_seconds)) {
      const double paired_ratio =
          feedback->pending_probe_elapsed_seconds / incumbent_bracket_seconds;
      SemanticBatchArmFeedback *probe =
          FindArmFeedback(feedback, feedback->pending_probe_policy);
      if (probe != nullptr && IsPositiveFinite(paired_ratio)) {
        probe->paired_comparisons =
            SaturatingIncrement(probe->paired_comparisons);
        probe->last_paired_elapsed_ratio = paired_ratio;
        if (paired_ratio <= kPairedPromotionRatio) {
          probe->paired_wins = SaturatingIncrement(probe->paired_wins);
          probe->suspended = false;
          probe->indifferent = false;
          if (!probe->quarantined &&
              probe->paired_wins >= kPromotionPairedWins &&
              probe->paired_wins > probe->paired_losses) {
            feedback->incumbent_policy = feedback->pending_probe_policy;
            probe->quarantined = false;
            promoted = true;
          }
        } else if (paired_ratio >= kPairedSuspensionRatio) {
          probe->paired_losses = SaturatingIncrement(probe->paired_losses);
          probe->suspension_count =
              SaturatingIncrement(probe->suspension_count);
          probe->suspended = true;
          probe->indifferent = false;
          if (probe->suspension_count >= 2) probe->quarantined = true;
        } else if (paired_ratio <= kPairedTieRatio) {
          // A result within 5% cannot justify changing a known-safe policy.
          probe->indifferent = true;
          probe->suspended = true;
        }
      }
    }

    ClearPendingProbe(feedback);
    feedback->incumbent_executions_since_probe = 0;
    if (promoted) {
      // Measure a promoted policy before selecting another comparison.
      feedback->last_incumbent_elapsed_seconds = 0.0;
      feedback->last_incumbent_helper_latency_scale = 0.0;
      feedback->last_incumbent_producer_seconds_per_row = 0.0;
      feedback->last_incumbent_batch_count = 0;
    } else {
      RecordLastIncumbentObservation(feedback, sample, elapsed_seconds);
    }
    return;
  }

  if (SamePolicyArm(sample.observation_policy, feedback->incumbent_policy)) {
    feedback->incumbent_executions_since_probe =
        SaturatingIncrement(feedback->incumbent_executions_since_probe);
    RecordLastIncumbentObservation(feedback, sample, elapsed_seconds);
  }
}

void PublishFeedback(const std::string &key,
                     const SemanticBatchLearnedFeedback &sample) {
  if (key.empty()) return;
  FeedbackCache &cache = GetFeedbackCache();
  std::lock_guard<std::mutex> lock(cache.mutex);
  const auto found =
      std::find_if(cache.entries.begin(), cache.entries.end(),
                   [&](const auto &entry) { return entry.key == key; });
  if (found == cache.entries.end()) {
    SemanticBatchLearnedFeedback initial = sample;
    if (initial.frozen_shortlist.size() > kMaximumShortlistArms) {
      initial.frozen_shortlist.resize(kMaximumShortlistArms);
    }
    if (!initial.frozen_shortlist.empty()) {
      initial.arms.erase(
          std::remove_if(initial.arms.begin(), initial.arms.end(),
                         [&](const SemanticBatchArmFeedback &arm) {
                           return !ShortlistContainsArm(
                               initial.frozen_shortlist, arm.launch_rows,
                               arm.target_rows);
                         }),
          initial.arms.end());
    }
    if (initial.arms.size() > kMaximumArmsPerFeedbackEntry) {
      initial.arms.resize(kMaximumArmsPerFeedbackEntry);
    }
    ApplyRegretBoundedObservation(&initial, sample);
    cache.entries.push_front(
        {key, std::move(initial), std::chrono::steady_clock::now()});
    if (cache.entries.size() > kMaximumFeedbackEntries) {
      cache.entries.pop_back();
    }
    return;
  }

  SemanticBatchLearnedFeedback &value = found->feedback;
  if (!value.provider_epoch.empty() && !sample.provider_epoch.empty() &&
      value.provider_epoch != sample.provider_epoch) {
    // Do not combine observations from different helper epochs.
    value = sample;
    value.completed_executions = 0;
    value.arms.clear();
    value.frozen_shortlist.clear();
    found->updated_at = std::chrono::steady_clock::now();
    cache.entries.splice(cache.entries.begin(), cache.entries, found);
    return;
  }
  if (value.provider_epoch.empty() && !sample.provider_epoch.empty()) {
    value.provider_epoch = sample.provider_epoch;
  }
  if (value.has_result_fingerprint && sample.has_result_fingerprint) {
    const bool same_input =
        value.input_fingerprint == sample.input_fingerprint &&
        value.fingerprint_input_rows == sample.fingerprint_input_rows;
    if (!same_input) {
      // Reset feedback when the input changes.
      value = sample;
      value.completed_executions = 0;
      value.arms.clear();
      value.frozen_shortlist.clear();
      found->updated_at = std::chrono::steady_clock::now();
      cache.entries.splice(cache.entries.begin(), cache.entries, found);
      return;
    }

    const bool same_output =
        value.output_fingerprint == sample.output_fingerprint &&
        value.fingerprint_output_rows == sample.fingerprint_output_rows;
    if (!same_output || !value.result_feedback_stable) {
      // Result changes invalidate elapsed-time feedback.
      value.result_feedback_stable = false;
      value.completed_executions = 0;
      value.arms.clear();
      value.frozen_shortlist.clear();
      value.has_incumbent = false;
      ClearPendingProbe(&value);
      found->updated_at = std::chrono::steady_clock::now();
      cache.entries.splice(cache.entries.begin(), cache.entries, found);
      return;
    }
  }
  if (!value.has_result_fingerprint && sample.has_result_fingerprint) {
    value.has_result_fingerprint = true;
    value.input_fingerprint = sample.input_fingerprint;
    value.output_fingerprint = sample.output_fingerprint;
    value.fingerprint_input_rows = sample.fingerprint_input_rows;
    value.fingerprint_output_rows = sample.fingerprint_output_rows;
    value.result_feedback_stable = sample.result_feedback_stable;
  }
  value.completed_executions = SaturatingIncrement(value.completed_executions);
  value.expected_input_rows =
      Ewma(value.expected_input_rows, sample.expected_input_rows);
  value.producer_seconds_per_row =
      Ewma(value.producer_seconds_per_row, sample.producer_seconds_per_row);
  value.helper_latency_scale =
      Ewma(value.helper_latency_scale, sample.helper_latency_scale);
  // Keep the candidate set fixed until the cache entry is reset.
  if (value.frozen_shortlist.empty() && !sample.frozen_shortlist.empty()) {
    value.frozen_shortlist = sample.frozen_shortlist;
    if (value.frozen_shortlist.size() > kMaximumShortlistArms) {
      value.frozen_shortlist.resize(kMaximumShortlistArms);
    }
    value.arms.erase(std::remove_if(value.arms.begin(), value.arms.end(),
                                    [&](const SemanticBatchArmFeedback &arm) {
                                      return !ShortlistContainsArm(
                                          value.frozen_shortlist,
                                          arm.launch_rows, arm.target_rows);
                                    }),
                     value.arms.end());
  }
  for (const SemanticBatchArmFeedback &arm_sample : sample.arms) {
    if (arm_sample.completed_executions == 0 ||
        !IsPositiveFinite(arm_sample.mean_elapsed_seconds)) {
      continue;
    }
    if (!value.frozen_shortlist.empty() &&
        !ShortlistContainsArm(value.frozen_shortlist, arm_sample.launch_rows,
                              arm_sample.target_rows)) {
      // Ignore observations outside the fixed candidate set.
      continue;
    }
    const auto arm = std::find_if(
        value.arms.begin(), value.arms.end(), [&](const auto &candidate) {
          return candidate.launch_rows == arm_sample.launch_rows &&
                 candidate.target_rows == arm_sample.target_rows;
        });
    if (arm == value.arms.end()) {
      if (value.arms.size() >= kMaximumArmsPerFeedbackEntry) {
        const auto victim = std::min_element(
            value.arms.begin(), value.arms.end(),
            [](const auto &left, const auto &right) {
              return left.completed_executions < right.completed_executions;
            });
        value.arms.erase(victim);
      }
      value.arms.push_back(arm_sample);
      continue;
    }

    UpdateSemanticBatchArmFeedback(&*arm, arm_sample.mean_elapsed_seconds);
  }
  ApplyRegretBoundedObservation(&value, sample);
  found->updated_at = std::chrono::steady_clock::now();
  cache.entries.splice(cache.entries.begin(), cache.entries, found);
}

SemanticBatchPolicy ColdPolicy(const SemanticOperatorProfile &profile,
                               const SemanticBatchSelectionContext &context) {
  const size_t cap =
      SemanticProfiledBatchSafeCap(profile, context.hard_safe_cap_rows);
  const size_t selected_launch = SelectSemanticLaunchWatermark(
      profile, context.operator_kind,
      std::max<size_t>(1, context.predicate_count),
      context.maximum_observed_input_bytes);
  const size_t launch = std::clamp(selected_launch, size_t{1}, cap);
  // Output backpressure is deliberately independent from both input
  // watermarks.  In particular, choosing a large launch/target must not also
  // grow the ready-result backlog that an outer iterator can accumulate.
  const size_t refill = std::clamp(profile.min_batch_size, size_t{1}, cap);
  return {launch, cap, refill, cap};
}

std::vector<size_t> ProfiledCandidates(const SemanticOperatorProfile &profile,
                                       size_t cap, size_t cold_launch) {
  std::vector<size_t> candidates;
  candidates.reserve(profile.latency_curve.size() + 3);
  candidates.push_back(std::clamp(cold_launch, size_t{1}, cap));
  candidates.push_back(std::clamp(profile.min_batch_size, size_t{1}, cap));
  candidates.push_back(cap);
  const size_t minimum_effective_candidate =
      std::clamp(profile.min_batch_size, size_t{1}, cap);
  for (const SemanticBatchMeasurement &point : profile.latency_curve) {
    // Exclude launch sizes below the prefix already buffered by ViperFlow.
    if (point.batch_size >= minimum_effective_candidate &&
        point.batch_size <= cap && MeasurementLatency(point) > 0.0) {
      candidates.push_back(point.batch_size);
    }
  }
  std::sort(candidates.begin(), candidates.end());
  candidates.erase(std::unique(candidates.begin(), candidates.end()),
                   candidates.end());
  if (candidates.size() <= kMaximumCandidateValues) return candidates;

  // Bound the candidate set while retaining measured endpoints.
  std::vector<size_t> bounded;
  bounded.reserve(kMaximumCandidateValues);
  const auto add = [&](size_t value) {
    if (bounded.size() == kMaximumCandidateValues) return;
    value = std::clamp(value, size_t{1}, cap);
    if (std::find(bounded.begin(), bounded.end(), value) == bounded.end())
      bounded.push_back(value);
  };
  add(std::clamp(profile.min_batch_size, size_t{1}, cap));
  add(cap);
  add(cold_launch);
  const auto parallelism = std::lower_bound(
      candidates.begin(), candidates.end(),
      std::min(std::max<size_t>(1, profile.helper_parallelism), cap));
  if (parallelism != candidates.end()) add(*parallelism);

  for (size_t slot = 0; slot < kMaximumCandidateValues &&
                        bounded.size() < kMaximumCandidateValues;
       ++slot) {
    const size_t index =
        slot * (candidates.size() - 1) / (kMaximumCandidateValues - 1);
    add(candidates[index]);
  }
  for (size_t value : candidates) {
    if (bounded.size() == kMaximumCandidateValues) break;
    add(value);
  }
  std::sort(bounded.begin(), bounded.end());
  return bounded;
}

size_t RoundedExpectedRows(double expected_rows) {
  if (!IsPositiveFinite(expected_rows)) return 0;
  if (expected_rows >=
      static_cast<double>(std::numeric_limits<size_t>::max())) {
    return std::numeric_limits<size_t>::max();
  }
  return static_cast<size_t>(std::ceil(expected_rows));
}

bool UpdateSemanticBatchArmFeedback(SemanticBatchArmFeedback *feedback,
                                    double elapsed_seconds) {
  if (feedback == nullptr || !IsPositiveFinite(elapsed_seconds)) return false;

  if (feedback->completed_executions == 0 ||
      !IsPositiveFinite(feedback->mean_elapsed_seconds) ||
      !IsNonnegativeFinite(feedback->elapsed_m2_seconds_squared)) {
    feedback->completed_executions = 1;
    feedback->mean_elapsed_seconds = elapsed_seconds;
    feedback->ewma_elapsed_seconds = elapsed_seconds;
    feedback->elapsed_m2_seconds_squared = 0.0;
    feedback->recent_elapsed_seconds.clear();
    feedback->recent_elapsed_seconds.push_back(elapsed_seconds);
    return true;
  }

  feedback->recent_elapsed_seconds.push_back(elapsed_seconds);
  if (feedback->recent_elapsed_seconds.size() > kRobustRecentSampleCount) {
    feedback->recent_elapsed_seconds.erase(
        feedback->recent_elapsed_seconds.begin(),
        feedback->recent_elapsed_seconds.begin() +
            (feedback->recent_elapsed_seconds.size() -
             kRobustRecentSampleCount));
  }

  if (!IsPositiveFinite(feedback->ewma_elapsed_seconds))
    feedback->ewma_elapsed_seconds = feedback->mean_elapsed_seconds;
  feedback->ewma_elapsed_seconds =
      Ewma(feedback->ewma_elapsed_seconds, elapsed_seconds);

  const uint64_t next_count =
      SaturatingIncrement(feedback->completed_executions);
  if (next_count == feedback->completed_executions) return true;

  const double delta = elapsed_seconds - feedback->mean_elapsed_seconds;
  const double next_mean =
      feedback->mean_elapsed_seconds + delta / static_cast<double>(next_count);
  const double m2_increment = delta * (elapsed_seconds - next_mean);
  feedback->mean_elapsed_seconds = next_mean;
  feedback->completed_executions = next_count;
  if (std::isfinite(m2_increment)) {
    // The exact Welford term is nonnegative. Clamp a tiny negative caused by
    // floating-point cancellation instead of turning the arm into infinite
    // uncertainty.
    const double nonnegative_increment = std::max(0.0, m2_increment);
    if (nonnegative_increment <= std::numeric_limits<double>::max() -
                                     feedback->elapsed_m2_seconds_squared) {
      feedback->elapsed_m2_seconds_squared += nonnegative_increment;
    } else {
      feedback->elapsed_m2_seconds_squared = std::numeric_limits<double>::max();
    }
  } else {
    feedback->elapsed_m2_seconds_squared = std::numeric_limits<double>::max();
  }
  return true;
}

size_t SemanticProfiledBatchSafeCap(const SemanticOperatorProfile &profile,
                                    size_t hard_safe_cap_rows) {
  size_t cap = std::max<size_t>(1, profile.max_batch_size);
  if (hard_safe_cap_rows != 0) cap = std::min(cap, hard_safe_cap_rows);

  size_t largest_profiled = 0;
  for (const SemanticBatchMeasurement &point : profile.latency_curve) {
    if (point.batch_size != 0 && MeasurementLatency(point) > 0.0) {
      largest_profiled = std::max(largest_profiled, point.batch_size);
    }
  }
  if (largest_profiled != 0) cap = std::min(cap, largest_profiled);
  return std::max<size_t>(1, cap);
}

SemanticFillBeforePopSimulationResult SimulateSemanticFillBeforePop(
    const SemanticOperatorProfile &profile,
    const SemanticFillBeforePopSimulationInput &input) {
  SemanticFillBeforePopSimulationResult result;
  if (input.total_input_rows == 0) return result;

  const size_t profile_cap = SemanticProfiledBatchSafeCap(profile);
  const size_t policy_cap =
      std::min(profile_cap, std::max<size_t>(1, input.policy.safe_cap_rows));
  const size_t target =
      std::clamp(input.policy.target_rows, size_t{1}, policy_cap);
  const size_t launch = std::clamp(input.policy.launch_rows, size_t{1}, target);
  const double producer_seconds_per_row =
      IsNonnegativeFinite(input.producer_seconds_per_row)
          ? input.producer_seconds_per_row
          : 0.0;
  const double latency_scale =
      IsPositiveFinite(input.helper_latency_scale)
          ? std::clamp(input.helper_latency_scale, kMinimumLatencyScale,
                       kMaximumLatencyScale)
          : 1.0;

  double now = 0.0;
  double helper_completion = 0.0;
  double first_batch_completion = 0.0;
  size_t remaining = input.total_input_rows;

  struct CycleState {
    size_t remaining;
    double now;
    double helper_busy_seconds;
    double producer_stall_seconds;
    uint64_t submitted_batches;
  };
  std::unordered_map<size_t, CycleState> cycle_states;

  const auto submit = [&](size_t rows, double submission_time,
                          SemanticFillBeforePopSimulationResult *out,
                          double *completion) {
    const double service =
        ProfileBatchLatencySeconds(profile, rows) * latency_scale;
    *completion = submission_time + service;
    out->helper_busy_seconds =
        SaturatingAddSeconds(out->helper_busy_seconds, service);
    out->submitted_batches = SaturatingIncrement(out->submitted_batches);
    out->largest_submitted_batch_rows =
        std::max(out->largest_submitted_batch_rows, rows);
  };

  const size_t first_rows = std::min(remaining, launch);
  now = static_cast<double>(first_rows) * producer_seconds_per_row;
  remaining -= first_rows;
  submit(first_rows, now, &result, &helper_completion);
  first_batch_completion = helper_completion;
  size_t previous_batch_rows = first_rows;

  while (remaining != 0) {
    // Fast-forward repeated scheduling states.
    if (remaining >= target) {
      const auto inserted = cycle_states.emplace(
          previous_batch_rows,
          CycleState{remaining, now, result.helper_busy_seconds,
                     result.producer_stall_seconds, result.submitted_batches});
      if (!inserted.second) {
        const CycleState &prior = inserted.first->second;
        const size_t rows_per_cycle = prior.remaining - remaining;
        if (rows_per_cycle != 0) {
          const size_t repetitions = (remaining - target) / rows_per_cycle;
          if (repetitions != 0) {
            const double cycle_seconds = now - prior.now;
            const double cycle_helper_seconds =
                result.helper_busy_seconds - prior.helper_busy_seconds;
            const double cycle_stall_seconds =
                result.producer_stall_seconds - prior.producer_stall_seconds;
            const uint64_t cycle_batches =
                result.submitted_batches - prior.submitted_batches;
            remaining -= repetitions * rows_per_cycle;
            now = SaturatingAddSeconds(
                now, static_cast<double>(repetitions) * cycle_seconds);
            result.helper_busy_seconds = SaturatingAddSeconds(
                result.helper_busy_seconds,
                static_cast<double>(repetitions) * cycle_helper_seconds);
            result.producer_stall_seconds = SaturatingAddSeconds(
                result.producer_stall_seconds,
                static_cast<double>(repetitions) * cycle_stall_seconds);
            result.submitted_batches = SaturatingAddProduct(
                result.submitted_batches, cycle_batches, repetitions);
            helper_completion =
                now + ProfileBatchLatencySeconds(profile, previous_batch_rows) *
                          latency_scale;
            cycle_states.clear();
          }
        }
      }
    }

    const size_t launch_rows = std::min(remaining, launch);
    const size_t target_rows = std::min(remaining, target);
    const double launch_time =
        now + static_cast<double>(launch_rows) * producer_seconds_per_row;
    const double target_time =
        now + static_cast<double>(target_rows) * producer_seconds_per_row;

    size_t batch_rows = launch_rows;
    double submission_time = launch_time;
    if (remaining < launch) {
      // EOF is a correctness boundary: finish the residual, then wait for the
      // prior request before submitting it below the launch watermark.
      batch_rows = remaining;
      submission_time = std::max(target_time, helper_completion);
    } else if (helper_completion <= launch_time) {
      // The helper is observed idle when the next launch watermark is met.
      batch_rows = launch;
      submission_time = launch_time;
    } else if (producer_seconds_per_row > 0.0 &&
               helper_completion < target_time) {
      // The helper completes while the producer is filling. It is observed on
      // the first subsequent row, which becomes this batch's final row.
      const double rows_until_idle =
          std::ceil((helper_completion - now) / producer_seconds_per_row);
      batch_rows = static_cast<size_t>(
          std::clamp(rows_until_idle, static_cast<double>(launch_rows),
                     static_cast<double>(target_rows)));
      submission_time =
          now + static_cast<double>(batch_rows) * producer_seconds_per_row;
      submission_time = std::max(submission_time, helper_completion);
    } else {
      // The target fills first. ViperFlow blocks at the cap until the in-flight
      // helper request completes, then immediately submits the full buffer.
      batch_rows = target_rows;
      submission_time = std::max(target_time, helper_completion);
      if (target_rows == target) {
        result.producer_stall_seconds = SaturatingAddSeconds(
            result.producer_stall_seconds, submission_time - target_time);
      }
    }

    remaining -= batch_rows;
    now = submission_time;

    // The iterator latches one fill-before-pop obligation at the start of its
    // public Read(). If input remains, it cannot fetch/expose the first result
    // until this second request is submitted, even when the first result batch
    // already exceeds the output refill watermark.
    if (result.first_output_seconds == 0.0) {
      result.first_output_seconds = now;
    }
    submit(batch_rows, now, &result, &helper_completion);
    previous_batch_rows = batch_rows;
  }

  if (result.first_output_seconds == 0.0) {
    // With only one request, EOF releases the latch and the first result is
    // exposed at completion.
    result.first_output_seconds = first_batch_completion;
  }
  result.elapsed_seconds = helper_completion;
  return result;
}

SemanticBatchPolicy SelectSemanticBatchPolicy(
    const SemanticOperatorProfile &profile,
    const SemanticBatchSelectionContext &context,
    SemanticBatchSelectionDiagnostics *diagnostics) {
  SemanticBatchSelectionDiagnostics selection;
  const auto finish = [&](SemanticBatchPolicy policy,
                          SemanticBatchSelectionPhase phase) {
    selection.phase = phase;
    if (diagnostics != nullptr) *diagnostics = selection;
    return policy;
  };

  const SemanticBatchPolicy cold = ColdPolicy(profile, context);

  // Binary semantic joins have query-specific build/index behavior that the
  // unary initialization curve does not measure. Never transfer unary batch
  // feedback into that operator class.
  const SemanticBatchLearnedFeedback &learned = context.learned;
  if (context.operator_kind == SemanticBatchOperatorKind::kBinaryJoin ||
      !learned.result_feedback_stable || learned.completed_executions == 0 ||
      !IsPositiveFinite(learned.expected_input_rows) ||
      !IsNonnegativeFinite(learned.producer_seconds_per_row) ||
      !IsPositiveFinite(learned.helper_latency_scale) ||
      profile.latency_curve.empty()) {
    return finish(cold, SemanticBatchSelectionPhase::kColdStart);
  }

  const size_t expected_rows = RoundedExpectedRows(learned.expected_input_rows);
  if (expected_rows == 0)
    return finish(cold, SemanticBatchSelectionPhase::kColdStart);

  // Candidate sizes must come from configured or measured values.
  const std::vector<size_t> candidates =
      ProfiledCandidates(profile, cold.safe_cap_rows, cold.launch_rows);

  struct RankedArm {
    SemanticBatchPolicy policy;
    SemanticFillBeforePopSimulationResult simulation;
  };
  std::vector<RankedArm> ranked_arms;
  ranked_arms.reserve(candidates.size() * (candidates.size() + 1) / 2);
  for (size_t launch : candidates) {
    for (size_t target : candidates) {
      if (launch > target || target > cold.safe_cap_rows) continue;
      SemanticBatchPolicy candidate;
      candidate.launch_rows = launch;
      candidate.target_rows = target;
      candidate.refill_watermark_rows =
          std::clamp(profile.min_batch_size, size_t{1}, target);
      candidate.safe_cap_rows = cold.safe_cap_rows;

      SemanticFillBeforePopSimulationInput simulation_input{
          expected_rows, learned.producer_seconds_per_row,
          learned.helper_latency_scale, candidate};
      ranked_arms.push_back({candidate, SimulateSemanticFillBeforePop(
                                            profile, simulation_input)});
    }
  }
  std::sort(
      ranked_arms.begin(), ranked_arms.end(),
      [](const RankedArm &left, const RankedArm &right) {
        // Preserve a strict weak ordering.
        if (left.simulation.elapsed_seconds != right.simulation.elapsed_seconds)
          return left.simulation.elapsed_seconds <
                 right.simulation.elapsed_seconds;
        if (left.simulation.first_output_seconds !=
            right.simulation.first_output_seconds)
          return left.simulation.first_output_seconds <
                 right.simulation.first_output_seconds;
        if (left.policy.target_rows != right.policy.target_rows) {
          return left.policy.target_rows < right.policy.target_rows;
        }
        return left.policy.launch_rows < right.policy.launch_rows;
      });
  if (ranked_arms.empty())
    return finish(cold, SemanticBatchSelectionPhase::kColdStart);

  SemanticBatchPolicy best = ranked_arms.front().policy;

  // Normalize values from malformed profiles.
  best.target_rows =
      std::clamp(best.target_rows, size_t{1}, cold.safe_cap_rows);
  best.launch_rows = std::clamp(best.launch_rows, size_t{1}, best.target_rows);
  best.refill_watermark_rows =
      std::clamp(best.refill_watermark_rows, size_t{1}, best.target_rows);
  best.safe_cap_rows = cold.safe_cap_rows;

  // Limit the candidate set to distinct safe schedules.
  const auto same_predicted_schedule = [&](const RankedArm &left,
                                           const RankedArm &right) {
    constexpr double kScheduleTimeToleranceSeconds = 1.0e-9;
    return left.simulation.submitted_batches ==
               right.simulation.submitted_batches &&
           left.simulation.largest_submitted_batch_rows ==
               right.simulation.largest_submitted_batch_rows &&
           std::abs(left.simulation.first_output_seconds -
                    right.simulation.first_output_seconds) <=
               kScheduleTimeToleranceSeconds &&
           std::abs(left.simulation.helper_busy_seconds -
                    right.simulation.helper_busy_seconds) <=
               kScheduleTimeToleranceSeconds &&
           std::abs(left.simulation.producer_stall_seconds -
                    right.simulation.producer_stall_seconds) <=
               kScheduleTimeToleranceSeconds;
  };

  std::vector<SemanticBatchPolicy> shortlist;
  shortlist.reserve(kMaximumShortlistArms);
  const auto add_shortlist = [&](SemanticBatchPolicy candidate) {
    candidate.target_rows =
        std::clamp(candidate.target_rows, size_t{1}, cold.safe_cap_rows);
    candidate.launch_rows =
        std::clamp(candidate.launch_rows, size_t{1}, candidate.target_rows);
    candidate.refill_watermark_rows =
        std::clamp(profile.min_batch_size, size_t{1}, candidate.target_rows);
    candidate.safe_cap_rows = cold.safe_cap_rows;
    const bool duplicate = std::any_of(
        shortlist.begin(), shortlist.end(),
        [&](const auto &arm) { return SamePolicyArm(arm, candidate); });
    if (!duplicate && shortlist.size() < kMaximumShortlistArms)
      shortlist.push_back(candidate);
  };

  if (!learned.frozen_shortlist.empty()) {
    // Normalize the fixed candidate set against the current cap.
    for (const SemanticBatchPolicy &candidate : learned.frozen_shortlist) {
      add_shortlist(candidate);
    }
  } else {
    // Include the policy used for the first observation.
    add_shortlist(cold);

    const size_t minimum =
        std::clamp(profile.min_batch_size, size_t{1}, cold.safe_cap_rows);
    const size_t desired_small_target = SaturatingAdd(minimum, minimum);
    auto small_target_position =
        std::lower_bound(candidates.begin(), candidates.end(),
                         std::min(desired_small_target, cold.safe_cap_rows));
    if (small_target_position == candidates.end())
      small_target_position = std::prev(candidates.end());
    add_shortlist(
        {minimum, *small_target_position, minimum, cold.safe_cap_rows});

    // Include a measured target near one quarter of the safe range.
    const size_t quarter_cap = SaturatingAdd(cold.safe_cap_rows, size_t{3}) / 4;
    const size_t desired_mid_target =
        std::max(desired_small_target, quarter_cap);
    auto mid_target_position =
        std::lower_bound(candidates.begin(), candidates.end(),
                         std::min(desired_mid_target, cold.safe_cap_rows));
    if (mid_target_position == candidates.end())
      mid_target_position = std::prev(candidates.end());
    add_shortlist({minimum, *mid_target_position, minimum, cold.safe_cap_rows});

    add_shortlist({minimum, cold.safe_cap_rows, minimum, cold.safe_cap_rows});

    const size_t desired_helper_width =
        std::clamp(std::max<size_t>(1, profile.helper_parallelism), minimum,
                   cold.safe_cap_rows);
    auto helper_width_position = std::lower_bound(
        candidates.begin(), candidates.end(), desired_helper_width);
    if (helper_width_position == candidates.end())
      helper_width_position = std::prev(candidates.end());
    add_shortlist({*helper_width_position, cold.safe_cap_rows, minimum,
                   cold.safe_cap_rows});

    // Include a second helper-width launch.
    const size_t desired_two_helper_width =
        std::clamp(SaturatingAdd(desired_helper_width, desired_helper_width),
                   minimum, cold.safe_cap_rows);
    auto two_helper_width_position = std::lower_bound(
        candidates.begin(), candidates.end(), desired_two_helper_width);
    if (two_helper_width_position == candidates.end())
      two_helper_width_position = std::prev(candidates.end());
    add_shortlist({*two_helper_width_position, cold.safe_cap_rows, minimum,
                   cold.safe_cap_rows});
    add_shortlist(
        {cold.safe_cap_rows, cold.safe_cap_rows, minimum, cold.safe_cap_rows});

    // Fill remaining slots with distinct request shapes.
    std::vector<const RankedArm *> ranked_schedule_shapes;
    ranked_schedule_shapes.reserve(2);
    for (const RankedArm &ranked : ranked_arms) {
      const bool duplicate_shape =
          std::any_of(ranked_schedule_shapes.begin(),
                      ranked_schedule_shapes.end(), [&](const auto *selected) {
                        return same_predicted_schedule(*selected, ranked);
                      });
      if (duplicate_shape) continue;
      const size_t prior_size = shortlist.size();
      add_shortlist(ranked.policy);
      if (shortlist.size() != prior_size) {
        ranked_schedule_shapes.push_back(&ranked);
      }
      if (ranked_schedule_shapes.size() == 2 ||
          shortlist.size() == kMaximumShortlistArms) {
        break;
      }
    }
  }

  if (shortlist.empty()) {
    // ProfiledCandidates() always contributes the cold envelope, but retain a
    // defensive fallback for malformed input.
    add_shortlist(best);
  }
  selection.safe_shortlist = shortlist;

  const auto arm_feedback = [&](const SemanticBatchPolicy &policy)
      -> const SemanticBatchArmFeedback * {
    const auto found = std::find_if(
        learned.arms.begin(), learned.arms.end(), [&](const auto &arm) {
          return arm.launch_rows == policy.launch_rows &&
                 arm.target_rows == policy.target_rows &&
                 arm.completed_executions != 0 &&
                 IsPositiveFinite(arm.mean_elapsed_seconds) &&
                 IsNonnegativeFinite(arm.elapsed_m2_seconds_squared);
        });
    return found == learned.arms.end() ? nullptr : &*found;
  };
  const auto arm_samples = [&](const SemanticBatchPolicy &policy) {
    const SemanticBatchArmFeedback *feedback = arm_feedback(policy);
    return feedback == nullptr ? uint64_t{0} : feedback->completed_executions;
  };

  // Use the fixed candidate selector until an incumbent is available.
  if (learned.has_incumbent) {
    SemanticBatchPolicy incumbent_policy = learned.incumbent_policy;
    incumbent_policy.target_rows =
        std::clamp(incumbent_policy.target_rows, size_t{1}, cold.safe_cap_rows);
    incumbent_policy.launch_rows = std::clamp(
        incumbent_policy.launch_rows, size_t{1}, incumbent_policy.target_rows);
    incumbent_policy.refill_watermark_rows = std::clamp(
        profile.min_batch_size, size_t{1}, incumbent_policy.target_rows);
    incumbent_policy.safe_cap_rows = cold.safe_cap_rows;

    const auto incumbent_ranked = std::find_if(
        ranked_arms.begin(), ranked_arms.end(), [&](const RankedArm &entry) {
          return SamePolicyArm(entry.policy, incumbent_policy);
        });
    if (incumbent_ranked != ranked_arms.end() &&
        ShortlistContainsArm(shortlist, incumbent_policy.launch_rows,
                             incumbent_policy.target_rows)) {
      if (learned.recovery_required) {
        return finish(incumbent_policy,
                      SemanticBatchSelectionPhase::kIncumbentRecovery);
      }

      if (arm_samples(incumbent_policy) < kIncumbentWarmupSamples) {
        return finish(incumbent_policy,
                      SemanticBatchSelectionPhase::kIncumbentWarmup);
      }

      if (learned.incumbent_executions_since_probe <
          kIncumbentExecutionsBetweenProbes) {
        return finish(incumbent_policy,
                      SemanticBatchSelectionPhase::kRegretBoundedExploit);
      }

      // Compare only adjacent policies in the fixed candidate set.
      std::vector<SemanticBatchPolicy> neighbors;
      neighbors.reserve(4);
      const SemanticBatchPolicy *lower_launch = nullptr;
      const SemanticBatchPolicy *upper_launch = nullptr;
      const SemanticBatchPolicy *lower_target = nullptr;
      const SemanticBatchPolicy *upper_target = nullptr;
      for (const SemanticBatchPolicy &candidate : shortlist) {
        if (SamePolicyArm(candidate, incumbent_policy)) continue;
        if (candidate.target_rows == incumbent_policy.target_rows) {
          if (candidate.launch_rows < incumbent_policy.launch_rows &&
              (lower_launch == nullptr ||
               candidate.launch_rows > lower_launch->launch_rows)) {
            lower_launch = &candidate;
          }
          if (candidate.launch_rows > incumbent_policy.launch_rows &&
              (upper_launch == nullptr ||
               candidate.launch_rows < upper_launch->launch_rows)) {
            upper_launch = &candidate;
          }
        }
        if (candidate.launch_rows == incumbent_policy.launch_rows) {
          if (candidate.target_rows < incumbent_policy.target_rows &&
              (lower_target == nullptr ||
               candidate.target_rows > lower_target->target_rows)) {
            lower_target = &candidate;
          }
          if (candidate.target_rows > incumbent_policy.target_rows &&
              (upper_target == nullptr ||
               candidate.target_rows < upper_target->target_rows)) {
            upper_target = &candidate;
          }
        }
      }
      const auto add_neighbor = [&](const SemanticBatchPolicy *candidate) {
        if (candidate != nullptr &&
            std::none_of(neighbors.begin(), neighbors.end(),
                         [&](const auto &existing) {
                           return SamePolicyArm(existing, *candidate);
                         })) {
          neighbors.push_back(*candidate);
        }
      };
      add_neighbor(lower_launch);
      add_neighbor(upper_launch);
      add_neighbor(lower_target);
      add_neighbor(upper_target);

      const auto raw_arm_feedback = [&](const SemanticBatchPolicy &policy)
          -> const SemanticBatchArmFeedback * {
        const auto found = std::find_if(
            learned.arms.begin(), learned.arms.end(), [&](const auto &arm) {
              return arm.launch_rows == policy.launch_rows &&
                     arm.target_rows == policy.target_rows;
            });
        return found == learned.arms.end() ? nullptr : &*found;
      };
      const uint64_t incumbent_batches =
          incumbent_ranked->simulation.submitted_batches;
      const uint64_t half_incumbent_batches =
          incumbent_batches / 2 + incumbent_batches % 2;
      const uint64_t maximum_probe_batches = SaturatingAddUint64(
          incumbent_batches,
          std::max<uint64_t>(uint64_t{1}, half_incumbent_batches));
      const double incumbent_elapsed =
          incumbent_ranked->simulation.elapsed_seconds;

      struct EligibleProbe {
        const RankedArm *ranked{nullptr};
      };
      const auto best_eligible_probe = [&](bool suspended_only) {
        EligibleProbe selected;
        for (const SemanticBatchPolicy &candidate : neighbors) {
          const SemanticBatchArmFeedback *feedback =
              raw_arm_feedback(candidate);
          if (feedback != nullptr && feedback->quarantined) continue;
          if (suspended_only) {
            if (feedback == nullptr || !feedback->suspended) continue;
          } else if (feedback != nullptr &&
                     (feedback->suspended || feedback->indifferent)) {
            continue;
          }

          const auto ranked =
              std::find_if(ranked_arms.begin(), ranked_arms.end(),
                           [&](const RankedArm &entry) {
                             return SamePolicyArm(entry.policy, candidate);
                           });
          if (ranked == ranked_arms.end() ||
              !IsPositiveFinite(incumbent_elapsed) ||
              !IsPositiveFinite(ranked->simulation.elapsed_seconds)) {
            continue;
          }
          const double predicted_ratio =
              ranked->simulation.elapsed_seconds / incumbent_elapsed;
          if (!IsPositiveFinite(predicted_ratio) ||
              predicted_ratio > kMaximumPredictedProbeRatio ||
              ranked->simulation.submitted_batches > maximum_probe_batches) {
            continue;
          }
          if (selected.ranked == nullptr ||
              ranked->simulation.elapsed_seconds <
                  selected.ranked->simulation.elapsed_seconds ||
              (ranked->simulation.elapsed_seconds ==
                   selected.ranked->simulation.elapsed_seconds &&
               (ranked->policy.target_rows <
                    selected.ranked->policy.target_rows ||
                (ranked->policy.target_rows ==
                     selected.ranked->policy.target_rows &&
                 ranked->policy.launch_rows <
                     selected.ranked->policy.launch_rows)))) {
            selected = {&*ranked};
          }
        }
        return selected;
      };

      const bool sparse_reprobe_due =
          learned.completed_executions != 0 &&
          learned.completed_executions % kSparseReprobeInterval == 0;
      EligibleProbe probe;
      SemanticBatchSelectionPhase probe_phase =
          SemanticBatchSelectionPhase::kGuardedProbe;
      if (sparse_reprobe_due) {
        probe = best_eligible_probe(/*suspended_only=*/true);
        if (probe.ranked != nullptr)
          probe_phase = SemanticBatchSelectionPhase::kSparseReprobe;
      }
      if (probe.ranked == nullptr)
        probe = best_eligible_probe(/*suspended_only=*/false);

      if (probe.ranked != nullptr) {
        return finish(probe.ranked->policy, probe_phase);
      }
      return finish(incumbent_policy,
                    SemanticBatchSelectionPhase::kRegretBoundedExploit);
    }
  }

  // Sample every candidate three times in deterministic order.
  for (uint64_t required_samples = 1;
       required_samples <= kShortlistMinimumSamples; ++required_samples) {
    for (size_t offset = 0; offset < shortlist.size(); ++offset) {
      size_t index = offset;
      if (required_samples == 2) {
        index = shortlist.size() - 1 - offset;
      } else if (required_samples == 3) {
        index = (offset + 1) % shortlist.size();
      }
      const SemanticBatchPolicy &candidate = shortlist[index];
      if (arm_samples(candidate) < required_samples) {
        return finish(candidate,
                      SemanticBatchSelectionPhase::kPrioritizedExploration);
      }
    }
  }

  SemanticBatchPolicy empirical_best = shortlist.front();
  const SemanticBatchArmFeedback *best_feedback = nullptr;
  for (const SemanticBatchPolicy &candidate : shortlist) {
    const SemanticBatchArmFeedback *feedback = arm_feedback(candidate);
    if (feedback == nullptr ||
        feedback->completed_executions < kShortlistMinimumSamples) {
      continue;
    }
    constexpr double kEmpiricalTieToleranceSeconds = 1.0e-6;
    if (best_feedback == nullptr ||
        RobustArmElapsedSeconds(*feedback) + kEmpiricalTieToleranceSeconds <
            RobustArmElapsedSeconds(*best_feedback) ||
        (std::abs(RobustArmElapsedSeconds(*feedback) -
                  RobustArmElapsedSeconds(*best_feedback)) <=
             kEmpiricalTieToleranceSeconds &&
         (candidate.target_rows < empirical_best.target_rows ||
          (candidate.target_rows == empirical_best.target_rows &&
           candidate.launch_rows < empirical_best.launch_rows)))) {
      empirical_best = candidate;
      best_feedback = feedback;
    }
  }
  if (best_feedback == nullptr) {
    // Fall back when cached feedback is incomplete.
    return finish(best, SemanticBatchSelectionPhase::kPrioritizedExploration);
  }

  // Prefer smaller buffers when observations differ by at most 5%.
  SemanticBatchPolicy selected_best = empirical_best;
  const double indifference_limit =
      RobustArmElapsedSeconds(*best_feedback) * kIndifferenceRatio;
  for (const SemanticBatchPolicy &candidate : shortlist) {
    const SemanticBatchArmFeedback *feedback = arm_feedback(candidate);
    if (feedback == nullptr ||
        RobustArmElapsedSeconds(*feedback) > indifference_limit) {
      continue;
    }
    if (candidate.target_rows < selected_best.target_rows ||
        (candidate.target_rows == selected_best.target_rows &&
         candidate.launch_rows < selected_best.launch_rows)) {
      selected_best = candidate;
    }
  }

  struct ConfidenceInterval {
    SemanticBatchPolicy policy;
    const SemanticBatchArmFeedback *feedback;
    double lower_seconds;
    double upper_seconds;
    double half_width_seconds;
  };
  const auto student_multiplier = [](uint64_t samples) {
    if (samples <= 2) return 12.706;
    if (samples == 3) return 4.303;
    if (samples == 4) return 3.182;
    if (samples == 5) return 2.776;
    if (samples == 6) return 2.571;
    if (samples == 7) return 2.447;
    if (samples == 8) return 2.365;
    if (samples == 9) return 2.306;
    if (samples == 10) return 2.262;
    if (samples <= 15) return 2.228;
    if (samples <= 20) return 2.131;
    if (samples <= 30) return 2.086;
    return 1.960;
  };
  std::vector<ConfidenceInterval> intervals;
  intervals.reserve(shortlist.size());
  for (const SemanticBatchPolicy &candidate : shortlist) {
    const SemanticBatchArmFeedback *feedback = arm_feedback(candidate);
    if (feedback == nullptr ||
        feedback->completed_executions < kShortlistMinimumSamples) {
      continue;
    }
    uint64_t uncertainty_sample_count = feedback->completed_executions;
    double uncertainty_mean = feedback->mean_elapsed_seconds;
    double uncertainty_m2 = feedback->elapsed_m2_seconds_squared;
    std::vector<double> recent_samples;
    recent_samples.reserve(feedback->recent_elapsed_seconds.size());
    for (double sample : feedback->recent_elapsed_seconds) {
      if (IsPositiveFinite(sample)) recent_samples.push_back(sample);
    }
    if (recent_samples.size() >= kShortlistMinimumSamples) {
      // Use the same bounded window for ranking and variance.
      uncertainty_sample_count = recent_samples.size();
      uncertainty_mean = 0.0;
      uncertainty_m2 = 0.0;
      uint64_t seen = 0;
      for (double sample : recent_samples) {
        ++seen;
        const double delta = sample - uncertainty_mean;
        uncertainty_mean += delta / static_cast<double>(seen);
        uncertainty_m2 += delta * (sample - uncertainty_mean);
      }
    }
    const double samples = static_cast<double>(uncertainty_sample_count);
    const double variance =
        uncertainty_m2 / static_cast<double>(uncertainty_sample_count - 1);
    double half_width = student_multiplier(uncertainty_sample_count) *
                        std::sqrt(variance / samples);
    // Apply a decreasing floor to zero-variance intervals.
    const double lifetime_samples = static_cast<double>(
        std::max<uint64_t>(uint64_t{1}, feedback->completed_executions));
    const double noise_floor =
        std::max(1.0e-6, kConfidenceNoiseFloorRatio * uncertainty_mean /
                             std::sqrt(lifetime_samples));
    if (!IsNonnegativeFinite(half_width))
      half_width = std::numeric_limits<double>::max();
    half_width = std::max(half_width, noise_floor);
    const double robust_center = RobustArmElapsedSeconds(*feedback);
    const double lower = std::max(0.0, robust_center - half_width);
    const double upper =
        half_width > std::numeric_limits<double>::max() - robust_center
            ? std::numeric_limits<double>::max()
            : robust_center + half_width;
    intervals.push_back({candidate, feedback, lower, upper, half_width});
  }

  // Anchor intervals to the policy used between sampling runs.
  const auto confidence_anchor_position =
      std::find_if(intervals.begin(), intervals.end(), [&](const auto &entry) {
        return SamePolicyArm(entry.policy, selected_best);
      });
  if (confidence_anchor_position == intervals.end()) {
    return finish(selected_best,
                  SemanticBatchSelectionPhase::kIncumbentExploit);
  }
  const ConfidenceInterval &incumbent = *confidence_anchor_position;
  const bool confidence_separated =
      std::all_of(intervals.begin(), intervals.end(), [&](const auto &entry) {
        return SamePolicyArm(entry.policy, incumbent.policy) ||
               entry.lower_seconds > incumbent.upper_seconds;
      });
  const bool indifference_zone_stable =
      std::all_of(intervals.begin(), intervals.end(), [&](const auto &entry) {
        const bool empirically_equivalent =
            RobustArmElapsedSeconds(*entry.feedback) <= indifference_limit;
        const bool confidently_slower =
            entry.lower_seconds > incumbent.upper_seconds;
        return empirically_equivalent || confidently_slower;
      });

  const bool stable_best_tested =
      confidence_separated || indifference_zone_stable;
  if (!stable_best_tested &&
      learned.completed_executions % kConfidenceRaceInterval == 0) {
    // Periodically sample the widest overlapping competing interval.
    const ConfidenceInterval *race = nullptr;
    for (const ConfidenceInterval &entry : intervals) {
      const bool is_incumbent = SamePolicyArm(entry.policy, incumbent.policy);
      const bool materially_slower =
          RobustArmElapsedSeconds(*entry.feedback) > indifference_limit;
      const bool still_ambiguous =
          entry.lower_seconds <= incumbent.upper_seconds;
      // Do not select the incumbent as its own comparison.
      if (is_incumbent || !materially_slower || !still_ambiguous) {
        continue;
      }
      if (race == nullptr ||
          entry.half_width_seconds > race->half_width_seconds ||
          (entry.half_width_seconds == race->half_width_seconds &&
           entry.feedback->completed_executions <
               race->feedback->completed_executions) ||
          (entry.half_width_seconds == race->half_width_seconds &&
           entry.feedback->completed_executions ==
               race->feedback->completed_executions &&
           (entry.policy.target_rows < race->policy.target_rows ||
            (entry.policy.target_rows == race->policy.target_rows &&
             entry.policy.launch_rows < race->policy.launch_rows)))) {
        race = &entry;
      }
    }
    if (race != nullptr) {
      return finish(race->policy, SemanticBatchSelectionPhase::kConfidenceRace);
    }
  }

  if (!stable_best_tested) {
    // Use the current empirical best between periodic samples.
    return finish(selected_best, SemanticBatchSelectionPhase::kEmpiricalBest);
  }

  // Periodically sample another policy after initial coverage.
  if (learned.completed_executions != 0 &&
      learned.completed_executions % kSparseReprobeInterval == 0 &&
      shortlist.size() > 1) {
    const size_t start = static_cast<size_t>(
        (learned.completed_executions / kSparseReprobeInterval) %
        shortlist.size());
    for (size_t offset = 0; offset < shortlist.size(); ++offset) {
      const SemanticBatchPolicy &candidate =
          shortlist[(start + offset) % shortlist.size()];
      if (!SamePolicyArm(candidate, selected_best)) {
        return finish(candidate, SemanticBatchSelectionPhase::kSparseReprobe);
      }
    }
  }
  return finish(selected_best, SemanticBatchSelectionPhase::kEmpiricalBest);
}

}  // namespace

SemanticBatchController::SemanticBatchController(
    const SemanticOperatorProfile &profile,
    SemanticBatchOperatorKind operator_kind, size_t predicate_count,
    size_t maximum_observed_input_bytes, std::string feedback_key,
    size_t hard_safe_cap_rows)
    : m_profile(profile),
      m_operator_kind(operator_kind),
      m_predicate_count(std::max<size_t>(1, predicate_count)),
      m_maximum_observed_input_bytes(maximum_observed_input_bytes),
      m_feedback_key(std::move(feedback_key)),
      m_hard_safe_cap_rows(hard_safe_cap_rows),
      m_cache_key(MakeCacheKey(
          m_feedback_key, m_operator_kind, m_predicate_count,
          m_maximum_observed_input_bytes, m_hard_safe_cap_rows, m_profile)) {
  SemanticBatchSelectionContext context;
  context.operator_kind = m_operator_kind;
  context.predicate_count = m_predicate_count;
  context.maximum_observed_input_bytes = maximum_observed_input_bytes;
  context.hard_safe_cap_rows = hard_safe_cap_rows;
  if (m_operator_kind != SemanticBatchOperatorKind::kBinaryJoin) {
    ReadFeedback(m_cache_key, &context.learned);
  }
  m_policy =
      SelectSemanticBatchPolicy(m_profile, context, &m_selection_diagnostics);
  m_execution_started_at = std::chrono::steady_clock::now();
}

void SemanticBatchController::RecordInputRows(size_t rows) {
  if (m_stats.completed) return;
  m_stats.total_input_rows = SaturatingAdd(m_stats.total_input_rows, rows);
}

void SemanticBatchController::RecordProducerActiveSeconds(double seconds) {
  if (m_stats.completed || !IsNonnegativeFinite(seconds)) return;
  m_stats.producer_active_seconds =
      SaturatingAddSeconds(m_stats.producer_active_seconds, seconds);
}

void SemanticBatchController::RecordSuccessfulBatch(size_t rows,
                                                    double service_seconds) {
  if (m_stats.completed || rows == 0 || !IsPositiveFinite(service_seconds)) {
    return;
  }
  m_stats.successful_batch_count =
      SaturatingIncrement(m_stats.successful_batch_count);
  m_stats.successful_batch_rows =
      SaturatingAdd(m_stats.successful_batch_rows, rows);
  m_stats.successful_batch_service_seconds = SaturatingAddSeconds(
      m_stats.successful_batch_service_seconds, service_seconds);
  m_stats.profiled_batch_service_seconds =
      SaturatingAddSeconds(m_stats.profiled_batch_service_seconds,
                           ProfileBatchLatencySeconds(m_profile, rows));
}

void SemanticBatchController::RecordResultFingerprint(
    uint64_t input_fingerprint, uint64_t output_fingerprint,
    size_t output_rows) {
  if (m_stats.completed) return;
  m_stats.has_result_fingerprint = true;
  m_stats.input_fingerprint = input_fingerprint;
  m_stats.output_fingerprint = output_fingerprint;
  m_stats.fingerprint_output_rows = output_rows;
}

void SemanticBatchController::RecordProviderEpoch(std::string provider_epoch) {
  if (m_stats.completed) return;
  if (provider_epoch.empty()) {
    m_stats.provider_epoch_changed = true;
    return;
  }
  if (!m_stats.provider_epoch_observed) {
    m_stats.provider_epoch = std::move(provider_epoch);
    m_stats.provider_epoch_observed = true;
    return;
  }
  if (m_stats.provider_epoch != provider_epoch)
    m_stats.provider_epoch_changed = true;
}

void SemanticBatchController::RecordMemoryForcedBatch() {
  if (!m_stats.completed) m_stats.had_memory_forced_batch = true;
}

void SemanticBatchController::RecordEof() {
  if (!m_stats.completed) m_stats.saw_eof = true;
}

void SemanticBatchController::Complete(bool had_error) {
  if (m_stats.completed) return;
  m_stats.completed = true;
  m_stats.had_error = had_error;
  m_stats.execution_elapsed_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                    m_execution_started_at)
          .count();

  // Feedback is deliberately conservative: a partial execution would bias
  // cardinality downward, an error can corrupt service timing, and the binary
  // join's behavior is not represented by the unary helper latency curve.
  if (m_feedback_key.empty() || had_error || !m_stats.saw_eof ||
      m_operator_kind == SemanticBatchOperatorKind::kBinaryJoin ||
      m_stats.total_input_rows == 0 ||
      !IsPositiveFinite(m_stats.producer_active_seconds) ||
      m_stats.had_memory_forced_batch || m_stats.successful_batch_count == 0 ||
      m_stats.successful_batch_rows != m_stats.total_input_rows ||
      !IsPositiveFinite(m_stats.successful_batch_service_seconds) ||
      !IsPositiveFinite(m_stats.profiled_batch_service_seconds) ||
      !m_stats.provider_epoch_observed || m_stats.provider_epoch_changed) {
    return;
  }
  if (m_stats.has_result_fingerprint &&
      m_stats.fingerprint_output_rows != m_stats.total_input_rows) {
    // Reject incomplete result fingerprints.
    return;
  }

  SemanticBatchLearnedFeedback sample;
  sample.completed_executions = 1;
  sample.expected_input_rows = static_cast<double>(m_stats.total_input_rows);
  sample.producer_seconds_per_row =
      m_stats.producer_active_seconds /
      static_cast<double>(m_stats.total_input_rows);
  sample.helper_latency_scale =
      std::clamp(m_stats.successful_batch_service_seconds /
                     m_stats.profiled_batch_service_seconds,
                 kMinimumLatencyScale, kMaximumLatencyScale);
  sample.has_result_fingerprint = m_stats.has_result_fingerprint;
  sample.input_fingerprint = m_stats.input_fingerprint;
  sample.output_fingerprint = m_stats.output_fingerprint;
  sample.fingerprint_input_rows = m_stats.total_input_rows;
  sample.fingerprint_output_rows = m_stats.fingerprint_output_rows;
  sample.result_feedback_stable = true;
  sample.provider_epoch = m_stats.provider_epoch;
  sample.observation_phase = m_selection_diagnostics.phase;
  sample.observation_policy = m_policy;
  sample.observation_batch_count = m_stats.successful_batch_count;
  if (IsPositiveFinite(m_stats.execution_elapsed_seconds)) {
    SemanticBatchArmFeedback arm_sample;
    arm_sample.launch_rows = m_policy.launch_rows;
    arm_sample.target_rows = m_policy.target_rows;
    (void)UpdateSemanticBatchArmFeedback(&arm_sample,
                                         m_stats.execution_elapsed_seconds);
    sample.arms.push_back(std::move(arm_sample));
  }
  sample.frozen_shortlist = m_selection_diagnostics.safe_shortlist;
  if (sample.frozen_shortlist.empty()) {
    // Freeze candidates after the first valid execution.
    SemanticBatchSelectionContext freeze_context;
    freeze_context.operator_kind = m_operator_kind;
    freeze_context.predicate_count = m_predicate_count;
    freeze_context.maximum_observed_input_bytes =
        m_maximum_observed_input_bytes;
    freeze_context.hard_safe_cap_rows = m_hard_safe_cap_rows;
    freeze_context.learned = sample;
    SemanticBatchSelectionDiagnostics freeze_diagnostics;
    (void)SelectSemanticBatchPolicy(m_profile, freeze_context,
                                    &freeze_diagnostics);
    sample.frozen_shortlist = freeze_diagnostics.safe_shortlist;
  }
  if (sample.frozen_shortlist.size() > kMaximumShortlistArms) {
    sample.frozen_shortlist.resize(kMaximumShortlistArms);
  }
  PublishFeedback(m_cache_key, sample);
}

}  // namespace vipersql
