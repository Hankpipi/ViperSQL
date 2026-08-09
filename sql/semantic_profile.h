/* Copyright (c) 2026, ViperSQL contributors.

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License, version 2.0.
*/

#ifndef SQL_SEMANTIC_PROFILE_H_
#define SQL_SEMANTIC_PROFILE_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace vipersql {

/** One point on the initialization-time helper service curve. */
struct SemanticBatchMeasurement {
  size_t batch_size{0};
  double latency_seconds{0.0};
  size_t sample_count{1};
  double p90_latency_seconds{0.0};
  double mad_latency_seconds{0.0};
};

/** Persistent parameters used by the semantic cost model. */
struct SemanticOperatorProfile {
  double transfer_bandwidth_bytes_per_second{500.0 * 1024.0 * 1024.0};
  double processing_throughput_rows_per_second{10.0};
  double invocation_overhead_seconds{0.02};
  double input_bytes_per_row{64.0};
  double result_bytes_per_input{1.0};
  double default_selectivity{0.5};
  size_t min_batch_size{25};
  size_t max_batch_size{50};
  /**
    Enable runtime selection of the launch watermark from observable operator
    and input properties. The maximum remains a profiled safety bound.
  */
  bool adaptive_batching{false};
  /** Number of row requests the helper can execute concurrently. */
  size_t helper_parallelism{64};
  uint64_t operator_memory_bytes{0};
  uint64_t helper_memory_bytes{0};
  bool measured{false};
  std::vector<SemanticBatchMeasurement> latency_curve;
};

/** Runtime semantic-filter shape used by the batch policy. */
enum class SemanticBatchOperatorKind {
  kUnaryFilter,
  kTwoColumnFilter,
  kBinaryJoin,
  kOther
};

/**
  Select an initial/idle-helper launch watermark without using query identity
  or optimizer cardinality estimates.

  The configured minimum is retained for inexpensive unary inputs. Expensive
  compound or long inputs launch at enough rows to feed two helper waves, and
  a two-column filter coalesces up to the maximum/EOF boundary. ViperFlow still
  determines every actual batch from helper readiness and buffer occupancy.
*/
size_t SelectSemanticLaunchWatermark(const SemanticOperatorProfile &profile,
                                     SemanticBatchOperatorKind operator_kind,
                                     size_t predicate_count,
                                     size_t maximum_observed_input_bytes);

/** Versioned, immutable-after-startup optimizer profile. */
struct SemanticProfile {
  static constexpr int kFormatVersion = 1;

  int format_version{kFormatVersion};
  std::string generated_at_utc;
  std::string source{"built_in_defaults"};
  double native_row_evaluation_seconds{5.0e-8};
  double profiling_epsilon{0.10};
  /** Maximum number of vectorized operators; zero disables them. */
  uint32_t max_vectorized_ops{0};
  SemanticOperatorProfile unary_filter;
  SemanticOperatorProfile binary_join;
};

/** Return the process-wide profile, or conservative defaults before startup. */
const SemanticProfile &GetSemanticProfile();

/** Return <datadir>/vipersql-semantic-profile.json. */
std::string SemanticProfilePath(const char *data_directory);

/**
  Load a profile generated during --initialize.

  This function never contacts the external helper. If the file is missing or
  invalid, conservative built-in defaults remain active.
*/
bool LoadSemanticProfile(const char *data_directory);

/**
  Profile the external helper and atomically persist the result.

  Profiling failure is non-fatal. A profile containing conservative defaults
  is persisted so a missing helper cannot make --initialize unrecoverable.
*/
bool GenerateSemanticProfile(const char *data_directory);

/** Estimate unary semantic-filter service time in seconds. */
double EstimateUnarySemanticSeconds(double rows, double input_row_bytes);

/** Estimated number of adaptive helper dispatches for a unary operator. */
uint64_t EstimateUnarySemanticBatches(double rows);

/** Estimate binary semantic-join service time in seconds. */
double EstimateBinarySemanticSeconds(double build_rows, double build_row_bytes,
                                     double probe_rows, double probe_row_bytes);

}  // namespace vipersql

#endif  // SQL_SEMANTIC_PROFILE_H_
