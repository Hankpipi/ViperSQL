/* Copyright (c) 2026, Zihao Yu.

   Initialization-time profiling and persistent semantic cost parameters.
*/

#include "sql/semantic_profile.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <sys/stat.h>

#include <nlohmann/json.hpp>
#include <zmq.hpp>

#include "my_io.h"
#include "my_sys.h"
#include "sql/log.h"
#include "sql/iterators/helpers/semantic_helper_endpoint.h"

#ifndef _WIN32
#include <unistd.h>
#endif

namespace vipersql {
namespace {

using Clock = std::chrono::steady_clock;
using json = nlohmann::json;

std::mutex g_profile_mutex;
SemanticProfile g_profile;

bool IsPositiveFinite(double value) {
  return std::isfinite(value) && value > 0.0;
}

uint64_t AvailableMemoryBytes() {
#ifndef _WIN32
  const long pages = sysconf(_SC_AVPHYS_PAGES);
  const long page_size = sysconf(_SC_PAGESIZE);
  if (pages > 0 && page_size > 0) {
    return static_cast<uint64_t>(pages) *
           static_cast<uint64_t>(page_size);
  }
#endif
  return 0;
}

double ProfileNativeRowEvaluationSeconds() {
  // Volatile state prevents the compiler from deleting the calibration loop.
  constexpr uint64_t kIterations = 5000000;
  volatile uint64_t state = 0x9e3779b97f4a7c15ULL;
  const auto start = Clock::now();
  for (uint64_t i = 0; i < kIterations; ++i) {
    state ^= i + (state << 6) + (state >> 2);
  }
  const auto end = Clock::now();
  (void)state;
  const double seconds = std::chrono::duration<double>(end - start).count();
  const double per_row = seconds / static_cast<double>(kIterations);
  return IsPositiveFinite(per_row) ? per_row : 5.0e-8;
}

json OperatorToJson(const SemanticOperatorProfile &profile) {
  json curve = json::array();
  for (const SemanticBatchMeasurement &point : profile.latency_curve) {
    curve.push_back({{"batch_size", point.batch_size},
                     {"latency_seconds", point.latency_seconds},
                     {"sample_count", point.sample_count},
                     {"p90_latency_seconds", point.p90_latency_seconds},
                     {"mad_latency_seconds", point.mad_latency_seconds}});
  }
  return {{"transfer_bandwidth_bytes_per_second",
           profile.transfer_bandwidth_bytes_per_second},
          {"processing_throughput_rows_per_second",
           profile.processing_throughput_rows_per_second},
          {"invocation_overhead_seconds",
           profile.invocation_overhead_seconds},
          {"input_bytes_per_row", profile.input_bytes_per_row},
          {"result_bytes_per_input", profile.result_bytes_per_input},
          {"default_selectivity", profile.default_selectivity},
          {"min_batch_size", profile.min_batch_size},
          {"max_batch_size", profile.max_batch_size},
          {"adaptive_batching", profile.adaptive_batching},
          {"helper_parallelism", profile.helper_parallelism},
          {"operator_memory_bytes", profile.operator_memory_bytes},
          {"helper_memory_bytes", profile.helper_memory_bytes},
          {"measured", profile.measured},
          {"latency_curve", std::move(curve)}};
}

json ProfileToJson(const SemanticProfile &profile) {
  return {{"format_version", profile.format_version},
          {"generated_at_utc", profile.generated_at_utc},
          {"source", profile.source},
          {"native_row_evaluation_seconds",
           profile.native_row_evaluation_seconds},
          {"profiling_epsilon", profile.profiling_epsilon},
          {"max_vectorized_ops", profile.max_vectorized_ops},
          {"unary_filter", OperatorToJson(profile.unary_filter)},
          {"binary_join", OperatorToJson(profile.binary_join)}};
}

bool JsonToOperator(const json &value, SemanticOperatorProfile *profile) {
  try {
    profile->transfer_bandwidth_bytes_per_second =
        value.at("transfer_bandwidth_bytes_per_second").get<double>();
    profile->processing_throughput_rows_per_second =
        value.at("processing_throughput_rows_per_second").get<double>();
    profile->invocation_overhead_seconds =
        value.at("invocation_overhead_seconds").get<double>();
    profile->input_bytes_per_row =
        value.at("input_bytes_per_row").get<double>();
    profile->result_bytes_per_input =
        value.at("result_bytes_per_input").get<double>();
    profile->default_selectivity =
        value.at("default_selectivity").get<double>();
    profile->min_batch_size = value.at("min_batch_size").get<size_t>();
    profile->max_batch_size = value.at("max_batch_size").get<size_t>();
    profile->adaptive_batching = value.value("adaptive_batching", false);
    profile->helper_parallelism =
        value.value("helper_parallelism", size_t{64});
    profile->operator_memory_bytes =
        value.at("operator_memory_bytes").get<uint64_t>();
    profile->helper_memory_bytes =
        value.at("helper_memory_bytes").get<uint64_t>();
    profile->measured = value.at("measured").get<bool>();
    profile->latency_curve.clear();
    for (const json &point : value.at("latency_curve")) {
      profile->latency_curve.push_back(
          {point.at("batch_size").get<size_t>(),
           point.at("latency_seconds").get<double>(),
           point.value("sample_count", size_t{1}),
           point.value("p90_latency_seconds",
                       point.at("latency_seconds").get<double>()),
           point.value("mad_latency_seconds", 0.0)});
    }
  } catch (...) {
    return false;
  }

  if (!IsPositiveFinite(profile->transfer_bandwidth_bytes_per_second) ||
      !IsPositiveFinite(profile->processing_throughput_rows_per_second) ||
      !IsPositiveFinite(profile->invocation_overhead_seconds) ||
      !IsPositiveFinite(profile->input_bytes_per_row) ||
      !std::isfinite(profile->result_bytes_per_input) ||
      profile->result_bytes_per_input < 0.0 ||
      !std::isfinite(profile->default_selectivity) ||
      profile->default_selectivity <= 0.0 ||
      profile->default_selectivity > 1.0 || profile->min_batch_size == 0 ||
      profile->max_batch_size == 0 ||
      profile->min_batch_size > profile->max_batch_size ||
      profile->helper_parallelism == 0 || profile->helper_parallelism > 4096) {
    return false;
  }
  size_t previous_batch_size = 0;
  for (const SemanticBatchMeasurement &point : profile->latency_curve) {
    if (point.batch_size == 0 || !IsPositiveFinite(point.latency_seconds)) {
      return false;
    }
    if (point.sample_count == 0 ||
        !IsPositiveFinite(point.p90_latency_seconds) ||
        !std::isfinite(point.mad_latency_seconds) ||
        point.mad_latency_seconds < 0.0) {
      return false;
    }
    if (point.batch_size <= previous_batch_size) {
      return false;
    }
    previous_batch_size = point.batch_size;
  }
  if (!profile->latency_curve.empty() &&
      profile->max_batch_size > profile->latency_curve.back().batch_size) {
    return false;
  }
  return true;
}

bool LoadProfile(const std::string &path, SemanticProfile *profile) {
  std::ifstream input(path);
  if (!input.good()) return false;

  try {
    json document;
    input >> document;
    profile->format_version = document.at("format_version").get<int>();
    profile->generated_at_utc =
        document.at("generated_at_utc").get<std::string>();
    profile->source = document.at("source").get<std::string>();
    profile->native_row_evaluation_seconds =
        document.at("native_row_evaluation_seconds").get<double>();
    profile->profiling_epsilon =
        document.at("profiling_epsilon").get<double>();
    profile->max_vectorized_ops =
        document.value("max_vectorized_ops", uint32_t{0});
    if (profile->format_version != SemanticProfile::kFormatVersion ||
        profile->generated_at_utc.empty() || profile->source.empty() ||
        !IsPositiveFinite(profile->native_row_evaluation_seconds) ||
        !std::isfinite(profile->profiling_epsilon) ||
        profile->profiling_epsilon < 0.0 ||
        profile->profiling_epsilon > 1.0 ||
        profile->max_vectorized_ops > 64 ||
        !JsonToOperator(document.at("unary_filter"),
                        &profile->unary_filter) ||
        !JsonToOperator(document.at("binary_join"),
                        &profile->binary_join)) {
      return false;
    }
  } catch (...) {
    return false;
  }
  return true;
}

bool PersistProfile(const std::string &path,
                    const SemanticProfile &profile) {
  const std::string temporary_path = path + ".tmp";
  const std::string contents = ProfileToJson(profile).dump(2) + "\n";
#ifdef _WIN32
  const int create_mode = 0;  // my_create ignores this argument on Windows.
#else
  const int create_mode = S_IRUSR | S_IWUSR;
#endif
  const File descriptor =
      my_create(temporary_path.c_str(), create_mode,
                O_WRONLY | O_TRUNC, MYF(0));
  if (descriptor < 0) return false;

  const bool write_failed =
      my_write(descriptor,
               reinterpret_cast<const uchar *>(contents.data()),
               contents.size(), MYF(MY_NABP)) == MY_FILE_ERROR;
  const bool sync_failed = !write_failed && my_sync(descriptor, MYF(0)) != 0;
  const bool close_failed = my_close(descriptor, MYF(0)) != 0;
  if (write_failed || sync_failed || close_failed) {
    my_delete(temporary_path.c_str(), MYF(0));
    return false;
  }

  if (my_rename(temporary_path.c_str(), path.c_str(), MYF(0)) != 0) {
    my_delete(temporary_path.c_str(), MYF(0));
    return false;
  }
  return true;
}

constexpr char kProfilePredicate[] = "The statement is true.";

/**
  Profiling client with a separate REQ socket. A timed-out ZeroMQ REQ socket
  cannot send another request until its outstanding receive is resolved.
*/
class ProfileRpcClient {
 public:
  ProfileRpcClient(int receive_timeout_ms, int send_timeout_ms)
      : m_context(1),
        m_receive_timeout_ms(receive_timeout_ms),
        m_send_timeout_ms(send_timeout_ms) {}

  bool Call(const std::string &name,
            const std::vector<std::string> &values, json *response,
            double *latency_seconds) {
    try {
      EnsureSocket();
      json request_json{{"name", name},
                        {"values", values},
                        {"predicate", kProfilePredicate}};
      const std::string request_text = request_json.dump();
      zmq::message_t request(request_text.size());
      std::memcpy(request.data(), request_text.data(), request_text.size());

      const auto start = Clock::now();
      if (!m_socket->send(request, zmq::send_flags::none)) {
        ResetSocket();
        return false;
      }
      zmq::message_t reply;
      if (!m_socket->recv(reply, zmq::recv_flags::none)) {
        ResetSocket();
        return false;
      }
      const auto end = Clock::now();

      const std::string reply_text(static_cast<const char *>(reply.data()),
                                   reply.size());
      *response = json::parse(reply_text);
      *latency_seconds =
          std::chrono::duration<double>(end - start).count();
      return IsPositiveFinite(*latency_seconds);
    } catch (...) {
      ResetSocket();
      return false;
    }
  }

 private:
  void EnsureSocket() {
    if (m_socket != nullptr) return;
    m_socket.reset(new zmq::socket_t(m_context, zmq::socket_type::req));
    m_socket->setsockopt(ZMQ_RCVTIMEO, m_receive_timeout_ms);
    m_socket->setsockopt(ZMQ_SNDTIMEO, m_send_timeout_ms);
    m_socket->setsockopt(ZMQ_LINGER, 0);
    m_socket->connect(semhelpers::SemanticHelperEndpoint());
  }

  void ResetSocket() { m_socket.reset(); }

  zmq::context_t m_context;
  std::unique_ptr<zmq::socket_t> m_socket;
  int m_receive_timeout_ms;
  int m_send_timeout_ms;
};

double MeasureUnknownTaskLatency(size_t payload_bytes) {
  constexpr size_t kRepetitions = 3;
  std::vector<double> latencies;
  latencies.reserve(kRepetitions);
  const std::vector<std::string> values(1,
                                        std::string(payload_bytes, 'x'));

  for (size_t repetition = 0; repetition < kRepetitions; ++repetition) {
    // Use a fresh socket for every sample. This keeps connection setup equal
    // between payload sizes and prevents a timed-out REQ socket from affecting
    // either the next sample or the real profiling client.
    ProfileRpcClient client(3000, 1000);
    json response;
    double latency_seconds = 0.0;
    if (!client.Call("__vipersql_profile_transport__", values, &response,
                     &latency_seconds)) {
      return 0.0;
    }
    latencies.push_back(latency_seconds);
  }

  std::sort(latencies.begin(), latencies.end());
  return latencies[latencies.size() / 2];
}

double MeasureTransportBandwidth() {
  constexpr double kFallbackBytesPerSecond = 500.0 * 1024.0 * 1024.0;
  constexpr size_t kSmallPayloadBytes = 4 * 1024;
  constexpr size_t kLargePayloadBytes = 256 * 1024;
  const double small_seconds =
      MeasureUnknownTaskLatency(kSmallPayloadBytes);
  const double large_seconds =
      MeasureUnknownTaskLatency(kLargePayloadBytes);
  if (!IsPositiveFinite(small_seconds) ||
      !IsPositiveFinite(large_seconds) || large_seconds <= small_seconds) {
    return kFallbackBytesPerSecond;
  }

  const double bandwidth =
      static_cast<double>(kLargePayloadBytes - kSmallPayloadBytes) /
      (large_seconds - small_seconds);
  if (!IsPositiveFinite(bandwidth)) return kFallbackBytesPerSecond;

  // Bound noisy loopback measurements while retaining a conservative floor.
  constexpr double kMinimumBytesPerSecond = 1024.0 * 1024.0;
  constexpr double kMaximumBytesPerSecond =
      100.0 * 1024.0 * 1024.0 * 1024.0;
  return std::clamp(bandwidth, kMinimumBytesPerSecond,
                    kMaximumBytesPerSecond);
}

void AccumulateSelectivity(const json &response, size_t *positive_results,
                           size_t *total_results) {
  if (!response.is_object() || !response.contains("values") ||
      !response["values"].is_array()) {
    return;
  }

  for (const json &value : response["values"]) {
    bool positive = false;
    if (value.is_boolean()) {
      positive = value.get<bool>();
    } else if (value.is_number_unsigned()) {
      positive = value.get<uint64_t>() != 0;
    } else if (value.is_number_integer()) {
      positive = value.get<int64_t>() != 0;
    } else if (value.is_number_float()) {
      positive = value.get<double>() != 0.0;
    } else {
      continue;
    }
    ++(*total_results);
    if (positive) ++(*positive_results);
  }
}

std::string UtcTimestamp() {
  const std::time_t now = std::time(nullptr);
  std::tm broken_down{};
#ifdef _WIN32
  gmtime_s(&broken_down, &now);
#else
  gmtime_r(&now, &broken_down);
#endif
  std::ostringstream output;
  output << std::put_time(&broken_down, "%Y-%m-%dT%H:%M:%SZ");
  return output.str();
}

SemanticProfile RunProfiler() {
  SemanticProfile profile;
  profile.generated_at_utc = UtcTimestamp();
  profile.native_row_evaluation_seconds =
      ProfileNativeRowEvaluationSeconds();
  profile.unary_filter.helper_memory_bytes = AvailableMemoryBytes();
  profile.binary_join.helper_memory_bytes =
      profile.unary_filter.helper_memory_bytes;

  // Fail fast when --initialize runs without the helper. The ping uses a
  // separate socket so a timeout cannot poison the long-running profile
  // socket, and an unknown-operation response still proves reachability.
  ProfileRpcClient availability_client(2000, 1000);
  json availability_response;
  double availability_seconds = 0.0;
  if (!availability_client.Call("__vipersql_profile_ping__", {},
                                &availability_response,
                                &availability_seconds)) {
    profile.source = "built_in_defaults_helper_unavailable";
    return profile;
  }

  profile.unary_filter.transfer_bandwidth_bytes_per_second =
      MeasureTransportBandwidth();

  // The helper itself has a 120-second operator timeout. Leave enough time
  // for it to serialize and return that timeout response.
  ProfileRpcClient profile_client(130000, 5000);
  const std::vector<std::string> warmup_values{"The sky is blue."};
  json warmup;
  double warmup_seconds = 0.0;
  if (!profile_client.Call("sem_lite_llm_filter", warmup_values,
                           &warmup, &warmup_seconds) ||
      !warmup.is_object() || !warmup.value("ok", false) ||
      !warmup.contains("values") || !warmup["values"].is_array() ||
      warmup["values"].size() != warmup_values.size()) {
    profile.source = "built_in_defaults_helper_warmup_failed";
    return profile;
  }

  // Keep the robust small-batch sweep and add progressively cheaper samples
  // at helper-concurrency wave boundaries. Large points define a safe
  // fill-before-pop cap; they are not repeated five times because each point
  // issues one model request per input row.
  const std::vector<size_t> batch_sizes = {
      1, 2, 4, 8, 16, 25, 32, 50, 64, 128, 256, 512, 1000};
  const std::vector<size_t> sample_targets = {
      5, 5, 5, 5, 5, 5, 5, 5, 3, 3, 1, 1, 1};
  constexpr size_t kRepetitions = 5;
  constexpr size_t kMandatoryMaximum = 50;
  constexpr double kMaximumSafeP90Seconds = 90.0;
  size_t positive_results = 0;
  size_t total_results = 0;
  size_t total_inputs = warmup_values.size();
  uint64_t total_input_bytes = warmup_values.front().size();
  double result_payload_bytes =
      static_cast<double>(warmup["values"].dump().size());
  AccumulateSelectivity(warmup, &positive_results, &total_results);
  // Alternate sweep direction so time-dependent latency changes are
  // distributed across batch sizes.
  std::vector<std::vector<double>> latency_samples(batch_sizes.size());
  for (std::vector<double> &samples : latency_samples) {
    samples.reserve(kRepetitions);
  }
  bool optional_profiling_stopped = false;
  for (size_t repetition = 0; repetition < kRepetitions; ++repetition) {
    for (size_t offset = 0; offset < batch_sizes.size(); ++offset) {
      const size_t batch_index = repetition % 2 == 0
                                     ? offset
                                     : batch_sizes.size() - 1 - offset;
      const size_t batch_size = batch_sizes[batch_index];
      if (repetition >= sample_targets[batch_index]) continue;
      if (optional_profiling_stopped && batch_size > kMandatoryMaximum) {
        continue;
      }
      std::vector<std::string> prompts;
      prompts.reserve(batch_size);
      for (size_t i = 0; i < batch_size; ++i) {
        prompts.push_back(
            i % 2 == 0 ? "Water freezes below zero Celsius."
                       : "Water freezes at one hundred Celsius.");
      }
      json response;
      double seconds = 0.0;
      const bool valid_response =
          profile_client.Call("sem_lite_llm_filter", prompts, &response,
                              &seconds) &&
          response.is_object() && response.value("ok", false) &&
          response.contains("values") && response["values"].is_array() &&
          response["values"].size() == batch_size;
      if (!valid_response ||
          (batch_size > kMandatoryMaximum &&
           seconds >= kMaximumSafeP90Seconds)) {
        if (batch_size > kMandatoryMaximum) {
          optional_profiling_stopped = true;
          continue;
        }
        profile.source = "built_in_defaults_profile_incomplete";
        return profile;
      }
      latency_samples[batch_index].push_back(seconds);
      result_payload_bytes +=
          static_cast<double>(response["values"].dump().size());
      AccumulateSelectivity(response, &positive_results, &total_results);
      total_inputs += prompts.size();
      for (const std::string &prompt : prompts) {
        total_input_bytes += prompt.size();
      }
    }
  }

  for (size_t batch_index = 0; batch_index < batch_sizes.size();
       ++batch_index) {
    const size_t batch_size = batch_sizes[batch_index];
    std::vector<double> &latencies = latency_samples[batch_index];
    if (latencies.empty()) continue;
    std::sort(latencies.begin(), latencies.end());
    const double median = latencies[latencies.size() / 2];
    std::vector<double> absolute_deviations;
    absolute_deviations.reserve(latencies.size());
    for (const double latency : latencies) {
      absolute_deviations.push_back(std::abs(latency - median));
    }
    std::sort(absolute_deviations.begin(), absolute_deviations.end());
    const double p90_rank =
        0.90 * static_cast<double>(latencies.size() - 1);
    const size_t p90_left = static_cast<size_t>(std::floor(p90_rank));
    const size_t p90_right = static_cast<size_t>(std::ceil(p90_rank));
    const double p90_fraction = p90_rank - static_cast<double>(p90_left);
    const double p90 = latencies[p90_left] +
                       p90_fraction *
                           (latencies[p90_right] - latencies[p90_left]);
    profile.unary_filter.latency_curve.push_back(
        {batch_size, median, latencies.size(), p90,
         absolute_deviations[absolute_deviations.size() / 2]});
  }

  if (total_inputs > 0) {
    profile.unary_filter.input_bytes_per_row =
        static_cast<double>(total_input_bytes) /
        static_cast<double>(total_inputs);
  }
  profile.unary_filter.result_bytes_per_input =
      total_results == 0 ? 1.0
                         : result_payload_bytes /
                               static_cast<double>(total_results);
  if (total_results > 0) {
    profile.unary_filter.default_selectivity = std::clamp(
        static_cast<double>(positive_results) /
            static_cast<double>(total_results),
        0.05, 0.95);
  }

  const auto &curve = profile.unary_filter.latency_curve;
  const double first_latency = curve.front().latency_seconds;
  const double last_latency = curve.back().latency_seconds;
  const double observed_seconds_per_row =
      (last_latency - first_latency) /
      static_cast<double>(curve.back().batch_size -
                          curve.front().batch_size);
  const double transfer_seconds_per_row =
      (profile.unary_filter.input_bytes_per_row +
       profile.unary_filter.result_bytes_per_input) /
      profile.unary_filter.transfer_bandwidth_bytes_per_second;
  const double fitted_processing_seconds =
      observed_seconds_per_row - transfer_seconds_per_row;
  double processing_seconds_per_row = 1.0e-9;
  if (IsPositiveFinite(fitted_processing_seconds)) {
    processing_seconds_per_row = fitted_processing_seconds;
  }
  // A flat curve means batching hid per-row processing; model it as a high
  // bounded throughput rather than falling back to the slow default 10 rows/s.
  profile.unary_filter.processing_throughput_rows_per_second =
      1.0 / processing_seconds_per_row;

  const double first_batch_variable_seconds =
      static_cast<double>(curve.front().batch_size) *
      (processing_seconds_per_row + transfer_seconds_per_row);
  profile.unary_filter.invocation_overhead_seconds =
      std::max(1.0e-6, first_latency - first_batch_variable_seconds);

  // Use the smallest batch whose P90 latency per row is within the profiling
  // tolerance of the optimum.
  const auto best_batch = std::min_element(
      curve.begin(), curve.end(),
      [](const SemanticBatchMeasurement &left,
         const SemanticBatchMeasurement &right) {
        const double left_per_row = left.p90_latency_seconds /
                                    static_cast<double>(left.batch_size);
        const double right_per_row = right.p90_latency_seconds /
                                     static_cast<double>(right.batch_size);
        if (left_per_row != right_per_row)
          return left_per_row < right_per_row;
        const double left_median_per_row =
            left.latency_seconds / static_cast<double>(left.batch_size);
        const double right_median_per_row =
            right.latency_seconds / static_cast<double>(right.batch_size);
        if (left_median_per_row != right_median_per_row)
          return left_median_per_row < right_median_per_row;
        return left.batch_size < right.batch_size;
      });
  const double best_p90_per_row =
      best_batch->p90_latency_seconds /
      static_cast<double>(best_batch->batch_size);
  const auto smallest_near_best_batch = std::find_if(
      curve.begin(), curve.end(),
      [&](const SemanticBatchMeasurement &point) {
        return point.p90_latency_seconds /
                   static_cast<double>(point.batch_size) <=
               (1.0 + profile.profiling_epsilon) * best_p90_per_row;
      });
  profile.unary_filter.max_batch_size =
      smallest_near_best_batch->batch_size;

  // Choose the initial submission threshold independently from the maximum
  // batch size. Restrict it to the repeatedly sampled mandatory range so short
  // inputs do not wait for EOF.
  const auto launch_curve_end = std::find_if(
      curve.begin(), curve.end(), [](const SemanticBatchMeasurement &point) {
        return point.batch_size > kMandatoryMaximum;
      });
  const auto best_launch_batch = std::min_element(
      curve.begin(), launch_curve_end,
      [](const SemanticBatchMeasurement &left,
         const SemanticBatchMeasurement &right) {
        const double left_per_row = left.p90_latency_seconds /
                                    static_cast<double>(left.batch_size);
        const double right_per_row = right.p90_latency_seconds /
                                     static_cast<double>(right.batch_size);
        if (left_per_row != right_per_row)
          return left_per_row < right_per_row;
        return left.batch_size < right.batch_size;
      });
  const double best_launch_per_row =
      best_launch_batch->latency_seconds /
      static_cast<double>(best_launch_batch->batch_size);
  profile.unary_filter.min_batch_size = best_launch_batch->batch_size;
  const size_t minimum_occupancy =
      (best_launch_batch->batch_size + 1) / 2;
  for (const SemanticBatchMeasurement &point : curve) {
    if (point.batch_size > best_launch_batch->batch_size) break;
    if (point.batch_size < minimum_occupancy) continue;
    if (point.latency_seconds / static_cast<double>(point.batch_size) <=
        (1.0 + profile.profiling_epsilon) * best_launch_per_row) {
      profile.unary_filter.min_batch_size = point.batch_size;
      break;
    }
  }
  profile.unary_filter.adaptive_batching = true;
  profile.unary_filter.operator_memory_bytes = static_cast<uint64_t>(
      profile.unary_filter.max_batch_size *
      (profile.unary_filter.input_bytes_per_row +
       profile.unary_filter.result_bytes_per_input));
  profile.unary_filter.measured = true;

  // Binary semantic joins share the RPC transport. Until a representative
  // build-side corpus is available during initialization, derive conservative
  // processing parameters rather than pretending an empty datadir can profile
  // query-specific join selectivity or index memory.
  profile.binary_join = profile.unary_filter;
  profile.binary_join.input_bytes_per_row =
      2.0 * profile.unary_filter.input_bytes_per_row;
  profile.binary_join.result_bytes_per_input = sizeof(uint32_t);
  profile.binary_join.default_selectivity = 0.1;
  profile.binary_join.operator_memory_bytes =
      static_cast<uint64_t>(profile.binary_join.max_batch_size *
                            (profile.binary_join.input_bytes_per_row +
                             sizeof(uint32_t)));
  profile.binary_join.measured = false;
  profile.source = "initialization_adaptive_profile";
  return profile;
}

double InterpolateBatchLatency(const SemanticOperatorProfile &profile,
                               double batch_rows) {
  if (!(batch_rows > 0.0) || profile.latency_curve.empty()) return 0.0;
  if (batch_rows <=
      static_cast<double>(profile.latency_curve.front().batch_size)) {
    return profile.latency_curve.front().latency_seconds;
  }

  for (size_t i = 1; i < profile.latency_curve.size(); ++i) {
    const double right_rows =
        static_cast<double>(profile.latency_curve[i].batch_size);
    if (batch_rows > right_rows) continue;
    const double left_rows =
        static_cast<double>(profile.latency_curve[i - 1].batch_size);
    const double fraction =
        (batch_rows - left_rows) / (right_rows - left_rows);
    return profile.latency_curve[i - 1].latency_seconds +
           fraction * (profile.latency_curve[i].latency_seconds -
                       profile.latency_curve[i - 1].latency_seconds);
  }

  if (profile.latency_curve.size() == 1)
    return profile.latency_curve.front().latency_seconds;
  const size_t last = profile.latency_curve.size() - 1;
  const double left_rows =
      static_cast<double>(profile.latency_curve[last - 1].batch_size);
  const double right_rows =
      static_cast<double>(profile.latency_curve[last].batch_size);
  const double slope = std::max(
      0.0, (profile.latency_curve[last].latency_seconds -
            profile.latency_curve[last - 1].latency_seconds) /
               (right_rows - left_rows));
  return profile.latency_curve[last].latency_seconds +
         (batch_rows - right_rows) * slope;
}

uint64_t ExpectedAdaptiveBatchCount(double integral_rows,
                                    const SemanticOperatorProfile &profile) {
  if (!(integral_rows > 0.0)) return 0;
  const double min_batch = static_cast<double>(std::clamp(
      profile.min_batch_size, size_t{1},
      std::max<size_t>(1, profile.max_batch_size)));
  const double max_batch =
      static_cast<double>(std::max<size_t>(1, profile.max_batch_size));
  if (integral_rows <= min_batch) return 1;
  const double batches = 1.0 + std::ceil((integral_rows - min_batch) /
                                         max_batch);
  if (!std::isfinite(batches) ||
      batches >= static_cast<double>(std::numeric_limits<uint64_t>::max())) {
    return std::numeric_limits<uint64_t>::max();
  }
  return static_cast<uint64_t>(batches);
}

double EmpiricalBatchedSeconds(double rows, double input_row_bytes,
                               const SemanticOperatorProfile &profile) {
  if (!(rows > 0.0)) return 0.0;
  if (!std::isfinite(rows)) return std::numeric_limits<double>::max();

  const double integral_rows = std::ceil(rows);
  const size_t min_batch_size = std::clamp(
      profile.min_batch_size, size_t{1},
      std::max<size_t>(1, profile.max_batch_size));
  const size_t max_batch_size =
      std::max<size_t>(1, profile.max_batch_size);
  if (profile.latency_curve.empty()) {
    const double batches = static_cast<double>(
        ExpectedAdaptiveBatchCount(integral_rows, profile));
    const double seconds =
        integral_rows * (input_row_bytes + profile.result_bytes_per_input) /
            profile.transfer_bandwidth_bytes_per_second +
        integral_rows / profile.processing_throughput_rows_per_second +
        batches * profile.invocation_overhead_seconds;
    return std::isfinite(seconds) ? seconds
                                  : std::numeric_limits<double>::max();
  }

  // Model the first submission with min_batch_size and later submissions with
  // max_batch_size. Fill-before-pop continues consuming rows while the helper
  // is active.
  const double first_rows =
      std::min(integral_rows, static_cast<double>(min_batch_size));
  double seconds = InterpolateBatchLatency(profile, first_rows);
  const double remaining_rows = integral_rows - first_rows;
  const double full_batches =
      std::floor(remaining_rows / static_cast<double>(max_batch_size));
  const double tail_rows = remaining_rows -
                           full_batches *
                               static_cast<double>(max_batch_size);
  seconds += full_batches *
             InterpolateBatchLatency(profile, max_batch_size);
  if (tail_rows > 0.0) {
    seconds += InterpolateBatchLatency(profile, tail_rows);
  }

  // The empirical curve already includes transfer of the profiled payload.
  // Charge only bytes beyond it so transfer is not counted twice.
  const double excess_input_bytes =
      std::max(0.0, input_row_bytes - profile.input_bytes_per_row);
  seconds += integral_rows * excess_input_bytes /
             profile.transfer_bandwidth_bytes_per_second;
  return std::isfinite(seconds) ? seconds
                                : std::numeric_limits<double>::max();
}

}  // namespace

size_t SelectSemanticLaunchWatermark(
    const SemanticOperatorProfile &profile,
    SemanticBatchOperatorKind operator_kind, size_t predicate_count,
    size_t maximum_observed_input_bytes) {
  const size_t maximum = std::max<size_t>(1, profile.max_batch_size);
  const size_t baseline =
      std::clamp(profile.min_batch_size, size_t{1}, maximum);
  if (!profile.adaptive_batching) return baseline;

  // A two-column filter is normally fed by a selective relational valley.
  // Coalescing until EOF or the safe maximum avoids an extra serialized RPC.
  if (operator_kind == SemanticBatchOperatorKind::kTwoColumnFilter) {
    return maximum;
  }

  // Compound predicates and long text keep each helper worker busy longer.
  // Two worker waves amortize the RPC without forcing all such streams to
  // wait until EOF. Short unary filters retain the early baseline launch.
  const double profiled_bytes =
      std::max(1.0, profile.input_bytes_per_row);
  const size_t long_input_threshold = static_cast<size_t>(
      std::ceil(std::max(512.0, 8.0 * profiled_bytes)));
  const bool expensive_input =
      predicate_count > 1 ||
      maximum_observed_input_bytes >= long_input_threshold;
  if (!expensive_input ||
      operator_kind == SemanticBatchOperatorKind::kBinaryJoin) {
    return baseline;
  }

  const size_t parallelism =
      std::clamp(profile.helper_parallelism, size_t{1}, maximum);
  const size_t two_waves =
      parallelism > maximum / 2 ? maximum : parallelism * 2;
  return std::clamp(std::max(baseline, two_waves), size_t{1}, maximum);
}

const SemanticProfile &GetSemanticProfile() { return g_profile; }

std::string SemanticProfilePath(const char *data_directory) {
  std::string path = data_directory == nullptr ? "" : data_directory;
  if (!path.empty() && path.back() != '/' && path.back() != '\\') {
    path.push_back('/');
  }
  path += "vipersql-semantic-profile.json";
  return path;
}

bool LoadSemanticProfile(const char *data_directory) {
  std::lock_guard<std::mutex> guard(g_profile_mutex);
  const std::string path = SemanticProfilePath(data_directory);
  SemanticProfile loaded;
  if (LoadProfile(path, &loaded)) {
    g_profile = std::move(loaded);
    sql_print_information("ViperSQL loaded semantic cost profile from %s",
                          path.c_str());
    return true;
  }

  g_profile = SemanticProfile{};
  sql_print_warning(
      "ViperSQL semantic cost profile %s is missing or invalid; using "
      "built-in optimizer defaults",
      path.c_str());
  return false;
}

bool GenerateSemanticProfile(const char *data_directory) {
  try {
    std::lock_guard<std::mutex> guard(g_profile_mutex);
    const std::string path = SemanticProfilePath(data_directory);
    SemanticProfile measured;
    try {
      measured = RunProfiler();
    } catch (...) {
      // No helper/protocol/serialization failure may escape process_bootstrap.
      measured = SemanticProfile{};
      measured.generated_at_utc = "1970-01-01T00:00:00Z";
      measured.source = "built_in_defaults_profile_exception";
    }

    bool persisted = false;
    try {
      persisted = PersistProfile(path, measured);
    } catch (...) {
      persisted = false;
    }

    const bool helper_was_measured = measured.unary_filter.measured;
    g_profile = std::move(measured);
    if (!persisted) {
      sql_print_warning(
          "ViperSQL could not persist semantic cost profile at %s; using the "
          "in-memory profile",
          path.c_str());
      return false;
    }
    if (helper_was_measured) {
      sql_print_information("ViperSQL generated semantic cost profile at %s",
                            path.c_str());
    } else {
      sql_print_warning(
          "ViperSQL helper profiling was unavailable during initialization; "
          "persisted conservative defaults at %s",
          path.c_str());
    }
    return helper_was_measured;
  } catch (...) {
    // This outer guard includes path construction, locking, assignment and
    // logging preparation. Preserve --initialize success even in those cases.
    try {
      std::lock_guard<std::mutex> guard(g_profile_mutex);
      g_profile = SemanticProfile{};
    } catch (...) {
    }
    sql_print_warning(
        "ViperSQL semantic profiling failed unexpectedly; using built-in "
        "optimizer defaults");
    return false;
  }
}

double EstimateUnarySemanticSeconds(double rows, double input_row_bytes) {
  if (rows <= 0.0) return 0.0;
  if (!std::isfinite(rows)) return std::numeric_limits<double>::max();
  const SemanticOperatorProfile &parameters = g_profile.unary_filter;
  if (!std::isfinite(input_row_bytes) || input_row_bytes < 0.0) {
    input_row_bytes = parameters.input_bytes_per_row;
  }
  return EmpiricalBatchedSeconds(rows, input_row_bytes, parameters);
}

uint64_t EstimateUnarySemanticBatches(double rows) {
  if (!(rows > 0.0)) return 0;
  if (!std::isfinite(rows)) return std::numeric_limits<uint64_t>::max();
  return ExpectedAdaptiveBatchCount(std::ceil(rows), g_profile.unary_filter);
}

double EstimateBinarySemanticSeconds(double build_rows, double build_row_bytes,
                                     double probe_rows,
                                     double probe_row_bytes) {
  if (!std::isfinite(build_rows) || !std::isfinite(probe_rows) ||
      build_rows < 0.0 || probe_rows < 0.0) {
    return std::numeric_limits<double>::max();
  }
  const SemanticOperatorProfile &parameters = g_profile.binary_join;
  if (!std::isfinite(build_row_bytes) || build_row_bytes < 0.0) {
    build_row_bytes = parameters.input_bytes_per_row / 2.0;
  }
  if (!std::isfinite(probe_row_bytes) || probe_row_bytes < 0.0) {
    probe_row_bytes = parameters.input_bytes_per_row / 2.0;
  }
  if (probe_rows < build_rows) {
    std::swap(build_rows, probe_rows);
    std::swap(build_row_bytes, probe_row_bytes);
  }
  if (!std::isfinite(parameters.default_selectivity) ||
      parameters.default_selectivity <= 0.0 ||
      parameters.default_selectivity > 1.0) {
    return std::numeric_limits<double>::max();
  }

  const uint64_t build_batches =
      ExpectedAdaptiveBatchCount(std::ceil(build_rows), parameters);
  const uint64_t probe_batches =
      ExpectedAdaptiveBatchCount(std::ceil(probe_rows), parameters);
  // Runtime sends every build batch, one BUILD_DONE request, then every probe
  // batch.
  const long double invocation_count =
      static_cast<long double>(build_batches) + 1.0L + probe_batches;
  const long double expected_result_rows =
      static_cast<long double>(build_rows) * probe_rows *
      parameters.default_selectivity;
  const long double transfer_bytes =
      static_cast<long double>(build_rows) * build_row_bytes +
      static_cast<long double>(probe_rows) * probe_row_bytes +
      expected_result_rows * parameters.result_bytes_per_input;
  const long double processed_rows =
      static_cast<long double>(build_rows) + probe_rows;
  const long double seconds =
      transfer_bytes / parameters.transfer_bandwidth_bytes_per_second +
      processed_rows / parameters.processing_throughput_rows_per_second +
      invocation_count * parameters.invocation_overhead_seconds;
  if (!std::isfinite(seconds) || seconds > std::numeric_limits<double>::max()) {
    return std::numeric_limits<double>::max();
  }
  return static_cast<double>(seconds);
}

}  // namespace vipersql
