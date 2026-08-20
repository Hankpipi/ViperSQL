/* Copyright (c) 2026, Zihao Yu.

   Narrow process-lifetime endpoint selector for the e701 timing binary.
*/

#ifndef SQL_ITERATORS_HELPERS_SEMANTIC_HELPER_ENDPOINT_H_
#define SQL_ITERATORS_HELPERS_SEMANTIC_HELPER_ENDPOINT_H_

#include <cstdlib>
#include <stdexcept>
#include <string>
#include <string_view>

namespace semhelpers {

inline constexpr char kSemanticHelperEndpointEnvironment[] =
    "VIPERSQL_SEMANTIC_HELPER_ENDPOINT";
inline constexpr char kDefaultSemanticHelperEndpoint[] =
    "tcp://127.0.0.1:5555";
inline constexpr char kIsolatedSemanticHelperEndpoint[] =
    "tcp://127.0.0.1:5556";

struct SemanticHelperEndpointConfig {
  bool valid{false};
  bool overridden{false};
  std::string endpoint;
  std::string error;
};

/** Resolve only the production or isolated loopback helper endpoint. */
inline SemanticHelperEndpointConfig ResolveSemanticHelperEndpoint(
    const char *configured_endpoint) {
  SemanticHelperEndpointConfig config;
  if (configured_endpoint == nullptr) {
    config.valid = true;
    config.endpoint = kDefaultSemanticHelperEndpoint;
    return config;
  }

  config.overridden = true;
  const std::string_view candidate(configured_endpoint);
  if (candidate == kDefaultSemanticHelperEndpoint ||
      candidate == kIsolatedSemanticHelperEndpoint) {
    config.valid = true;
    config.endpoint.assign(candidate.data(), candidate.size());
    return config;
  }

  config.error =
      "VIPERSQL_SEMANTIC_HELPER_ENDPOINT must be exactly "
      "tcp://127.0.0.1:5555 or tcp://127.0.0.1:5556";
  return config;
}

/** Read and freeze the selector once for the lifetime of the mysqld process. */
inline const SemanticHelperEndpointConfig &
CachedSemanticHelperEndpointConfig() {
  static const SemanticHelperEndpointConfig config =
      ResolveSemanticHelperEndpoint(
          std::getenv(kSemanticHelperEndpointEnvironment));
  return config;
}

/** Invalid explicit configuration never falls back to the production helper. */
inline const std::string &SemanticHelperEndpoint() {
  const SemanticHelperEndpointConfig &config =
      CachedSemanticHelperEndpointConfig();
  if (!config.valid) throw std::runtime_error(config.error);
  return config.endpoint;
}

}  // namespace semhelpers

#endif  // SQL_ITERATORS_HELPERS_SEMANTIC_HELPER_ENDPOINT_H_
