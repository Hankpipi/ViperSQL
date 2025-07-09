#ifndef SQL_ITERATORS_EXTERNAL_HELPER_INTERFACE_H_
#define SQL_ITERATORS_EXTERNAL_HELPER_INTERFACE_H_

#include <iostream>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <string>
#include <chrono>


static constexpr int BATCH_SIZE = 65536;

inline void log_to_file(const std::string& msg) {
  auto now = std::chrono::system_clock::now();
  std::time_t now_time = std::chrono::system_clock::to_time_t(now);

  std::ofstream log_file("/home/zihao/spatial-x-db/debug.log", std::ios::app);
  if (log_file.is_open()) {
      log_file << std::ctime(&now_time) << msg << std::endl;
      log_file.close();
  } else {
      std::cerr << "Unable to open log file." << std::endl;
  }
}

// Base class interface for GPU helpers
class ExternalHelperInterface {
public:
  virtual ~ExternalHelperInterface() = default;

  /// Initialize GPU resources (buffers, streams, etc.)
  /// @param capacity maximum capacity or buffer size
  /// @return true on failure, false on success
  virtual bool Init(size_t capacity) = 0;

  /// Submit a batch of input tuples to GPU for processing
  /// @param host_data pointer to host input data
  /// @param n_rows number of tuples in batch
  /// @param stream CUDA stream for asynchronous execution
  /// @return true on failure, false on success
  virtual bool SubmitBatch(const void* host_data, size_t n_rows) = 0;

  /// Fetch results from GPU to host
  /// @param out_buffer host buffer to copy results into
  /// @param max_results maximum results buffer size
  /// @param out_result_count number of results copied back
  /// @return true on failure, false on success
  virtual bool FetchResults(void* out_buffer, size_t* out_result_count) = 0;

  /// Synchronize GPU execution and streams
  /// @return true on failure, false on success
  virtual bool Synchronize() = 0;

  /// Release GPU resources
  virtual void Destroy() = 0;

  /// @brief Set the status of the external helpers.
  /// @param status 
  virtual void SetStatus(const std::string& status) = 0;
};

#endif  // SQL_ITERATORS_EXTERNAL_HELPER_INTERFACE_H_
