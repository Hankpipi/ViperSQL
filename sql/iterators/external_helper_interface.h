#ifndef SQL_ITERATORS_EXTERNAL_HELPER_INTERFACE_H_
#define SQL_ITERATORS_EXTERNAL_HELPER_INTERFACE_H_

#include <cstddef>
#include <cstdint>
#include <string>

// Default number of rows submitted to an external helper.
constexpr size_t kExternalHelperBatchSize = 50;

// Helper operations return true on failure and false on success.
class ExternalHelperInterface {
 public:
  virtual ~ExternalHelperInterface() = default;

  // Initialize helper resources.
  virtual bool Init() = 0;

  // Submit a batch of input tuples for processing.
  virtual bool SubmitBatch(const void *host_data, size_t n_rows) = 0;

  // Fetch results into host memory.
  virtual bool FetchResults(void *out_buffer, size_t *out_result_count) = 0;

  // Required host slots after Synchronize(). Helpers with fan-out override
  // this; one-to-one helpers use the submitted-row count.
  virtual size_t ResultBufferCapacity(size_t submitted_rows) const {
    return submitted_rows;
  }

  // Client-observed service time for the most recently completed batch.
  // Zero means that the helper does not expose timing. The non-pure default
  // keeps existing ExternalHelper implementations source compatible.
  virtual double LastBatchServiceSeconds() const { return 0.0; }

  // Opaque process/configuration epoch reported by a remote helper.
  virtual std::string LastServerEpoch() const { return {}; }

  // Synchronize helper execution.
  virtual bool Synchronize() = 0;

  // Nonblocking readiness check for the single request owned by this helper.
  virtual bool IsIdle() const = 0;

  // Release helper resources.
  virtual void Destroy() = 0;

  // Set the phase of helpers that maintain build/probe state.
  virtual void SetStatus(const std::string &) {}
};

#endif  // SQL_ITERATORS_EXTERNAL_HELPER_INTERFACE_H_
