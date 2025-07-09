#include "sql/iterators/gpu_helpers/gpu_agg.h"
#include <iostream>

#define CUDA_CHECK(call)                                                   \
  do {                                                                     \
    cudaError_t _e = (call);                                               \
    if (_e != cudaSuccess) {                                               \
      std::cerr << "CUDA Error: "                                          \
                << cudaGetErrorString(_e)                                 \
                << " at " << __FILE__ << ":" << __LINE__ << "\n";         \
      return false;                                                        \
    }                                                                      \
  } while (0)

// kernel definition matches the declaration in gpu_agg.h:
extern "C" __global__
void gpu_agg_kernel(uint64_t*       table_keys,
                    double*         sum_vals,
                    long*           count_vals,
                    size_t          capacity,
                    const uint64_t* in_keys,
                    const Payload*  in_vals,
                    size_t          n_rows)
{
  const size_t tid = blockIdx.x*blockDim.x + threadIdx.x;
  if (tid >= n_rows) return;

  const uint64_t key = in_keys[tid];
  const Payload p   = in_vals[tid];
  const size_t mask = capacity - 1;

  size_t idx = (size_t)key & mask;
  while (true) {
    unsigned long long old_ll =
      atomicCAS(
        reinterpret_cast<unsigned long long*>(&table_keys[idx]),
        EMPTY_KEY_LL,
        (unsigned long long)key);
    uint64_t old = (uint64_t)old_ll;
    if (old == EMPTY_KEY_LL || old == key) {
      break;
    }
    idx = (idx + 1) & mask;
  }

  const size_t base = idx * MAX_SUM_FUNCS;
  #pragma unroll
  for (int f = 0; f < MAX_SUM_FUNCS; f++) {
    atomicAdd(&sum_vals[base + f], p.sum_vals[f]);
    atomicAdd(reinterpret_cast<unsigned long long*>(&count_vals[base + f]),
              (unsigned long long)p.count_vals[f]);
  }
}

bool gpu_agg_batch(GPUHashTable*   ht,
                   GPUAccumulator* acc,
                   const uint64_t* host_keys,
                   const Payload*  host_vals,
                   uint64_t* d_in_keys,
                   Payload*  d_in_vals,
                   size_t          n_rows,
                   cudaStream_t    stream)
{
  // Make sure previous batch finished
  cudaStreamSynchronize(stream);

  // Async copy to device (overlap-able with compute on other streams)
  CUDA_CHECK(cudaMemcpyAsync(d_in_keys, host_keys,
                              n_rows * sizeof(*d_in_keys),
                              cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(d_in_vals, host_vals,
                              n_rows * sizeof(*d_in_vals),
                              cudaMemcpyHostToDevice, stream));

  // Kernel launch (on same stream, so queued after copies)
  const int threads = 256;
  const int blocks  = (n_rows + threads - 1) / threads;
  gpu_agg_kernel<<<blocks, threads, 0, stream>>>(
      ht->d_keys,
      acc->d_sum_vals,
      acc->d_count_vals,
      ht->capacity,
      d_in_keys,
      d_in_vals,
      n_rows
  );
  CUDA_CHECK(cudaGetLastError());
  return true;
}