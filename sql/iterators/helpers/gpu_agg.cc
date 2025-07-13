#include "sql/iterators/helpers/gpu_agg.h"
#include <cstdlib>
#include <iostream>

// Simple error‐checking macro
#define CUDA_CHECK(call)                                                   \
  do {                                                                     \
    cudaError_t _e = (call);                                               \
    if (_e != cudaSuccess) {                                               \
      std::cerr << "CUDA Error: " << cudaGetErrorString(_e)               \
                << " at " << __FILE__ << ":" << __LINE__ << "\n";         \
      return false;                                                        \
    }                                                                      \
  } while (0)

//
// gpu_agg_init:
//   Allocate and zero-initialize the on-GPU structures.
//   capacity is a hint for how many distinct groups we will see.
//
bool gpu_agg_init(GPUHashTable**   ht,
                  GPUAccumulator** acc,
                  size_t           capacity) {
  if ((capacity & (capacity-1)) != 0) return false;

  // 1) Allocate your device arrays
  uint64_t* d_keys;      cudaMalloc(&d_keys,      capacity * sizeof(uint64_t));
  double*   d_sums;      cudaMalloc(&d_sums,      capacity*MAX_SUM_FUNCS*sizeof(double));
  long*     d_counts;    cudaMalloc(&d_counts,    capacity*MAX_SUM_FUNCS*sizeof(long));
  cudaMemset(d_keys,   0xFF, capacity * sizeof(uint64_t));
  cudaMemset(d_sums,    0,   capacity*MAX_SUM_FUNCS*sizeof(double));
  cudaMemset(d_counts,  0,   capacity*MAX_SUM_FUNCS*sizeof(long));

  // 2) Allocate host‐side structs
  GPUHashTable* hth = new GPUHashTable{ d_keys, capacity };
  GPUAccumulator* ach = new GPUAccumulator{ d_sums, d_counts };

  *ht  = hth;
  *acc = ach;
  return true;
}

//
// gpu_agg_download:
//   Bring back every non‐empty bucket into host memory.
//   out_keys and out_vals must be at least capacity‐sized;
//   on return *out_n is the actual number of occupied slots.
//
bool gpu_agg_download(GPUHashTable*   ht,
                      GPUAccumulator* acc,
                      uint64_t*       out_keys,
                      Payload*        out_vals,
                      size_t*         out_n)
{
  size_t cap = ht->capacity;

  // 1) allocate flat host buffers
  uint64_t* h_keys   = (uint64_t*) std::malloc(cap * sizeof(uint64_t));
  double*   h_sums   = (double*  ) std::malloc(cap * MAX_SUM_FUNCS * sizeof(double));
  long*     h_counts = (long*    ) std::malloc(cap * MAX_SUM_FUNCS * sizeof(long));
  if (!h_keys || !h_sums || !h_counts) {
    std::cerr << "OOM in gpu_agg_download\n";
    return false;
  }

  // 2) copy back keys, sums, and counts separately
  CUDA_CHECK(cudaMemcpy(h_keys,   ht->d_keys,
                        cap * sizeof(uint64_t),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_sums,   acc->d_sum_vals,
                        cap * MAX_SUM_FUNCS * sizeof(double),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_counts, acc->d_count_vals,
                        cap * MAX_SUM_FUNCS * sizeof(long),
                        cudaMemcpyDeviceToHost));

  // 3) filter occupied slots by checking h_keys[i] != EMPTY_KEY_LL
  size_t cnt = 0;
  for (size_t i = 0; i < cap; ++i) {
    if (h_keys[i] != EMPTY_KEY_LL) {
      out_keys[cnt] = h_keys[i];
      // reconstruct Payload from the flat arrays
      for (int f = 0; f < MAX_SUM_FUNCS; ++f) {
        out_vals[cnt].sum_vals[f]   = h_sums  [i*MAX_SUM_FUNCS + f];
        out_vals[cnt].count_vals[f] = h_counts[i*MAX_SUM_FUNCS + f];
      }
      ++cnt;
    }
  }
  *out_n = cnt;

  // 4) clean up
  std::free(h_keys);
  std::free(h_sums);
  std::free(h_counts);

  return true;
}

//
// gpu_agg_destroy:
//   Free all GPU memory and host side structs.
//
void gpu_agg_destroy(GPUHashTable*   ht,
                     GPUAccumulator* acc)
{
  if (ht) {
    cudaFree(ht->d_keys);
    std::free(ht);
  }
  if (acc) {
    cudaFree(acc->d_sum_vals);
    cudaFree(acc->d_count_vals);
    std::free(acc);
  }
}
