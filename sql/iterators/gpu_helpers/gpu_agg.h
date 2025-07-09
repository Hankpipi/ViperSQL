#ifndef SQL_ITERATORS_GPU_AGG_H_
#define SQL_ITERATORS_GPU_AGG_H_

#include <cuda_runtime.h>
#include "sql/my_decimal.h"
#include "sql/item.h"
#include "sql/item_sum.h"
#include "sql/table.h"
#include "sql/sql_executor.h"
#include "sql/iterators/external_helper_interface.h"

// Forward declarations
struct GPUHashTable;
struct GPUAccumulator;
struct Item_sum;

// Must match aggr.cu’s capacity usage
static constexpr int MAX_SUM_FUNCS = 5;
static constexpr unsigned long long EMPTY_KEY_LL = (unsigned long long)(~0ULL);

// Payload mirrors the device layout in aggr.cu
struct Payload {
  double sum_vals[MAX_SUM_FUNCS];
  long   count_vals[MAX_SUM_FUNCS];
};

//
// Internals: our on‐GPU hash table and accumulator
//
struct GPUHashTable {
  uint64_t* d_keys;
  size_t    capacity;
};

struct GPUAccumulator {
  double* d_sum_vals;    // [capacity * MAX_SUM_FUNCS]
  long*   d_count_vals;  // [capacity * MAX_SUM_FUNCS]
};

// GPU aggregator lifecycle APIs
bool gpu_agg_init(GPUHashTable**   ht,
                  GPUAccumulator** acc,
                  size_t           capacity);

bool gpu_agg_batch(GPUHashTable*    ht,
                   GPUAccumulator*  acc,
                   const uint64_t*  host_keys,
                   const Payload*   host_vals,
                   uint64_t* d_in_keys,
                   Payload*  d_in_vals,
                   size_t           n_rows,
                   cudaStream_t    stream);

bool gpu_agg_download(GPUHashTable*    ht,
                      GPUAccumulator*  acc,
                      uint64_t*        out_keys,
                      Payload*         out_vals,
                      size_t*          out_n);

void gpu_agg_destroy(GPUHashTable*    ht,
                     GPUAccumulator*  acc);

inline std::string uchar_to_hex(const uchar* buffer, size_t len) {
    std::ostringstream oss;
    for (size_t i = 0; i < len; ++i) {
        oss << std::hex << std::setw(2) << std::setfill('0') << (int)buffer[i];
    }
    return oss.str();
}

inline void write_payload_to_record(TABLE *table,
                                   Item_sum **sum_items,
                                   const Payload &payload) {
  uchar *rec0 = table->record[0];
  uint key_fields = table->key_info[0].user_defined_key_parts;
  uint idx = key_fields;

  // idx in payload
  for (size_t agg_idx = 0; sum_items[agg_idx]; ++agg_idx, ++idx) {
    Field *fld = table->field[idx];

    // Always clear the NULL bit for NOT NULL output
    if (fld->is_nullable()) {
      uchar *null_bytes = rec0 + fld->null_offset();
      *null_bytes &= ~fld->null_bit;
    }

    uchar *dest = rec0 + static_cast<size_t>(fld->offset(rec0));

    switch (sum_items[agg_idx]->sum_func()) {
      case Item_sum::SUM_FUNC:
      case Item_sum::SUM_DISTINCT_FUNC: {
        if (fld->type() == MYSQL_TYPE_NEWDECIMAL) {
          my_decimal dec;
          double v = payload.sum_vals[agg_idx];
          double2my_decimal(E_DEC_FATAL_ERROR, v, &dec);
          int precision = fld->field_length;
          int scale = fld->decimals();
          my_decimal2binary(0, &dec, dest, precision, scale);
        } else if (fld->type() == MYSQL_TYPE_DOUBLE) {
          double v = payload.sum_vals[agg_idx];
          memcpy(dest, &v, sizeof(double));
        }
        break;
      }
      case Item_sum::COUNT_FUNC:
      case Item_sum::COUNT_DISTINCT_FUNC: {
        if (fld->type() == MYSQL_TYPE_LONGLONG) {
          longlong cnt = payload.count_vals[agg_idx];
          memcpy(dest, &cnt, sizeof(longlong));
        } else if (fld->type() == MYSQL_TYPE_NEWDECIMAL) {
          my_decimal dec;
          double2my_decimal(E_DEC_FATAL_ERROR, (double)payload.count_vals[agg_idx], &dec);
          int precision = fld->field_length;
          int scale = fld->decimals();
          my_decimal2binary(0, &dec, dest, precision, scale);
        }
        break;
      }
      case Item_sum::AVG_FUNC:
      case Item_sum::AVG_DISTINCT_FUNC: {
        uchar *dest = rec0 + fld->offset(rec0);
        // The packed decimal region (excluding the COUNT)
        int pack_length = fld->pack_length() - (int)sizeof(longlong);
        if (pack_length >= 16) {
          int precision = 2 * pack_length + 4;
          int scale = 0;
          my_decimal dec;
          double v = payload.sum_vals[agg_idx];
          double2my_decimal(E_DEC_FATAL_ERROR, v, &dec);
          my_decimal2binary(0, &dec, dest, precision, scale);
        }
        else {
          double v = payload.sum_vals[agg_idx];
          memcpy(dest, &v, sizeof(double));
        }

        // Write COUNT immediately after packed decimal region
        longlong cnt = payload.count_vals[agg_idx];
        memcpy(dest + pack_length, &cnt, sizeof(longlong));

        break;
      }
      default:
        // Optionally handle other aggregates
        assert(false);
    }
  }
}

/**
 * Fill the GROUP BY key columns into record[0], storing
 * the 64-bit hash if present, or else replaying each real
 * GROUP BY expression into its tmp-table field.
 *
 * @returns true on error (copy_funcs failure), false on success
 */
inline bool write_group_key_to_record(
  TABLE *table,
  uint64_t hash_key,
  const std::unordered_map<uint64_t, std::string> *hash_to_rawkey) {

  uchar *rec0 = table->record[0];

  if (table->hash_field) {
    table->hash_field->store(hash_key, /*save_org=*/ true);
  } else {
    ORDER *group;
    KEY_PART_INFO *key_part;

    // If map is null or empty, use hash_key as raw bytes
    const uchar *key_bytes = nullptr;
    size_t key_len = 0;

    if (hash_to_rawkey->empty()) {
      key_bytes = reinterpret_cast<const uchar *>(&hash_key);
      key_len = sizeof(uint64_t);
    } else {
      // Lookup the raw value in the map
      auto it = hash_to_rawkey->find(hash_key);
      if (it == hash_to_rawkey->end()) {
          // Error: can't find the raw value
          return true;
      }
      key_bytes = reinterpret_cast<const uchar *>(it->second.data());
      key_len = it->second.size();
    }

    // Walk the group columns and copy the correct key bytes to their offsets
    size_t copied = 0;
    for (group = table->group, key_part = table->key_info[0].key_part;
        group && copied < key_len;
        group = group->next, key_part++) {

      // Null indicator
      if (key_part->null_bit)
          memcpy(rec0 + key_part->offset - 1, group->buff - 1, 1);

      // Only copy as many bytes as needed for this key part
      size_t part_len = key_part->length;
      if (copied + part_len > key_len)
          part_len = key_len - copied;

      memcpy(rec0 + key_part->offset, key_bytes + copied, part_len);
      copied += part_len;
    }
  }
  return false;
}

// ------------------------------------------------------------------
// Compute the “delta” for each SUM/COUNT in sum_funcs[] and store
// it into the fixed-size arrays in delta.
//   - sum_funcs is null-terminated, up to MAX_SUM_FUNCS entries.
//   - For COUNT(*) (no args) we emit count=1, sum=0.
//   - For SUM(x) we emit sum=val_real(), count=0.
//   - For COUNT(x) we emit count=(arg0->is_null()?0:1), sum=0.
// ------------------------------------------------------------------
inline void compute_payload(Item_sum **sum_funcs, Payload &delta) {
  unsigned n_funcs = 0;
  for (; sum_funcs[n_funcs]; ++n_funcs) {}

  for (unsigned f = 0; f < n_funcs; ++f) {
    Item_sum *sf = sum_funcs[f];
    switch (sf->sum_func()) {
      case Item_sum::SUM_FUNC:
      case Item_sum::SUM_DISTINCT_FUNC: {
        Item *a = sf->arg(0);
        double v = (a && !a->is_null()) ? a->val_real() : 0.0;
        delta.sum_vals[f] = v;
        delta.count_vals[f] = 0;
        break;
      }
      case Item_sum::COUNT_FUNC:
      case Item_sum::COUNT_DISTINCT_FUNC: {
        if (sf->arg_num_count() == 0) {
          delta.sum_vals[f] = 0.0;
          delta.count_vals[f] = 1;
        } else {
          Item *a = sf->arg(0);
          delta.sum_vals[f] = 0.0;
          delta.count_vals[f] = (a && !a->is_null()) ? 1 : 0;
        }
        break;
      }
      case Item_sum::AVG_FUNC: {
        Item *a = sf->arg(0);
        double v = (a && !a->is_null()) ? a->val_real() : 0.0;
        delta.sum_vals[f] = v;
        delta.count_vals[f] = (a && !a->is_null()) ? 1 : 0;
        break;
      }
      default:
        delta.sum_vals[f] = 0.0;
        delta.count_vals[f] = 0;
        break;
    }
  }

  for (unsigned f = n_funcs; f < MAX_SUM_FUNCS; ++f) {
    delta.sum_vals[f] = 0.0;
    delta.count_vals[f] = 0;
  }
}

#endif 