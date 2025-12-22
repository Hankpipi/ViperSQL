/*
   Copyright (c) 2025, Songsong Mo

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License; version 2 of the License.
*/

#include "sql/item_func_semantic.h"
#include "sql/item_semantic_filter_func.h"
#include "sql/item_json_func.h"
#include "sql/sql_exception_handler.h"
#include "mysqld_error.h"
#include <stdexcept>
#include <map>

namespace {
#define SEMANTICDB_DISABLED_ERR                                            \
  do {                                                                      \
    my_error(ER_FEATURE_DISABLED, MYF(0), "semantic db", "WITH_SEMANTICDB"); \
    return error_real();                                                    \
  } while (0)
}  // anonymous namespace

// ---------- 通用解析函数 ----------

static bool get_item_string(Item *it, String &tmp, std::string &out) {
  if (!it) return true;
  String *s = it->val_str(&tmp);
  if (it->null_value || !s) return true;
  out.assign(s->ptr(), s->length());
  return false;
}

// =============== SEM_FILTER ==================
bool Item_func_sem_filter::resolve_type(THD *) {
  decimals = 0;
  max_length = 1;
  return false;
}

longlong Item_func_sem_filter::val_int() {
  null_value = false;
  return 1;
}

std::string Item_func_sem_filter::compute_value() {
  std::string v;
  if (get_item_string(args[1], m_tmp, v)) return {};
  return v;
}

std::string Item_func_sem_filter::compute_predicate() {
  std::string p;
  if (get_item_string(args[0], m_tmp, p)) return {};
  return p;
}

// =============== SEM_JOIN ==================
bool Item_func_sem_join::resolve_type(THD *) {
  decimals = 0;
  max_length = 1;
  return false;
}

longlong Item_func_sem_join::val_int() {
  null_value = false;
  return 1;
}

std::string Item_func_sem_join::prompt() {
  std::string p;
  if (get_item_string(args[0], m_tmp, p)) return {};
  return p;
}

