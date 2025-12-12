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

// bool parse_string_from_blob(Field *field, std::string &data) {
//   const Field_blob *field_blob = down_cast<const Field_blob *>(field);
//   const uint32 blob_length = field_blob->get_length();
//   const uchar *const blob_data = field_blob->get_blob_data();
//   data.assign(reinterpret_cast<const char *>(blob_data), blob_length);
//   return false;
// }

// bool parse_string_from_item(Item **args, uint arg_idx, String &str,
//                             const char *func_name, std::string &value,
//                             std::string *field_name) {
//   if (args[arg_idx]->data_type() == MYSQL_TYPE_VARCHAR) {
//     String *tmp_str = args[arg_idx]->val_str(&str);
//     if (!tmp_str) {
//       my_error(ER_WRONG_ARGUMENTS, MYF(0), func_name);
//       return true;
//     }
//     value.assign(tmp_str->ptr(), tmp_str->length());
//     if (field_name) field_name->clear();
//     return false;
//   }

//   if (args[arg_idx]->data_type() == MYSQL_TYPE_BLOB &&
//       args[arg_idx]->type() == Item::FIELD_ITEM) {
//     const Item_field *fi = down_cast<const Item_field *>(args[arg_idx]);
//     if (parse_string_from_blob(fi->field, value)) {
//       my_error(ER_INCORRECT_TYPE, MYF(0), std::to_string(arg_idx).c_str(),
//                func_name);
//       return true;
//     }
//     if (field_name)
//       *field_name = std::string(fi->table_name) + "." + fi->field_name;
//     return false;
//   }

//   return true;
// }

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

