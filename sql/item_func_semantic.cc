/*
   Copyright (c) 2025, Songsong Mo

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License; version 2 of the License.
*/

#include "sql/item_func_semantic.h"
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

// ------------ Tools --------------

static bool get_item_string(Item *it, String &tmp, std::string &out) {
  if (!it) return true;
  String *s = it->val_str(&tmp);
  if (it->null_value || !s) return true;
  out.assign(s->ptr(), s->length());
  return false;
}

bool parse_string_from_blob(Field *field, std::string &data) {
  const Field_blob *field_blob = down_cast<const Field_blob *>(field);
  const uint32 blob_length = field_blob->get_length();
  const uchar *const blob_data = field_blob->get_blob_data();
  data.assign(reinterpret_cast<const char*>(blob_data), blob_length);
  return false;
}

bool parse_string_from_item(Item **args, uint arg_idx, String &str,
                     const char *func_name, std::string &value, std::string *field_name) {
  // log_to_file("parse_string_from_item called with arg_idx: " + std::to_string(arg_idx));
  if (args[arg_idx]->data_type() == MYSQL_TYPE_VARCHAR) {
    // log_to_file("input string literal");
    String *tmp_str = args[arg_idx]->val_str(&str);  // Evaluate the item into `str`
    if (!tmp_str) {
        // log_to_file("Failed to evaluate item to string");
        my_error(ER_WRONG_ARGUMENTS, MYF(0), func_name);
        return true;
    }
    // log_to_file("String length: " + std::to_string(tmp_str->length()));
    // Copy value into std::string
    value.assign(tmp_str->ptr(), tmp_str->length());
    if (field_name != nullptr) {
      field_name->clear();
    }
    return false;
 }
  if (args[arg_idx]->data_type() == MYSQL_TYPE_BLOB &&
      args[arg_idx]->type() == Item::FIELD_ITEM) {
    // log_to_file("input field blob");
    const Item_field *fi = down_cast<const Item_field *>(args[arg_idx]);
    if (parse_string_from_blob(fi->field, value)) {
      my_error(ER_INCORRECT_TYPE, MYF(0), std::to_string(arg_idx).c_str(),
               func_name);
      return true;
    }
    // log_to_file("Parsed value from blob: " + value);
    *field_name = std::string(fi->table_name) + "." + fi->field_name;
    return false;
  }

  return true;
}

Item* find_semantic_func(Item* node) {
  if (!node) return nullptr;

  // 1) If this node *is* a semantic‐filter, bingo:
  if (auto *sf = dynamic_cast<Item_func_semantic_filter*>(node)) {
    return sf;
  }

  // 2) Otherwise, if it's any other function, recurse into its arguments
  auto *fn = dynamic_cast<Item_func*>(node);
  if (!fn) return nullptr;

  uint cnt = fn->argument_count();
  for (uint i = 0; i < cnt; i++) {
    Item *child = fn->arguments()[i];
    if (Item *hit = find_semantic_func(child)) {
      return hit;
    }
  }

  return nullptr;
}

Item_func_semantic_filter::Item_func_semantic_filter(THD * /* thd */,
                                                           const POS &pos,
                                                           PT_item_list *a)
    : Item_int_func(pos, a) {}

bool Item_func_semantic_filter::resolve_type(THD *thd) {
  if (args[1]->data_type() != MYSQL_TYPE_BLOB) {
    my_error(ER_WRONG_ARGUMENTS, MYF(0), func_name());
    return true;
  }
  if (param_type_is_default(thd, 1, 2, MYSQL_TYPE_BLOB)) return true;
  set_nullable(true);

  return false;
}

longlong Item_func_semantic_filter::val_int() {
  return 1;
}

std::string Item_func_semantic_filter::compute_prompt() {
  // log_to_file("Item_func_semantic_filter::val_int() called");
  if (args[0]->null_value || args[1]->null_value) {
    return "";
  }

  try {
    // log_to_file("starting parsing prompt and value");
    std::string prompt;
    std::string value1;
    std::string field_name1;
    if (parse_string_from_item(args, 0, m_value, func_name(), prompt, nullptr) ||
        parse_string_from_item(args, 1, m_value, func_name(), value1, &field_name1)) {
      return "";
    }
    // log_to_file("Parsed prompt: " + prompt);
    // log_to_file("Parsed value1: " + value1);
    std::map<std::string, std::string> value_dict;
    if (!field_name1.empty()) {
      value_dict[field_name1] = value1;
    }
    else {
      value_dict["value1"] = value1;
    }
    if (arg_count == 3) {
      if (args[2]->null_value) {
        return "";
      }
      std::string value2;
      std::string field_name2;
      if (parse_string_from_item(args, 2, m_value, func_name(), value2, &field_name2)) {
        return "";
      }
      // log_to_file("Parsed value2: " + value2);
      if (!field_name2.empty()) {
        value_dict[field_name2] = value2;
      }
      else {
        value_dict["value2"] = value2;
      }
    }

    std::string context="";
    context += (prompt + "\n");
    for (const auto& pair : value_dict) {
      context += (pair.first + ": " + pair.second + "\n");
    }

    return context;
  } catch (...) {
    handle_std_exception(func_name());
    return "";
  }

  return "";
}

// ------------ Semantic Filter --------------

Item_func_semantic_filter_single_col::Item_func_semantic_filter_single_col(THD *thd, const POS &pos,
                                               PT_item_list *a)
    : Item_func_semantic_filter(thd, pos, a) {}

const char *Item_func_semantic_filter_single_col::func_name() const { return "semantic_filter"; }

enum Item_func::Functype Item_func_semantic_filter_single_col::functype() const {
  return SEMANTIC_FILTER_SINGLE_COL;
}

Item_func_semantic_filter_two_col::Item_func_semantic_filter_two_col(THD *thd, const POS &pos,
                                               PT_item_list *a)
    : Item_func_semantic_filter(thd, pos, a) {}

const char *Item_func_semantic_filter_two_col::func_name() const { return "semantic_filter_two_col"; }

enum Item_func::Functype Item_func_semantic_filter_two_col::functype() const {
  return SEMANTIC_FILTER_TWO_COL;
}

// ------------ Semantic Generate --------------

const char *Item_func_semantic_generate::func_name() const { return "semantic_generate"; }

enum Item_func::Functype Item_func_semantic_generate::functype() const {
  return SEMANTIC_GENERATE;
}

Item_func_semantic_generate::Item_func_semantic_generate(THD *thd, const POS &pos,
                                               PT_item_list *a)
    : Item_func_semantic_filter(thd, pos, a) {}

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

