/*
   Copyright (c) 2025, Songsong Mo

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License; version 2 of the
   License.
*/

#include "sql/item_func_semantic.h"

#include <map>

#include "mysqld_error.h"
#include "sql/item_cmpfunc.h"
#include "sql/sql_exception_handler.h"

namespace {

bool get_item_string(Item *it, String &tmp, std::string &out) {
  if (!it) return true;
  String *s = it->val_str(&tmp);
  if (it->null_value || !s) return true;
  out.clear();
  if (s->length() != 0) {
    if (s->ptr() == nullptr) return true;
    out.assign(s->ptr(), s->length());
  }
  return false;
}

bool parse_string_from_blob(Field *field, std::string &data) {
  if (field == nullptr) return true;
  const Field_blob *field_blob = down_cast<const Field_blob *>(field);
  const uint32 blob_length = field_blob->get_length();
  if (blob_length == 0) {
    data.clear();
    return false;
  }
  const uchar *const blob_data = field_blob->get_blob_data();
  if (blob_data == nullptr) return true;
  data.assign(reinterpret_cast<const char *>(blob_data), blob_length);
  return false;
}

bool parse_string_from_item(Item **args, uint arg_idx, String &str,
                            const char *func_name, std::string &value,
                            std::string *field_name) {
  Item *item = args[arg_idx];
  if (item == nullptr) {
    my_error(ER_WRONG_ARGUMENTS, MYF(0), func_name);
    return true;
  }
  if (item->data_type() == MYSQL_TYPE_VARCHAR) {
    String *tmp_str = item->val_str(&str);
    if (!tmp_str) {
      if (item->null_value) return true;
      my_error(ER_WRONG_ARGUMENTS, MYF(0), func_name);
      return true;
    }
    if (item->null_value) return true;
    value.clear();
    if (tmp_str->length() != 0) {
      if (tmp_str->ptr() == nullptr) {
        my_error(ER_WRONG_ARGUMENTS, MYF(0), func_name);
        return true;
      }
      value.assign(tmp_str->ptr(), tmp_str->length());
    }
    if (field_name != nullptr) {
      field_name->clear();
    }
    return false;
  }
  if (item->data_type() == MYSQL_TYPE_BLOB &&
      item->type() == Item::FIELD_ITEM) {
    const Item_field *fi = down_cast<const Item_field *>(item);
    if (fi->field != nullptr && fi->field->is_null()) return true;
    if (parse_string_from_blob(fi->field, value)) {
      my_error(ER_INCORRECT_TYPE, MYF(0), std::to_string(arg_idx).c_str(),
               func_name);
      return true;
    }
    if (field_name != nullptr) {
      *field_name = std::string(fi->table_name) + "." + fi->field_name;
    }
    return false;
  }

  my_error(ER_INCORRECT_TYPE, MYF(0), std::to_string(arg_idx).c_str(),
           func_name);
  return true;
}

}  // namespace

Item_func_sem_join *AsSemJoin(Item *item) {
  if (item == nullptr || item->type() != Item::FUNC_ITEM) return nullptr;

  Item_func *f = down_cast<Item_func *>(item);
  return dynamic_cast<Item_func_sem_join *>(f);
}

bool ItemHasSemJoin(Item *item) {
  if (item == nullptr) return false;

  if (AsSemJoin(item) != nullptr) return true;

  if (item->type() == Item::COND_ITEM) {
    Item_cond *c = down_cast<Item_cond *>(item);
    List_iterator<Item> it(*c->argument_list());
    Item *arg;
    while ((arg = it++)) {
      if (ItemHasSemJoin(arg)) return true;
    }
    return false;
  }

  if (item->type() == Item::FUNC_ITEM) {
    Item_func *f = down_cast<Item_func *>(item);

    for (uint i = 0; i < f->argument_count(); ++i) {
      Item *arg = f->arguments()[i];
      if (ItemHasSemJoin(arg)) return true;
    }
    return false;
  }

  return false;
}

bool JoinConditionsHaveSemJoin(const std::vector<Item *> &conds) {
  for (Item *item : conds) {
    if (AsSemJoin(item) != nullptr) return true;
  }
  return false;
}

Item *find_semantic_func(Item *node) {
  if (!node) return nullptr;

  if (auto *sf = dynamic_cast<Item_func_semantic_filter *>(node)) {
    return sf;
  }

  auto *fn = dynamic_cast<Item_func *>(node);
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

Item_func_semantic_filter *AsSemanticFilterPredicate(Item *item) {
  if (auto *filter = dynamic_cast<Item_func_semantic_filter *>(item)) {
    return filter;
  }

  auto *function = dynamic_cast<Item_func *>(item);
  if (function == nullptr || function->functype() != Item_func::EQ_FUNC ||
      function->argument_count() != 2) {
    return nullptr;
  }

  Item *left = function->arguments()[0];
  Item *right = function->arguments()[1];
  auto is_literal_one = [](Item *candidate) {
    const auto *integer = dynamic_cast<const Item_int *>(candidate);
    return integer != nullptr && integer->value == 1;
  };

  if (auto *filter = dynamic_cast<Item_func_semantic_filter *>(left);
      filter != nullptr && is_literal_one(right)) {
    return filter;
  }
  if (auto *filter = dynamic_cast<Item_func_semantic_filter *>(right);
      filter != nullptr && is_literal_one(left)) {
    return filter;
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
  my_error(ER_NOT_SUPPORTED_YET, MYF(0),
           "semantic filter outside vectorized execution");
  null_value = true;
  return 0;
}

std::string Item_func_semantic_filter::compute_prompt() {
  try {
    std::string prompt;
    std::string value1;
    std::string field_name1;
    if (parse_string_from_item(args, 0, m_value, func_name(), prompt,
                               nullptr) ||
        parse_string_from_item(args, 1, m_value, func_name(), value1,
                               &field_name1)) {
      return "";
    }
    std::map<std::string, std::string> value_dict;
    if (!field_name1.empty()) {
      value_dict[field_name1] = value1;
    } else {
      value_dict["value1"] = value1;
    }
    if (arg_count == 3) {
      std::string value2;
      std::string field_name2;
      if (parse_string_from_item(args, 2, m_value, func_name(), value2,
                                 &field_name2)) {
        return "";
      }
      if (!field_name2.empty()) {
        value_dict[field_name2] = value2;
      } else {
        value_dict["value2"] = value2;
      }
    }

    std::string context;
    context += prompt + "\n";
    for (const auto &pair : value_dict) {
      context += pair.first + ": " + pair.second + "\n";
    }

    return context;
  } catch (...) {
    handle_std_exception(func_name());
    return "";
  }
}

Item_func_semantic_filter_single_col::Item_func_semantic_filter_single_col(
    THD *thd, const POS &pos, PT_item_list *a)
    : Item_func_semantic_filter(thd, pos, a) {}

const char *Item_func_semantic_filter_single_col::func_name() const {
  return "semantic_filter";
}

enum Item_func::Functype Item_func_semantic_filter_single_col::functype()
    const {
  return SEMANTIC_FILTER_SINGLE_COL;
}

Item_func_semantic_filter_two_col::Item_func_semantic_filter_two_col(
    THD *thd, const POS &pos, PT_item_list *a)
    : Item_func_semantic_filter(thd, pos, a) {}

const char *Item_func_semantic_filter_two_col::func_name() const {
  return "semantic_filter_two_col";
}

enum Item_func::Functype Item_func_semantic_filter_two_col::functype() const {
  return SEMANTIC_FILTER_TWO_COL;
}

const char *Item_func_semantic_generate::func_name() const {
  return "semantic_generate";
}

enum Item_func::Functype Item_func_semantic_generate::functype() const {
  return SEMANTIC_GENERATE;
}

Item_func_semantic_generate::Item_func_semantic_generate(THD *thd,
                                                         const POS &pos,
                                                         PT_item_list *a)
    : Item_func_semantic_filter(thd, pos, a) {}

bool Item_func_semantic_generate::resolve_type(THD *) {
  my_error(ER_NOT_SUPPORTED_YET, MYF(0), "SEMANTIC_GENERATE");
  return true;
}

bool Item_func_sem_join::resolve_type(THD *) {
  decimals = 0;
  max_length = 1;
  return false;
}

longlong Item_func_sem_join::val_int() {
  my_error(ER_NOT_SUPPORTED_YET, MYF(0),
           "SEM_JOIN outside a direct join predicate");
  null_value = true;
  return 0;
}

std::string Item_func_sem_join::prompt() {
  std::string p;
  if (get_item_string(args[0], m_tmp, p)) return {};
  return p;
}
