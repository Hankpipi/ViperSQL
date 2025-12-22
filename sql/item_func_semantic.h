/*
   Copyright (c) 2025, Songsong Mo

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License; version 2 of the License.
*/

#ifndef ITEM_FUNC_SEMANTIC_H
#define ITEM_FUNC_SEMANTIC_H

#include "sql/item.h"
#include "sql/item_func.h"
#include "sql/sql_class.h"
#include "sql/parse_tree_items.h"
#include <string>
#include <map>

#include "sql/iterators/external_helper_interface.h"


/**
  Semantic Filter function:
    SEM_FILTER('Is {poi.text} a positive comment?', poi.text)
*/
class Item_func_sem_filter : public Item_int_func {
public:
  Item_func_sem_filter(Item *prompt, Item *col)
      : Item_int_func(prompt, col) {}

  const char *func_name() const override { return "sem_filter"; }
  Item_result result_type() const override { return INT_RESULT; }
  bool resolve_type(THD *thd) override;
  longlong val_int() override;
  std::string compute_value();
  std::string compute_predicate();
private:
  String m_tmp;
};


/**
  Semantic Join function:
    SEM_JOIN('Is {A.content} relevant to {B.topic}?', A.content, B.topic)
*/
class Item_func_sem_join : public Item_int_func {
public:
  Item_func_sem_join(Item *prompt, Item *a, Item *b)
    : Item_int_func(prompt, a, b) {}
  Item_func_sem_join(THD *thd, const POS &pos, PT_item_list *item_list)
      : Item_int_func(pos, item_list) {}
  Item_func_sem_join(const POS &pos, Item *prompt, Item *a, Item *b)
    : Item_int_func(pos, prompt, a, b) {}

  const char *func_name() const override { return "sem_join"; }
  Item_result result_type() const override { return INT_RESULT; }
  bool resolve_type(THD *thd) override;
  longlong val_int() override;
  std::string prompt();
  Item *left_item() const { return args[1]; }
  Item *right_item() const { return args[2]; }
private:
  String m_tmp;
};


// 解析工具函数声明
static bool get_item_string(Item *it, String &tmp, std::string &out);

#endif  // ITEM_FUNC_SEMANTIC_H
