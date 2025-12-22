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
  parent class of semantic filter functions
*/
class Item_func_semantic_filter : public Item_int_func {
 public:
  Item_func_semantic_filter(THD *thd, const POS &pos, PT_item_list *a);

  bool resolve_type(THD *thd) override;

  // double val_real() override;

  std::string compute_prompt();
  longlong val_int() override;

 protected:
  /// String used when reading JSON binary values or JSON text values.
  String m_value;
};

/**
  Represents the function SEMANTIC_FILTER_SINGLE_COL()
*/
class Item_func_semantic_filter_single_col final : public Item_func_semantic_filter {
 public:
  Item_func_semantic_filter_single_col(THD *thd, const POS &pos, PT_item_list *a);

  const char *func_name() const override;
  enum Functype functype() const override;
};

/**
  Represents the function SEMANTIC_FILTER_TWO_COL()
*/
class Item_func_semantic_filter_two_col final : public Item_func_semantic_filter {
 public:
  Item_func_semantic_filter_two_col(THD *thd, const POS &pos, PT_item_list *a);

  const char *func_name() const override;
  enum Functype functype() const override;
};

class Item_func_semantic_generate final : public Item_func_semantic_filter {
 public:
  Item_func_semantic_generate(THD *thd, const POS &pos, PT_item_list *a);

  const char *func_name() const override;
  enum Functype functype() const override;
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


// Tools
static bool get_item_string(Item *it, String &tmp, std::string &out);
bool parse_string_from_item(Item **args, uint arg_idx, String &str,
                     const char *func_name, std::string &value, std::string *field_name);
bool parse_string_from_blob(Field *field, std::string &data);
Item *find_semantic_func(Item *node);


#endif  // ITEM_FUNC_SEMANTIC_H
