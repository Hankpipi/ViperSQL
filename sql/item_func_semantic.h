/*
   Copyright (c) 2025, Songsong Mo

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License; version 2 of the
   License.
*/

#ifndef ITEM_FUNC_SEMANTIC_H
#define ITEM_FUNC_SEMANTIC_H

#include <string>
#include <vector>

#include "sql/item.h"
#include "sql/item_func.h"
#include "sql/parse_tree_items.h"
#include "sql/sql_class.h"

/** Base class for semantic filter functions. */
class Item_func_semantic_filter : public Item_int_func {
 public:
  Item_func_semantic_filter(THD *thd, const POS &pos, PT_item_list *a);
  Item_func_semantic_filter(Item *prompt, Item *a, Item *b)
      : Item_int_func(prompt, a, b) {}

  bool resolve_type(THD *thd) override;

  std::string compute_prompt();
  longlong val_int() override;
  bool is_bool_func() const override { return true; }
  bool is_semantic_operator() const override { return true; }
  double get_semantic_cost(const Cost_model_server *cm) const override {
    return cm->row_semantic_evaluate_cost(1.0);
  }

 protected:
  String m_value;
};

/** Implements SEMANTIC_FILTER_SINGLE_COL(). */
class Item_func_semantic_filter_single_col final
    : public Item_func_semantic_filter {
 public:
  Item_func_semantic_filter_single_col(THD *thd, const POS &pos,
                                       PT_item_list *a);

  const char *func_name() const override;
  enum Functype functype() const override;
};

/** Implements SEMANTIC_FILTER_TWO_COL(). */
class Item_func_semantic_filter_two_col final
    : public Item_func_semantic_filter {
 public:
  Item_func_semantic_filter_two_col(THD *thd, const POS &pos, PT_item_list *a);
  Item_func_semantic_filter_two_col(Item *prompt, Item *a, Item *b)
      : Item_func_semantic_filter(prompt, a, b) {}

  const char *func_name() const override;
  enum Functype functype() const override;
};

class Item_func_semantic_generate final : public Item_func_semantic_filter {
 public:
  Item_func_semantic_generate(THD *thd, const POS &pos, PT_item_list *a);

  bool resolve_type(THD *thd) override;
  const char *func_name() const override;
  enum Functype functype() const override;
};

/** Implements SEM_JOIN(). */
class Item_func_sem_join : public Item_int_func {
 public:
  Item_func_sem_join(Item *prompt, Item *a, Item *b)
      : Item_int_func(prompt, a, b) {}
  Item_func_sem_join(THD *thd, const POS &pos, PT_item_list *item_list)
      : Item_int_func(pos, item_list) {}
  Item_func_sem_join(const POS &pos, Item *prompt, Item *a, Item *b)
      : Item_int_func(pos, prompt, a, b) {}

  const char *func_name() const override { return "sem_join"; }
  bool resolve_type(THD *thd) override;
  longlong val_int() override;
  bool is_bool_func() const override { return true; }
  std::string prompt();
  Item *left_item() const { return args[1]; }
  Item *right_item() const { return args[2]; }
  bool is_semantic_operator() const override { return true; }
  double get_semantic_cost(const Cost_model_server *cm) const override {
    return cm->row_semantic_evaluate_cost(1.0);
  }

 private:
  String m_tmp;
};

Item *find_semantic_func(Item *node);
/**
  Return the semantic filter represented by a supported top-level predicate.

  In addition to the direct boolean form, accept an equality to the literal
  integer 1 on either side. Other wrappers remain unsupported so the
  vectorized executor never changes the surrounding expression semantics.
*/
Item_func_semantic_filter *AsSemanticFilterPredicate(Item *item);
bool JoinConditionsHaveSemJoin(const std::vector<Item *> &conds);
bool ItemHasSemJoin(Item *item);
Item_func_sem_join *AsSemJoin(Item *item);

#endif  // ITEM_FUNC_SEMANTIC_H
