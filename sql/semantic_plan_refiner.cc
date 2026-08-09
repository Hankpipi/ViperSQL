/* Copyright (c) 2026, ViperSQL contributors.

   Semantic-aware predicate migration for the classic optimizer.
*/

#include "sql/semantic_plan_refiner.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>
#include <vector>

#include "my_bitmap.h"
#include "sql/field.h"
#include "sql/handler.h"
#include "sql/item_cmpfunc.h"
#include "sql/item_func_semantic.h"
#include "sql/semantic_plan_tiebreak_policy.h"
#include "sql/semantic_profile.h"
#include "sql/sql_class.h"
#include "sql/sql_optimizer.h"
#include "sql/sql_select.h"
#include "sql/table.h"

namespace vipersql {
namespace {

constexpr double kMinimumRows = 1.0;
constexpr double kComparisonEpsilon = 1.0e-12;

bool IsAndCondition(Item *item) {
  return item != nullptr && item->type() == Item::COND_ITEM &&
         static_cast<Item_cond *>(item)->functype() ==
             Item_func::COND_AND_FUNC;
}

void CollectTopLevelConjuncts(Item *condition,
                              std::vector<Item *> *conjuncts) {
  if (!IsAndCondition(condition)) {
    conjuncts->push_back(condition);
    return;
  }

  List_iterator<Item> iterator(
      *static_cast<Item_cond *>(condition)->argument_list());
  Item *child;
  while ((child = iterator++)) CollectTopLevelConjuncts(child, conjuncts);
}

double PositiveFiniteOr(double value, double fallback) {
  return std::isfinite(value) && value > 0.0 ? value : fallback;
}

double NonnegativeFiniteOr(double value, double fallback) {
  return std::isfinite(value) && value >= 0.0 ? value : fallback;
}

double ClampSelectivity(double value) {
  if (!std::isfinite(value) || value <= 0.0 || value > 1.0) return 1.0;
  return value;
}

bool IsUnhistogrammedTextField(const Item_field *item_field) {
  if (item_field == nullptr || item_field->field == nullptr ||
      item_field->field->table == nullptr) {
    return false;
  }
  const Field *const field = item_field->field;
  switch (field->real_type()) {
    case MYSQL_TYPE_TINY_BLOB:
    case MYSQL_TYPE_MEDIUM_BLOB:
    case MYSQL_TYPE_LONG_BLOB:
    case MYSQL_TYPE_BLOB:
      return field->table->s->find_histogram(field->field_index()) == nullptr;
    default:
      return false;
  }
}

/**
  Estimate equality/IN selectivity for TEXT columns without a collected
  histogram. The stock fallback is 10% per value, which overestimates results
  for high-cardinality text. sqrt(N) distinct values avoids a fixed
  selectivity that grows linearly with table size without assuming uniqueness.

  @return a replacement selectivity, or a negative value if this conjunct is
          not a supported local constant predicate.
*/
double TextConstantFallbackSelectivity(Item *condition,
                                       table_map filter_for_table,
                                       double rows_in_table) {
  if (condition == nullptr || condition->type() != Item::FUNC_ITEM ||
      (condition->used_tables() & ~PSEUDO_TABLE_BITS) != filter_for_table) {
    return -1.0;
  }

  Item_func *const function = static_cast<Item_func *>(condition);
  Item_field *field = nullptr;
  size_t constant_values = 0;
  if (function->functype() == Item_func::IN_FUNC) {
    Item_func_in *const in = static_cast<Item_func_in *>(function);
    if (in->negated || function->argument_count() < 2) return -1.0;
    Item *const real_item = function->arguments()[0]->real_item();
    if (real_item->type() != Item::FIELD_ITEM) return -1.0;
    field = static_cast<Item_field *>(real_item);
    for (uint i = 1; i < function->argument_count(); ++i) {
      if (!function->arguments()[i]->const_item()) return -1.0;
    }
    constant_values = function->argument_count() - 1;
  } else if ((function->functype() == Item_func::EQ_FUNC ||
              function->functype() == Item_func::EQUAL_FUNC) &&
             function->argument_count() == 2) {
    for (uint i = 0; i < 2; ++i) {
      Item *const real_item = function->arguments()[i]->real_item();
      if (real_item->type() == Item::FIELD_ITEM &&
          function->arguments()[1 - i]->const_item()) {
        field = static_cast<Item_field *>(real_item);
        constant_values = 1;
        break;
      }
    }
  } else if (function->functype() == Item_func::MULT_EQUAL_FUNC) {
    Item_equal *const equality = static_cast<Item_equal *>(function);
    if (equality->const_arg() == nullptr) return -1.0;
    for (Item_field &member : equality->get_fields()) {
      if ((member.used_tables() & ~PSEUDO_TABLE_BITS) == filter_for_table &&
          IsUnhistogrammedTextField(&member)) {
        field = &member;
        constant_values = 1;
        break;
      }
    }
  }

  if (!IsUnhistogrammedTextField(field) || constant_values == 0) return -1.0;
  const double estimated_distinct = std::sqrt(std::max(1.0, rows_in_table));
  return std::clamp(static_cast<double>(constant_values) /
                        estimated_distinct,
                    1.0 / std::max(1.0, rows_in_table), 1.0);
}

double EstimateLocalFilterSelectivity(const JOIN &join, JOIN_TAB *join_tab,
                                      double base_rows) {
  if (join.where_cond == nullptr || join_tab == nullptr ||
      join_tab->table() == nullptr || join_tab->table_ref == nullptr) {
    return 1.0;
  }

  MY_BITMAP fields_to_ignore;
  if (bitmap_init(&fields_to_ignore, nullptr,
                  join_tab->table()->s->fields)) {
    return 1.0;
  }
  bitmap_clear_all(&fields_to_ignore);
  double selectivity = join.where_cond->get_filtering_effect(
      join.thd, join_tab->table_ref->map(), 0, &fields_to_ignore, base_rows);

  // Without a collected histogram MySQL falls back to a fixed 10% equality
  // selectivity (20% for a two-value IN list). Correct only that missing-stats
  // case; indexed and histogram-backed estimates remain authoritative.
  std::vector<Item *> conjuncts;
  CollectTopLevelConjuncts(join.where_cond, &conjuncts);
  for (Item *const conjunct : conjuncts) {
    const double fallback = TextConstantFallbackSelectivity(
        conjunct, join_tab->table_ref->map(), base_rows);
    if (!(fallback > 0.0)) continue;
    const double native = conjunct->get_filtering_effect(
        join.thd, join_tab->table_ref->map(), 0, &fields_to_ignore, base_rows);
    if (native > 0.0 && native <= 1.0 && fallback < native) {
      selectivity *= fallback / native;
    }
  }
  bitmap_free(&fields_to_ignore);
  return ClampSelectivity(selectivity);
}

bool HasTrustedLocalFilterEstimate(const JOIN &join, JOIN_TAB *join_tab,
                                   double base_rows) {
  if (join.where_cond == nullptr || join_tab == nullptr ||
      join_tab->table_ref == nullptr) {
    return false;
  }
  const table_map table = join_tab->table_ref->map();
  std::vector<Item *> conjuncts;
  CollectTopLevelConjuncts(join.where_cond, &conjuncts);
  for (Item *const conjunct : conjuncts) {
    if ((conjunct->used_tables() & ~PSEUDO_TABLE_BITS) != table) continue;
    if (TextConstantFallbackSelectivity(conjunct, table, base_rows) > 0.0) {
      return true;
    }
    if (conjunct->type() != Item::FUNC_ITEM) continue;
    Item_func *const function = static_cast<Item_func *>(conjunct);
    for (uint i = 0; i < function->argument_count(); ++i) {
      Item *const real_item = function->arguments()[i]->real_item();
      if (real_item->type() != Item::FIELD_ITEM) continue;
      Item_field *const field = static_cast<Item_field *>(real_item);
      if ((field->used_tables() & ~PSEUDO_TABLE_BITS) == table &&
          field->field != nullptr && field->field->table != nullptr &&
          field->field->table->s->find_histogram(
              field->field->field_index()) != nullptr) {
        return true;
      }
    }
  }
  return false;
}

bool LocalConjunctHasTrustedFilterEstimate(Item *conjunct,
                                           table_map filter_for_table,
                                           double base_rows) {
  const bool null_safe_equality =
      conjunct != nullptr && conjunct->type() == Item::FUNC_ITEM &&
      static_cast<Item_func *>(conjunct)->functype() ==
          Item_func::EQUAL_FUNC;
  // The sqrt(N) text fallback does not model NULL multiplicity. Do not use it
  // as fanout-trust evidence for <=>; a real histogram may still prove the
  // local reduction below. Existing migration semantics remain unchanged.
  if (!null_safe_equality &&
      TextConstantFallbackSelectivity(conjunct, filter_for_table, base_rows) >
          0.0) {
    return true;
  }
  if (conjunct == nullptr || conjunct->type() != Item::FUNC_ITEM) return false;
  Item_func *const function = static_cast<Item_func *>(conjunct);
  for (uint i = 0; i < function->argument_count(); ++i) {
    Item *const real_item = function->arguments()[i]->real_item();
    if (real_item->type() != Item::FIELD_ITEM) continue;
    Item_field *const field = static_cast<Item_field *>(real_item);
    if ((field->used_tables() & ~PSEUDO_TABLE_BITS) == filter_for_table &&
        field->field != nullptr && field->field->table != nullptr &&
        field->field->table->s->find_histogram(field->field->field_index()) !=
            nullptr) {
      return true;
    }
  }
  return false;
}

/**
  Return true only when at least one local conjunct is estimated to reduce the
  introduced table and every such reducing conjunct has trusted statistics.

  This differs from HasTrustedLocalFilterEstimate(), whose any-conjunct
  semantics are retained for the existing migration rule. A
  trusted histogram predicate must not mask a second unhistogrammed LIKE when
  deciding whether an optional many-side table is safe before a semantic
  stage.
*/
bool HasOnlyTrustedFanoutFilters(const JOIN &join, JOIN_TAB *join_tab,
                                 double base_rows) {
  if (join.where_cond == nullptr || join_tab == nullptr ||
      join_tab->table() == nullptr || join_tab->table_ref == nullptr) {
    return false;
  }

  MY_BITMAP fields_to_ignore;
  if (bitmap_init(&fields_to_ignore, nullptr,
                  join_tab->table()->s->fields)) {
    return false;
  }
  bitmap_clear_all(&fields_to_ignore);
  const table_map table = join_tab->table_ref->map();
  bool saw_reducing_conjunct = false;
  std::vector<Item *> conjuncts;
  CollectTopLevelConjuncts(join.where_cond, &conjuncts);
  for (Item *const conjunct : conjuncts) {
    if ((conjunct->used_tables() & ~PSEUDO_TABLE_BITS) != table) continue;
    const double selectivity = conjunct->get_filtering_effect(
        join.thd, table, 0, &fields_to_ignore, base_rows);
    if (!std::isfinite(selectivity) || selectivity < 0.0 ||
        selectivity > 1.0) {
      bitmap_free(&fields_to_ignore);
      return false;
    }
    if (selectivity + kComparisonEpsilon >= 1.0) continue;
    saw_reducing_conjunct = true;
    if (!LocalConjunctHasTrustedFilterEstimate(conjunct, table, base_rows)) {
      bitmap_free(&fields_to_ignore);
      return false;
    }
  }
  bitmap_free(&fields_to_ignore);
  return saw_reducing_conjunct;
}

bool CanMigrateAcross(const POSITION &position) {
  if (position.table == nullptr || position.table->table_ref == nullptr)
    return false;
  if (position.table->emb_sj_nest != nullptr) return false;
  return !position.table->table_ref->is_inner_table_of_outer_join();
}

struct EqualityEdge {
  Item_field *left{nullptr};
  Item_field *right{nullptr};
  table_map left_table{0};
  table_map right_table{0};
  double unique_domain{0.0};
};

struct RelationalEdge {
  table_map tables{0};
};

void AddRelationalEdge(table_map tables,
                       std::vector<RelationalEdge> *edges) {
  if (tables == 0 || (tables & (tables - 1)) == 0) return;
  for (const RelationalEdge &edge : *edges) {
    if (edge.tables == tables) return;
  }
  edges->push_back({tables});
}

double SingleColumnUniqueDomain(Item_field *item_field) {
  if (item_field == nullptr || item_field->field == nullptr ||
      item_field->field->table == nullptr)
    return 0.0;

  Field *const field = item_field->field;
  TABLE *const table = field->table;
  for (uint key_number = 0; key_number < table->s->keys; ++key_number) {
    KEY *const key = table->key_info + key_number;
    if (key->user_defined_key_parts != 1 ||
        (key_number != table->s->primary_key &&
         (key->flags & HA_NOSAME) == 0) ||
        !key->key_part[0].field->eq(field)) {
      continue;
    }
    return PositiveFiniteOr(static_cast<double>(table->file->stats.records),
                            0.0);
  }
  return 0.0;
}

SemanticFanoutUniqueEvidence IntroducedSideUniqueEvidence(
    Item_field *item_field) {
  if (item_field == nullptr || item_field->field == nullptr ||
      item_field->field->table == nullptr) {
    return SemanticFanoutUniqueEvidence::kNone;
  }
  Field *const field = item_field->field;
  TABLE *const table = field->table;
  bool participates_in_composite_unique = false;
  for (uint key_number = 0; key_number < table->s->keys; ++key_number) {
    KEY *const key = table->key_info + key_number;
    if (key_number != table->s->primary_key && (key->flags & HA_NOSAME) == 0) {
      continue;
    }
    for (uint part = 0; part < key->user_defined_key_parts; ++part) {
      Field *const key_field = key->key_part[part].field;
      if (key_field == nullptr || !key_field->eq(field)) continue;
      if (key->user_defined_key_parts == 1) {
        return SemanticFanoutUniqueEvidence::kSingleColumnUnique;
      }
      participates_in_composite_unique = true;
    }
  }
  return participates_in_composite_unique
             ? SemanticFanoutUniqueEvidence::kCompositeUniqueUnknown
             : SemanticFanoutUniqueEvidence::kNone;
}

void AddEqualityEdge(Item_field *left, Item_field *right,
                     table_map query_tables,
                     std::vector<EqualityEdge> *edges,
                     double equality_domain = 0.0) {
  if (left == nullptr || right == nullptr || left->field == nullptr ||
      right->field == nullptr || left->field->eq(right->field))
    return;

  const table_map left_table = left->used_tables() & query_tables;
  const table_map right_table = right->used_tables() & query_tables;
  if (left_table == 0 || right_table == 0 || left_table == right_table) return;

  for (const EqualityEdge &edge : *edges) {
    const bool same_direction = edge.left->field->eq(left->field) &&
                                edge.right->field->eq(right->field);
    const bool reverse_direction = edge.left->field->eq(right->field) &&
                                   edge.right->field->eq(left->field);
    if (same_direction || reverse_direction) return;
  }

  const double left_domain = SingleColumnUniqueDomain(left);
  const double right_domain = SingleColumnUniqueDomain(right);
  const double unique_domain =
      equality_domain > 0.0
          ? equality_domain
          : std::max(left_domain, right_domain);
  if (!(unique_domain > 0.0)) return;
  edges->push_back(
      {left, right, left_table, right_table, unique_domain});
}

Item_field *AsField(Item *item) {
  if (item == nullptr) return nullptr;
  Item *const real_item = item->real_item();
  return real_item->type() == Item::FIELD_ITEM
             ? static_cast<Item_field *>(real_item)
             : nullptr;
}

Item_func_sem_join *FindSemanticJoin(Item *item);

struct FanoutEqualityEdge {
  table_map left_table{0};
  table_map right_table{0};
  SemanticFanoutUniqueEvidence left_unique_evidence{
      SemanticFanoutUniqueEvidence::kNone};
  SemanticFanoutUniqueEvidence right_unique_evidence{
      SemanticFanoutUniqueEvidence::kNone};
};

void AddFanoutEqualityEdge(Item_field *left, Item_field *right,
                           table_map query_tables,
                           std::vector<FanoutEqualityEdge> *edges) {
  if (left == nullptr || right == nullptr || left->field == nullptr ||
      right->field == nullptr || left->field->eq(right->field)) {
    return;
  }
  const table_map left_table = left->used_tables() & query_tables;
  const table_map right_table = right->used_tables() & query_tables;
  if (left_table == 0 || right_table == 0 || left_table == right_table) return;

  const SemanticFanoutUniqueEvidence left_unique_evidence =
      IntroducedSideUniqueEvidence(left);
  const SemanticFanoutUniqueEvidence right_unique_evidence =
      IntroducedSideUniqueEvidence(right);
  for (const FanoutEqualityEdge &edge : *edges) {
    if (edge.left_table == left_table && edge.right_table == right_table &&
        edge.left_unique_evidence == left_unique_evidence &&
        edge.right_unique_evidence == right_unique_evidence) {
      return;
    }
    if (edge.left_table == right_table && edge.right_table == left_table &&
        edge.left_unique_evidence == right_unique_evidence &&
        edge.right_unique_evidence == left_unique_evidence) {
      return;
    }
  }
  edges->push_back({left_table, right_table, left_unique_evidence,
                    right_unique_evidence});
}

void CollectFanoutEqualityEdges(Item *condition, table_map query_tables,
                                std::vector<FanoutEqualityEdge> *edges) {
  if (condition == nullptr || find_semantic_func(condition) != nullptr ||
      FindSemanticJoin(condition) != nullptr) {
    return;
  }
  if (IsAndCondition(condition)) {
    List_iterator<Item> iterator(
        *static_cast<Item_cond *>(condition)->argument_list());
    Item *child;
    while ((child = iterator++))
      CollectFanoutEqualityEdges(child, query_tables, edges);
    return;
  }
  if (condition->type() != Item::FUNC_ITEM) return;

  Item_func *const function = static_cast<Item_func *>(condition);
  // Null-safe <=> is intentionally excluded: nullable UNIQUE indexes permit
  // multiple NULLs, all of which can match under null-safe equality.
  if (function->functype() == Item_func::EQ_FUNC &&
      function->argument_count() == 2) {
    AddFanoutEqualityEdge(AsField(function->arguments()[0]),
                          AsField(function->arguments()[1]), query_tables,
                          edges);
    return;
  }
  if (function->functype() != Item_func::MULT_EQUAL_FUNC) return;

  Item_equal *const equality = static_cast<Item_equal *>(condition);
  std::vector<Item_field *> fields;
  for (Item_field &field : equality->get_fields()) fields.push_back(&field);
  for (size_t i = 0; i + 1 < fields.size(); ++i) {
    for (size_t j = i + 1; j < fields.size(); ++j) {
      AddFanoutEqualityEdge(fields[i], fields[j], query_tables, edges);
    }
  }
}

struct SemanticPrefixFanoutRisk {
  uint uncertain_fanout_tables{0};
  bool evidence_valid{true};
};

SemanticPrefixFanoutRisk CountSemanticPrefixFanoutRisks(
    const JOIN &join, const POSITION *positions, uint placement_stage,
    table_map required_tables, const std::vector<FanoutEqualityEdge> &edges) {
  table_map earlier_prefix = 0;
  bool has_earlier_nonconst_prefix = false;
  SemanticPrefixFanoutRisk result;
  for (uint stage = 0; stage <= placement_stage; ++stage) {
    JOIN_TAB *const join_tab = positions[stage].table;
    const table_map new_table = join_tab->table_ref->map();
    const bool is_const_table = stage < join.const_tables;
    bool equality_connected = false;
    SemanticFanoutUniqueEvidence introduced_side_unique_evidence =
        SemanticFanoutUniqueEvidence::kNone;
    for (const FanoutEqualityEdge &edge : edges) {
      if (edge.left_table == new_table &&
          (edge.right_table & earlier_prefix) != 0) {
        equality_connected = true;
        introduced_side_unique_evidence = MergeSemanticFanoutUniqueEvidence(
            introduced_side_unique_evidence, edge.left_unique_evidence);
      } else if (edge.right_table == new_table &&
                 (edge.left_table & earlier_prefix) != 0) {
        equality_connected = true;
        introduced_side_unique_evidence = MergeSemanticFanoutUniqueEvidence(
            introduced_side_unique_evidence, edge.right_unique_evidence);
      }
    }

    bool only_trusted_reducing_filters = false;
    if (!is_const_table) {
      const double base_rows =
          PositiveFiniteOr(static_cast<double>(join_tab->records()), 1.0);
      only_trusted_reducing_filters =
          HasOnlyTrustedFanoutFilters(join, join_tab, base_rows);
    }
    const SemanticPrefixTableFanoutAssessment assessment =
        ClassifySemanticPrefixTableFanout(
            is_const_table, has_earlier_nonconst_prefix,
            (required_tables & new_table) != 0, equality_connected,
            introduced_side_unique_evidence, only_trusted_reducing_filters);
    result.uncertain_fanout_tables += assessment.uncertain_fanout_tables;
    result.evidence_valid &= assessment.evidence_valid;
    earlier_prefix |= new_table;
    if (!is_const_table) has_earlier_nonconst_prefix = true;
  }
  return result;
}

Item_func_sem_join *FindSemanticJoin(Item *item) {
  if (item == nullptr) return nullptr;
  if (Item_func_sem_join *const semantic_join = AsSemJoin(item))
    return semantic_join;
  if (item->type() == Item::COND_ITEM) {
    List_iterator<Item> iterator(
        *static_cast<Item_cond *>(item)->argument_list());
    Item *child;
    while ((child = iterator++)) {
      if (Item_func_sem_join *const found = FindSemanticJoin(child))
        return found;
    }
    return nullptr;
  }
  if (item->type() != Item::FUNC_ITEM) return nullptr;
  Item_func *const function = static_cast<Item_func *>(item);
  for (uint i = 0; i < function->argument_count(); ++i) {
    if (Item_func_sem_join *const found =
            FindSemanticJoin(function->arguments()[i]))
      return found;
  }
  return nullptr;
}

void CollectRelationalEdges(Item *condition, table_map query_tables,
                            std::vector<RelationalEdge> *edges) {
  if (condition == nullptr || find_semantic_func(condition) != nullptr ||
      FindSemanticJoin(condition) != nullptr)
    return;
  if (IsAndCondition(condition)) {
    List_iterator<Item> iterator(
        *static_cast<Item_cond *>(condition)->argument_list());
    Item *child;
    while ((child = iterator++))
      CollectRelationalEdges(child, query_tables, edges);
    return;
  }

  if (condition->type() == Item::FUNC_ITEM) {
    Item_func *const function = static_cast<Item_func *>(condition);
    if (function->functype() == Item_func::MULT_EQUAL_FUNC) {
      Item_equal *const equality = static_cast<Item_equal *>(condition);
      std::vector<table_map> field_tables;
      for (Item_field &field : equality->get_fields()) {
        const table_map table = field.used_tables() & query_tables;
        if (table != 0) field_tables.push_back(table);
      }
      if (field_tables.size() >= 2) {
        // Every pair in a multiple equality is a valid relational edge. A
        // star rooted at the first field is insufficient here: for a class
        // such as a=b=c it can falsely classify the legal prefix (b,c) as a
        // Cartesian transition merely because a happened to be the root.
        // (Unique-domain cardinality correction below still uses a spanning
        // tree so the equality selectivity is not counted more than once.)
        for (size_t i = 0; i + 1 < field_tables.size(); ++i) {
          for (size_t j = i + 1; j < field_tables.size(); ++j)
            AddRelationalEdge(field_tables[i] | field_tables[j], edges);
        }
      }
      return;
    }
  }

  AddRelationalEdge(condition->used_tables() & query_tables, edges);
}

bool HasReadyRelationalEdge(const std::vector<RelationalEdge> &edges,
                            table_map prefix, table_map new_table) {
  const table_map available = prefix | new_table;
  for (const RelationalEdge &edge : edges) {
    if ((edge.tables & prefix) != 0 && (edge.tables & new_table) != 0 &&
        (edge.tables & ~available) == 0) {
      return true;
    }
  }
  return false;
}

bool HasAvoidableCartesianPrefix(const JOIN &join,
                                 const POSITION *positions,
                                 uint plan_count) {
  if (positions == nullptr || plan_count < 2) return false;
  std::vector<RelationalEdge> edges;
  CollectRelationalEdges(join.where_cond, join.all_table_map, &edges);
  if (edges.empty()) return false;

  table_map prefix = positions[0].table->table_ref->map();
  for (uint stage = 1; stage < plan_count; ++stage) {
    const table_map new_table = positions[stage].table->table_ref->map();
    if (!HasReadyRelationalEdge(edges, prefix, new_table)) {
      bool connected_alternative = false;
      for (uint future = stage + 1; future < plan_count; ++future) {
        const table_map future_table =
            positions[future].table->table_ref->map();
        if (HasReadyRelationalEdge(edges, prefix, future_table)) {
          connected_alternative = true;
          break;
        }
      }
      // A Cartesian transition is necessary only after the current connected
      // component is exhausted. Flag orders that skip a ready join edge.
      if (connected_alternative) return true;
    }
    prefix |= new_table;
  }
  return false;
}

bool TablesAreRelationallyConnected(
    table_map left, table_map right,
    const std::vector<RelationalEdge> &edges) {
  if (left == 0 || right == 0) return false;
  table_map reachable = left;
  bool changed;
  do {
    changed = false;
    for (const RelationalEdge &edge : edges) {
      if ((edge.tables & reachable) == 0) continue;
      const table_map expanded = reachable | edge.tables;
      if (expanded != reachable) {
        reachable = expanded;
        changed = true;
      }
    }
  } while (changed);
  return (right & reachable) == right;
}

bool IsCanonicalPositiveSemJoin(Item *wrapper,
                                Item_func_sem_join *semantic_join) {
  if (wrapper == semantic_join) return true;
  if (wrapper == nullptr || wrapper->type() != Item::FUNC_ITEM) return false;
  Item_func *const function = static_cast<Item_func *>(wrapper);
  if (function->argument_count() != 2 ||
      (function->functype() != Item_func::EQ_FUNC &&
       function->functype() != Item_func::EQUAL_FUNC &&
       function->functype() != Item_func::NE_FUNC)) {
    return false;
  }

  Item *other = nullptr;
  if (function->arguments()[0] == semantic_join)
    other = function->arguments()[1];
  else if (function->arguments()[1] == semantic_join)
    other = function->arguments()[0];
  if (other == nullptr || !other->const_item()) return false;
  const longlong value = other->val_int();
  if (other->null_value) return false;
  return function->functype() == Item_func::NE_FUNC ? value == 0 : value == 1;
}

uint TableStage(table_map table, const POSITION *positions, uint plan_count) {
  for (uint stage = 0; stage < plan_count; ++stage) {
    if ((positions[stage].table->table_ref->map() & table) != 0) return stage;
  }
  return plan_count;
}

void CollectUniqueDomainEdges(Item *condition, table_map query_tables,
                              const POSITION *positions, uint plan_count,
                              std::vector<EqualityEdge> *edges) {
  if (condition == nullptr) return;
  if (IsAndCondition(condition)) {
    List_iterator<Item> iterator(
        *static_cast<Item_cond *>(condition)->argument_list());
    Item *child;
    while ((child = iterator++))
      CollectUniqueDomainEdges(child, query_tables, positions, plan_count,
                               edges);
    return;
  }
  if (condition->type() != Item::FUNC_ITEM) return;

  Item_func *const function = static_cast<Item_func *>(condition);
  // Null-safe equality can match arbitrarily many NULL values even when a
  // nullable column has a UNIQUE index. The ordinary unique-domain reduction
  // is therefore sound only for '=' (MULT_EQUAL remains handled below).
  if (function->functype() == Item_func::EQ_FUNC &&
      function->argument_count() == 2) {
    AddEqualityEdge(AsField(function->arguments()[0]),
                    AsField(function->arguments()[1]), query_tables, edges);
    return;
  }
  if (function->functype() != Item_func::MULT_EQUAL_FUNC) return;

  Item_equal *const equality = static_cast<Item_equal *>(condition);
  std::vector<Item_field *> fields;
  for (Item_field &field : equality->get_fields()) fields.push_back(&field);
  if (fields.size() < 2) return;

  // A multiple equality is one equivalence class, not a clique of
  // independent reductions. Build a plan-dependent spanning tree rooted at
  // the earliest member, so every later member contributes exactly one domain
  // reduction as soon as it can actually join the prefix. A fixed unique-key
  // root delays corrections for legal prefixes that do not yet contain that
  // table (for example, a fact-to-bridge prefix before the dimension table).
  size_t anchor = 0;
  uint anchor_stage = plan_count;
  double equality_domain = 0.0;
  for (size_t i = 0; i < fields.size(); ++i) {
    equality_domain =
        std::max(equality_domain, SingleColumnUniqueDomain(fields[i]));
    const uint stage = TableStage(fields[i]->used_tables() & query_tables,
                                  positions, plan_count);
    if (stage < anchor_stage) {
      anchor = i;
      anchor_stage = stage;
    }
  }
  if (!(equality_domain > 0.0) || anchor_stage == plan_count) return;
  std::vector<std::pair<uint, size_t>> non_anchor_fields;
  non_anchor_fields.reserve(fields.size() - 1);
  const table_map anchor_table =
      fields[anchor]->used_tables() & query_tables;
  for (size_t i = 0; i < fields.size(); ++i) {
    if (i == anchor) continue;
    const table_map member_table = fields[i]->used_tables() & query_tables;
    if (member_table == 0 || member_table == anchor_table) continue;
    non_anchor_fields.push_back(
        {TableStage(member_table, positions, plan_count),
         i});
  }
  std::sort(non_anchor_fields.begin(), non_anchor_fields.end());
  for (size_t edge_rank = 0; edge_rank < non_anchor_fields.size();
       ++edge_rank) {
    // Large multiway equality classes frequently connect correlated fact
    // tables through the same entity key. Treating every equality as an
    // independent 1/D reduction can collapse a real intermediate to one row.
    // Preserve exact two- and three-table behavior, then use exponential
    // backoff for additional members, as is customary for correlated
    // selectivities. Sorting by plan stage keeps the estimate order-stable.
    const size_t backoff_rank = edge_rank > 1 ? edge_rank - 1 : 0;
    const double domain_exponent =
        std::ldexp(1.0, -static_cast<int>(backoff_rank));
    const double effective_domain =
        std::pow(equality_domain, domain_exponent);
    AddEqualityEdge(fields[anchor], fields[non_anchor_fields[edge_rank].second],
                    query_tables, edges, effective_domain);
  }
}

std::vector<double> EstimateKeyDomainPrefixRows(
    const JOIN &join, const POSITION *positions, uint plan_count,
    const std::vector<double> &fallback_rows,
    bool apply_local_filters = true) {
  std::vector<EqualityEdge> edges;
  CollectUniqueDomainEdges(join.where_cond, join.all_table_map, positions,
                           plan_count, &edges);
  if (edges.empty()) return fallback_rows;

  std::vector<double> result(plan_count, 0.0);
  std::vector<bool> applied(edges.size(), false);
  table_map prefix_tables = 0;
  double estimated_rows = 1.0;
  size_t applied_edges = 0;
  for (uint stage = 0; stage < plan_count; ++stage) {
    JOIN_TAB *const join_tab = positions[stage].table;
    const table_map table_bit = join_tab->table_ref->map();
    const double base_rows =
        PositiveFiniteOr(static_cast<double>(join_tab->records()), 1.0);
    const double local_selectivity =
        apply_local_filters
            ? EstimateLocalFilterSelectivity(join, join_tab, base_rows)
            : 1.0;
    estimated_rows *= std::max(1.0, base_rows * local_selectivity);
    prefix_tables |= table_bit;

    for (size_t edge_number = 0; edge_number < edges.size(); ++edge_number) {
      if (applied[edge_number]) continue;
      const EqualityEdge &edge = edges[edge_number];
      if ((edge.left_table & prefix_tables) == edge.left_table &&
          (edge.right_table & prefix_tables) == edge.right_table) {
        estimated_rows /= edge.unique_domain;
        applied[edge_number] = true;
        ++applied_edges;
      }
    }

    // Before the first recognized PK/unique-key equality, native estimates
    // are a safer fallback. Afterwards this generic domain model corrects the
    // many-order-of-magnitude prefix overestimates on correlated schemas.
    result[stage] = applied_edges == 0
                        ? fallback_rows[stage]
                        : std::max(1.0, estimated_rows);
  }
  return result;
}

}  // namespace

Item *SemanticPlanRefiner::ExtractSemanticConjuncts(Item *condition) {
  predicates_.clear();
  winning_placement_.clear();
  winning_corrected_prefix_rows_.clear();
  has_winning_placement_ = false;
  if (condition == nullptr) return nullptr;

  std::vector<Item *> conjuncts;
  CollectTopLevelConjuncts(condition, &conjuncts);

  std::vector<RelationalEdge> relational_edges;
  const table_map query_tables = condition->used_tables();
  for (Item *conjunct : conjuncts)
    CollectRelationalEdges(conjunct, query_tables, &relational_edges);

  std::vector<Item *> relational_conjuncts;
  relational_conjuncts.reserve(conjuncts.size());
  for (Item *conjunct : conjuncts) {
    Item *semantic_function = find_semantic_func(conjunct);
    if (dynamic_cast<Item_func_semantic_filter *>(semantic_function) !=
        nullptr) {
      predicates_.push_back({conjunct, conjunct->used_tables()});
      continue;
    }

    Item_func_sem_join *const semantic_join = FindSemanticJoin(conjunct);
    const table_map left_tables =
        semantic_join == nullptr ? 0
                                 : semantic_join->left_item()->used_tables();
    const table_map right_tables =
        semantic_join == nullptr ? 0
                                 : semantic_join->right_item()->used_tables();
    if (semantic_join == nullptr ||
        !IsCanonicalPositiveSemJoin(conjunct, semantic_join) ||
        !TablesAreRelationallyConnected(left_tables, right_tables,
                                        relational_edges)) {
      relational_conjuncts.push_back(conjunct);
      continue;
    }

    // When ordinary predicates already connect both semantic arguments, this
    // SQL operator is a pairwise predicate over an existing relational tuple,
    // not a join that should initialize two independent children. Lower it to
    // the equivalent two-column semantic filter so correlated ref lookups
    // remain inside the relational plan.
    Item *lowered = new (thd_->mem_root) Item_func_semantic_filter_two_col(
        semantic_join->arguments()[0], semantic_join->left_item(),
        semantic_join->right_item());
    if (lowered == nullptr) {
      predicates_.clear();
      return condition;
    }
    lowered->apply_is_true();
    if (!lowered->fixed && lowered->fix_fields(thd_, &lowered)) {
      predicates_.clear();
      return condition;
    }
    lowered->update_used_tables();
    predicates_.push_back({lowered, lowered->used_tables()});
  }

  if (predicates_.empty()) return condition;
  if (relational_conjuncts.empty()) return nullptr;

  Item *relational_condition = relational_conjuncts.front();
  for (size_t i = 1; i < relational_conjuncts.size(); ++i) {
    Item_cond_and *combined = new (thd_->mem_root)
        Item_cond_and(relational_condition, relational_conjuncts[i]);
    if (combined == nullptr) {
      predicates_.clear();
      return condition;
    }
    combined->quick_fix_field();
    combined->update_used_tables();
    relational_condition = combined;
  }
  return relational_condition;
}

SemanticPlanEvaluation SemanticPlanRefiner::EvaluateCandidate(
    const JOIN &join, const POSITION *positions, uint plan_count) const {
  SemanticPlanEvaluation result;
  result.applicable = !predicates_.empty();
  if (!result.applicable) {
    if (positions != nullptr && plan_count != 0) {
      result.total_cost = positions[plan_count - 1].prefix_cost;
      result.output_rows = positions[plan_count - 1].prefix_rowcount;
    }
    return result;
  }

  if (positions == nullptr || plan_count == 0) {
    result.feasible = false;
    result.total_cost = std::numeric_limits<double>::infinity();
    return result;
  }

  const SemanticProfile &profile = GetSemanticProfile();
  const SemanticOperatorProfile &operator_profile = profile.unary_filter;
  if (operator_profile.operator_memory_bytes != 0 &&
      operator_profile.helper_memory_bytes != 0 &&
      operator_profile.operator_memory_bytes >
          operator_profile.helper_memory_bytes) {
    result.feasible = false;
    result.total_cost = std::numeric_limits<double>::infinity();
    return result;
  }

  std::vector<table_map> prefix_tables(plan_count, 0);
  std::vector<double> prefix_rows(plan_count, 0.0);
  std::vector<double> prefix_costs(plan_count, 0.0);
  table_map accumulated_tables = 0;
  for (uint stage = 0; stage < plan_count; ++stage) {
    const POSITION &position = positions[stage];
    if (position.table == nullptr || position.table->table_ref == nullptr ||
        position.table->table() == nullptr) {
      result.feasible = false;
      result.total_cost = std::numeric_limits<double>::infinity();
      return result;
    }

    accumulated_tables |= position.table->table_ref->map();
    prefix_tables[stage] = accumulated_tables;
    prefix_rows[stage] =
        NonnegativeFiniteOr(position.prefix_rowcount, 0.0);
    prefix_costs[stage] =
        NonnegativeFiniteOr(position.prefix_cost, 0.0);
  }
  if (join.const_tables < plan_count) {
    result.fanout_risk_root_table =
        positions[join.const_tables].table->table_ref->map();
  }
  const bool has_avoidable_cartesian_prefix =
      HasAvoidableCartesianPrefix(join, positions, plan_count);
  const std::vector<double> native_prefix_rows = prefix_rows;
  prefix_rows = EstimateKeyDomainPrefixRows(join, positions, plan_count,
                                            prefix_rows);
  const std::vector<double> unfiltered_prefix_rows =
      EstimateKeyDomainPrefixRows(join, positions, plan_count,
                                  native_prefix_rows,
                                  /*apply_local_filters=*/false);
  // Refined costs for semantic plans often contain the same dominant helper
  // cost. Retain a plan-robustness signal that distinguishes such near-ties:
  // lower cumulative intermediate cardinality means fewer tuples are exposed
  // to estimation error before the common final result. Exclude the complete
  // prefix, which represents the same logical join result for every order.
  for (uint stage = 0; stage + 1 < plan_count; ++stage) {
    result.intermediate_cardinality_score +=
        std::log1p(std::max(0.0, prefix_rows[stage]));
  }

  result.placement_stages.reserve(predicates_.size());
  uint minimum_stage = std::min(join.const_tables, plan_count - 1);
  // The access-path builder lifts a semantic condition
  // attached to QEP index zero so it can batch above a join. Costing stage
  // zero would therefore describe a physical plan that cannot be executed.
  if (join.const_tables == 0 && plan_count > 1) minimum_stage = 1;
  double preceding_semantic_selectivity = 1.0;
  const double semantic_selectivity =
      ClampSelectivity(operator_profile.default_selectivity);
  const Cost_model_server *const cost_model = join.cost_model();

  for (const SemanticPredicate &predicate : predicates_) {
    const table_map required_tables =
        predicate.required_tables & join.all_table_map;
    uint stage = plan_count;
    bool fuse_with_previous = false;
    if (!result.placement_stages.empty()) {
      const uint previous_stage = result.placement_stages.back();
      // A later conjunct whose inputs already exist at the preceding semantic
      // stage belongs to the same physical VectorizedFilterIterator request.
      // Moving it alone would split one fused stream into two helper streams,
      // contradicting the replay cost below.
      fuse_with_previous =
          (required_tables & prefix_tables[previous_stage]) == required_tables;
      if (fuse_with_previous) stage = previous_stage;
    }
    if (!fuse_with_previous) {
      for (uint candidate_stage = minimum_stage; candidate_stage < plan_count;
           ++candidate_stage) {
        if ((required_tables & prefix_tables[candidate_stage]) ==
            required_tables) {
          stage = candidate_stage;
          break;
        }
      }
    }
    if (stage == plan_count) {
      result.feasible = false;
      result.total_cost = std::numeric_limits<double>::infinity();
      result.placement_stages.clear();
      return result;
    }

    // Compare migration costs one relational boundary at a time. Fixed
    // predicate order is enforced by minimum_stage.
    while (!fuse_with_previous && stage + 1 < plan_count) {
      const uint next_stage = stage + 1;
      if (!CanMigrateAcross(positions[next_stage])) break;

      const double rows_before =
          prefix_rows[stage] * preceding_semantic_selectivity;
      const double rows_after =
          prefix_rows[next_stage] * preceding_semantic_selectivity;

      if (positions[next_stage].use_join_buffer) {
        JOIN_TAB *const next_join_tab = positions[next_stage].table;
        const double next_base_rows = PositiveFiniteOr(
            static_cast<double>(next_join_tab->records()), 1.0);
        const double next_filter_effect = EstimateLocalFilterSelectivity(
            join, next_join_tab, next_base_rows);
        const bool next_filter_is_trusted = HasTrustedLocalFilterEstimate(
            join, next_join_tab, next_base_rows);
        const double batch_size = static_cast<double>(
            std::max<size_t>(1, operator_profile.max_batch_size));
        constexpr double kMaxSmallBufferedBuildBatches = 64.0;
        const double maximum_small_build_rows =
            kMaxSmallBufferedBuildBatches * batch_size;

        // A semantic iterator inside a hash build can have a severe streaming
        // penalty. For a bounded small prefix and an unfiltered fan-out, lift
        // it across the physical buffer; the cap prevents this workaround
        // from multiplying helper work across a large expansion. A locally
        // filtered scan remains on the semantic-last side when its default
        // selectivity is too uncertain to prove a reduction.
        if (next_filter_effect + kComparisonEpsilon >= 1.0 &&
            rows_before <= maximum_small_build_rows) {
          stage = next_stage;
          continue;
        }
        if (next_filter_effect + kComparisonEpsilon < 1.0 &&
            !next_filter_is_trusted &&
            !(unfiltered_prefix_rows[next_stage] + kComparisonEpsilon <
              unfiltered_prefix_rows[stage])) {
          break;
        }
      }

      // Outside the bounded physical-buffer exception, reorder only when the
      // relational operator reduces the semantic input cardinality.
      if (!(rows_after + kComparisonEpsilon < rows_before)) break;

      const double cardinality_correction =
          prefix_rows[next_stage] /
          std::max(native_prefix_rows[next_stage], kMinimumRows);
      const double native_delta =
          std::max(0.0, prefix_costs[next_stage] - prefix_costs[stage]) *
          cardinality_correction;
      // Compare exact profiled costs. Invocation overhead uses the maximum
      // batch size, so semantic cost is discontinuous at batch boundaries.
      const double semantic_first =
          cost_model->row_semantic_evaluate_cost(rows_before) +
          semantic_selectivity * native_delta;
      const double relational_first =
          native_delta + cost_model->row_semantic_evaluate_cost(rows_after);
      if (!(relational_first + kComparisonEpsilon < semantic_first)) break;
      stage = next_stage;
    }

    result.placement_stages.push_back(stage);
    minimum_stage = stage;
    preceding_semantic_selectivity *= semantic_selectivity;
  }

  // Record two robustness signals for each fused semantic group. First, a
  // semantic iterator below a later buffered join is lowered into that join's
  // build subtree and loses streaming overlap. Second, equality-joined tables
  // already present in the semantic prefix can hide fanout when neither a
  // unique introduced-side key nor a trusted local filter bounds them. These
  // are tie-break evidence only; they never make a legal plan infeasible.
  std::vector<FanoutEqualityEdge> fanout_edges;
  CollectFanoutEqualityEdges(join.where_cond, join.all_table_map,
                             &fanout_edges);
  for (size_t first_predicate = 0;
       first_predicate < result.placement_stages.size();) {
    const uint placement = result.placement_stages[first_predicate];
    table_map required_tables = 0;
    size_t next_predicate = first_predicate;
    do {
      required_tables |=
          predicates_[next_predicate].required_tables & join.all_table_map;
      ++next_predicate;
    } while (next_predicate < result.placement_stages.size() &&
             result.placement_stages[next_predicate] == placement);

    result.semantic_group_input_tables.push_back(prefix_tables[placement]);
    result.semantic_group_predicate_counts.push_back(next_predicate -
                                                     first_predicate);
    const SemanticPrefixFanoutRisk fanout_risk =
        CountSemanticPrefixFanoutRisks(join, positions, placement,
                                       required_tables, fanout_edges);
    result.uncertain_fanout_tables += fanout_risk.uncertain_fanout_tables;
    result.fanout_risk_evidence_valid &= fanout_risk.evidence_valid;
    for (uint stage = placement + 1; stage < plan_count; ++stage) {
      if (positions[stage].use_join_buffer) {
        ++result.buffered_build_semantic_groups;
        break;
      }
    }
    first_predicate = next_predicate;
  }

  // Replay the refined pipeline in cost space. Each native relational prefix
  // delta is corrected for key-domain cardinality and scaled by semantic
  // predicates already executed; semantic costs are already in native units.
  double total_cost = 0.0;
  double flow_scale = 1.0;
  size_t next_predicate = 0;
  for (uint stage = 0; stage < plan_count; ++stage) {
    const double previous_prefix_cost =
        stage == 0 ? 0.0 : prefix_costs[stage - 1];
    const double relational_delta =
        std::max(0.0, prefix_costs[stage] - previous_prefix_cost) *
        (prefix_rows[stage] /
         std::max(native_prefix_rows[stage], kMinimumRows));
    total_cost += relational_delta * flow_scale;

    double rows_at_stage = prefix_rows[stage] * flow_scale;
    if (next_predicate < predicates_.size() &&
        result.placement_stages[next_predicate] == stage) {
      // ViperFlow evaluates all semantic conjuncts attached to one QEP stage
      // in a single helper request. Charge the physical batch stream once,
      // then apply every logical predicate's selectivity to the returned row
      // stream. Multiple predicates at this stage therefore produce one set
      // of helper batches rather than independent invocations.
      const uint64_t estimated_batches =
          EstimateUnarySemanticBatches(rows_at_stage);
      result.semantic_group_estimated_input_rows.push_back(rows_at_stage);
      result.semantic_group_estimated_batch_counts.push_back(
          estimated_batches);
      if (estimated_batches == std::numeric_limits<uint64_t>::max() ||
          result.estimated_batch_count >
              std::numeric_limits<uint64_t>::max() -
                  estimated_batches) {
        result.estimated_batch_count =
            std::numeric_limits<uint64_t>::max();
      } else {
        result.estimated_batch_count += estimated_batches;
      }
      total_cost += cost_model->row_semantic_evaluate_cost(rows_at_stage);
      do {
        rows_at_stage *= semantic_selectivity;
        flow_scale *= semantic_selectivity;
        ++next_predicate;
      } while (next_predicate < predicates_.size() &&
               result.placement_stages[next_predicate] == stage);
    }
  }

  if (next_predicate != predicates_.size() || !std::isfinite(total_cost) ||
      result.semantic_group_estimated_input_rows.size() !=
          result.semantic_group_input_tables.size() ||
      result.semantic_group_estimated_batch_counts.size() !=
          result.semantic_group_input_tables.size() ||
      result.semantic_group_predicate_counts.size() !=
          result.semantic_group_input_tables.size()) {
    result.feasible = false;
    result.total_cost = std::numeric_limits<double>::infinity();
    result.placement_stages.clear();
    return result;
  }

  // An avoidable Cartesian transition is a bad join-order heuristic, not an
  // illegal query plan. Keep the penalty finite so a forced legal order can
  // still populate best_positions when it is the only enumerated candidate.
  // Connected alternatives remain strongly preferred during normal search.
  if (has_avoidable_cartesian_prefix) {
    constexpr double kAvoidableCartesianMultiplier = 1.0e6;
    const double maximum_finite_cost =
        std::numeric_limits<double>::max() / 2.0;
    total_cost = std::min(
        maximum_finite_cost,
        (std::max(total_cost, 1.0) + 1.0) *
            kAvoidableCartesianMultiplier);
  }

  result.total_cost = total_cost;
  result.output_rows = prefix_rows.back() * flow_scale;
  result.corrected_prefix_rows = std::move(prefix_rows);
  return result;
}

void SemanticPlanRefiner::RememberWinningPlacement(
    const SemanticPlanEvaluation &evaluation) {
  if (!evaluation.applicable || !evaluation.feasible ||
      evaluation.placement_stages.size() != predicates_.size()) {
    return;
  }
  winning_placement_ = evaluation.placement_stages;
  winning_corrected_prefix_rows_ = evaluation.corrected_prefix_rows;
  has_winning_placement_ = true;
}

bool SemanticPlanRefiner::WinningCorrectedPrefixRows(uint stage,
                                                     double *rows) const {
  if (rows == nullptr || !has_winning_placement_ ||
      stage >= winning_corrected_prefix_rows_.size()) {
    return false;
  }
  const double estimate = winning_corrected_prefix_rows_[stage];
  if (!std::isfinite(estimate) || estimate < 0.0) return false;
  *rows = estimate;
  return true;
}

bool SemanticPlanRefiner::ReplayWinningPlacement(JOIN *join) {
  if (predicates_.empty()) return false;

  if (!has_winning_placement_) {
    const SemanticPlanEvaluation fallback =
        EvaluateCandidate(*join, join->best_positions, join->tables);
    if (!fallback.feasible) return true;
    RememberWinningPlacement(fallback);
  }
  if (!has_winning_placement_ ||
      winning_placement_.size() != predicates_.size()) {
    return true;
  }

  for (size_t i = 0; i < predicates_.size(); ++i) {
    const uint stage = winning_placement_[i];
    if (stage >= join->primary_tables || join->best_ref[stage] == nullptr)
      return true;

    Item *const predicate = predicates_[i].wrapper;
    Item *const existing = join->best_ref[stage]->condition();
    if (existing == nullptr) {
      join->best_ref[stage]->set_condition(predicate);
      continue;
    }

    Item_cond_and *combined =
        new (thd_->mem_root) Item_cond_and(existing, predicate);
    if (combined == nullptr) return true;
    combined->quick_fix_field();
    combined->update_used_tables();
    join->best_ref[stage]->set_condition(combined);
  }
  return false;
}

}  // namespace vipersql
