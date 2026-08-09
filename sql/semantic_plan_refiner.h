/* Copyright (c) 2026, ViperSQL contributors.

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License, version 2.0.
*/

#ifndef SQL_SEMANTIC_PLAN_REFINER_H_
#define SQL_SEMANTIC_PLAN_REFINER_H_

#include <cstddef>
#include <cstdint>
#include <vector>

#include "my_inttypes.h"
#include "my_table_map.h"

class Item;
class JOIN;
struct POSITION;
class THD;

namespace vipersql {

/** Cost and exact semantic-filter placement for one complete join order. */
struct SemanticPlanEvaluation {
  bool applicable{false};
  bool feasible{true};
  double total_cost{0.0};
  double output_rows{0.0};
  uint64_t estimated_batch_count{0};
  uint buffered_build_semantic_groups{0};
  // Number of optional pre-semantic tables whose ordinary equality join can
  // fan out, whose introduced join field has no single-column unique proof,
  // and whose reducing local conjuncts are not all backed by trusted
  // estimates.
  uint uncertain_fanout_tables{0};
  // False if any relevant prefix table falls outside the narrow ordinary
  // direct-field equality/single-column-unique fanout model.
  bool fanout_risk_evidence_valid{true};
  // First non-const table in this join order; different roots make root-local
  // cardinality uncertainty incomparable.
  table_map fanout_risk_root_table{0};
  double intermediate_cardinality_score{0.0};
  // Relational cardinality after each table in this candidate, before any
  // semantic selectivity is applied.  The winning vector is also used when
  // constructing the classic optimizer's AccessPath tree so EXPLAIN reports
  // the same estimates that drove semantic plan selection.
  std::vector<double> corrected_prefix_rows;
  std::vector<table_map> semantic_group_input_tables;
  std::vector<double> semantic_group_estimated_input_rows;
  std::vector<uint64_t> semantic_group_estimated_batch_counts;
  std::vector<size_t> semantic_group_predicate_counts;
  std::vector<uint> placement_stages;
};

/**
  Refines classic-optimizer join orders with semantic predicate migration.

  The refiner owns no Item objects. It temporarily removes complete semantic
  predicate conjuncts from JOIN::where_cond, costs every complete relational
  plan, remembers the exact placement of the winner, and finally attaches the
  original predicate wrappers to the chosen JOIN_TABs.
*/
class SemanticPlanRefiner {
 public:
  explicit SemanticPlanRefiner(THD *thd) : thd_(thd) {}

  /** Extract movable top-level semantic conjuncts, preserving their order. */
  Item *ExtractSemanticConjuncts(Item *condition);

  bool empty() const { return predicates_.empty(); }

  /** Refine the cost and placement of a complete classic-optimizer plan. */
  SemanticPlanEvaluation EvaluateCandidate(const JOIN &join,
                                           const POSITION *positions,
                                           uint plan_count) const;

  /** Save the placement belonging to the plan copied to best_positions. */
  void RememberWinningPlacement(const SemanticPlanEvaluation &evaluation);

  /** Return the winning relational estimate for one classic-plan stage. */
  bool WinningCorrectedPrefixRows(uint stage, double *rows) const;

  /** Attach every original predicate wrapper at its saved winning stage. */
  bool ReplayWinningPlacement(JOIN *join);

 private:
  struct SemanticPredicate {
    Item *wrapper{nullptr};
    table_map required_tables{0};
  };

  THD *const thd_;
  std::vector<SemanticPredicate> predicates_;
  std::vector<uint> winning_placement_;
  std::vector<double> winning_corrected_prefix_rows_;
  bool has_winning_placement_{false};
};

}  // namespace vipersql

#endif  // SQL_SEMANTIC_PLAN_REFINER_H_
