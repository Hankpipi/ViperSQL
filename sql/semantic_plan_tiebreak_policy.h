/* Copyright (c) 2026, ViperSQL contributors.

   Pure policy helpers for robust semantic-plan tie-breaking.
*/

#ifndef SQL_SEMANTIC_PLAN_TIEBREAK_POLICY_H_
#define SQL_SEMANTIC_PLAN_TIEBREAK_POLICY_H_

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace vipersql {

/** Evidence that is safe to compare only inside a tight refined-cost tie. */
struct SemanticFanoutRiskEvidence {
  const std::vector<double> &estimated_input_rows_by_group;
  const std::vector<uint64_t> &estimated_batch_counts_by_group;
  // The number of ordered logical predicates fused into each group. This
  // makes different fusion boundaries incomparable even if aggregate helper
  // work happens to match.
  const std::vector<size_t> &predicate_counts_by_group;
  // Logical identity of the first non-const table. Root-local cardinality
  // uncertainty is comparable only when it is common to both plans.
  uint64_t first_nonconst_table;
  unsigned int uncertain_fanout_tables;
  // False means the ordinary-equality fanout model could not classify at
  // least one relevant prefix table.
  bool evidence_valid{true};
};

enum class SemanticFanoutUniqueEvidence {
  kNone,
  kSingleColumnUnique,
  kCompositeUniqueUnknown,
};

struct SemanticPrefixTableFanoutAssessment {
  unsigned int uncertain_fanout_tables{0};
  bool evidence_valid{true};
};

enum class SemanticFanoutRiskPreference {
  kNoPreference,
  kCandidate,
  kIncumbent,
};

inline double SemanticRobustnessTieBand(double profiling_epsilon) {
  if (!std::isfinite(profiling_epsilon) || profiling_epsilon <= 0.0) {
    return 0.0;
  }
  return std::min(0.001, profiling_epsilon);
}

inline double SemanticRobustnessGlobalLimit(double globally_best_cost,
                                            double robustness_tie_band) {
  if (!std::isfinite(globally_best_cost) || globally_best_cost < 0.0 ||
      !std::isfinite(robustness_tie_band) || robustness_tie_band < 0.0) {
    return -1.0;
  }
  const double multiplier = 1.0 + std::min(0.001, robustness_tie_band);
  if (globally_best_cost > std::numeric_limits<double>::max() / multiplier) {
    return std::numeric_limits<double>::max();
  }
  return globally_best_cost * multiplier;
}

/** Prevent pairwise lower-risk choices from ratcheting away from best cost. */
inline bool WithinTightSemanticCostBand(double candidate_cost,
                                        double incumbent_cost,
                                        double globally_best_cost,
                                        double profiling_epsilon) {
  if (!std::isfinite(candidate_cost) || !std::isfinite(incumbent_cost) ||
      !std::isfinite(profiling_epsilon) || candidate_cost < 0.0 ||
      incumbent_cost < 0.0 || profiling_epsilon < 0.0) {
    return false;
  }
  const double tie_band = SemanticRobustnessTieBand(profiling_epsilon);
  const double global_limit =
      SemanticRobustnessGlobalLimit(globally_best_cost, tie_band);
  if (global_limit < 0.0 || candidate_cost > global_limit ||
      incumbent_cost > global_limit) {
    return false;
  }
  const double pairwise_limit =
      tie_band * std::max(1.0, std::min(candidate_cost, incumbent_cost));
  return std::abs(candidate_cost - incumbent_cost) <= pairwise_limit;
}

inline SemanticFanoutUniqueEvidence MergeSemanticFanoutUniqueEvidence(
    SemanticFanoutUniqueEvidence left, SemanticFanoutUniqueEvidence right) {
  if (left == SemanticFanoutUniqueEvidence::kSingleColumnUnique ||
      right == SemanticFanoutUniqueEvidence::kSingleColumnUnique) {
    return SemanticFanoutUniqueEvidence::kSingleColumnUnique;
  }
  if (left == SemanticFanoutUniqueEvidence::kCompositeUniqueUnknown ||
      right == SemanticFanoutUniqueEvidence::kCompositeUniqueUnknown) {
    return SemanticFanoutUniqueEvidence::kCompositeUniqueUnknown;
  }
  return SemanticFanoutUniqueEvidence::kNone;
}

/** Classify one table in a semantic group's relational input prefix. */
constexpr SemanticPrefixTableFanoutAssessment ClassifySemanticPrefixTableFanout(
    bool is_const_table, bool has_earlier_nonconst_prefix,
    bool required_by_group, bool ordinary_field_equality_connected,
    SemanticFanoutUniqueEvidence introduced_side_unique_evidence,
    bool has_only_trusted_reducing_filters) {
  // The first non-const table has no introduced-side fanout to classify.
  // A prompt-required table cannot be moved out of this semantic prefix.
  if (is_const_table || !has_earlier_nonconst_prefix || required_by_group) {
    return {};
  }

  // The heuristic proves bounds only for ordinary direct-field equality
  // (including MULT_EQUAL). Do not let <=>, expression/range predicates, or a
  // Cartesian transition collapse to a misleading zero-risk score.
  if (!ordinary_field_equality_connected) return {0, false};

  if (introduced_side_unique_evidence ==
      SemanticFanoutUniqueEvidence::kSingleColumnUnique) {
    return {};
  }
  // Partial or even apparently fully bound composite UNIQUE keys need a
  // complete-key binding proof, which this narrow classifier does not model.
  if (introduced_side_unique_evidence ==
      SemanticFanoutUniqueEvidence::kCompositeUniqueUnknown) {
    return {0, false};
  }
  // A trusted local reduction is useful only after structural equality
  // evidence has made the introduced-side fanout classifiable.
  if (has_only_trusted_reducing_filters) return {};
  return {1, true};
}

inline bool SameEstimatedSemanticInputs(double left, double right) {
  if (!std::isfinite(left) || !std::isfinite(right) || left < 0.0 ||
      right < 0.0) {
    return false;
  }
  constexpr double kRelativeInputEpsilon = 1.0e-9;
  const double scale = std::max({1.0, std::abs(left), std::abs(right)});
  return std::abs(left - right) <= kRelativeInputEpsilon * scale;
}

inline bool SameEstimatedSemanticWork(const SemanticFanoutRiskEvidence &left,
                                      const SemanticFanoutRiskEvidence &right) {
  const size_t group_count = left.estimated_input_rows_by_group.size();
  if (group_count == 0 ||
      right.estimated_input_rows_by_group.size() != group_count ||
      left.estimated_batch_counts_by_group.size() != group_count ||
      right.estimated_batch_counts_by_group.size() != group_count ||
      left.predicate_counts_by_group.size() != group_count ||
      right.predicate_counts_by_group.size() != group_count ||
      left.estimated_batch_counts_by_group !=
          right.estimated_batch_counts_by_group ||
      left.predicate_counts_by_group != right.predicate_counts_by_group) {
    return false;
  }
  for (size_t group = 0; group < group_count; ++group) {
    if (!SameEstimatedSemanticInputs(
            left.estimated_input_rows_by_group[group],
            right.estimated_input_rows_by_group[group])) {
      return false;
    }
  }
  return true;
}

/**
  Prefer lower uncertain fanout only when semantic-work evidence is equal.

  The caller supplies `within_tight_refined_cost_tie` so the broader profiling
  uncertainty band cannot override a materially cheaper plan.
*/
inline SemanticFanoutRiskPreference CompareSemanticFanoutRisk(
    bool within_tight_refined_cost_tie,
    const SemanticFanoutRiskEvidence &candidate,
    const SemanticFanoutRiskEvidence &incumbent) {
  if (!within_tight_refined_cost_tie || !candidate.evidence_valid ||
      !incumbent.evidence_valid ||
      candidate.first_nonconst_table != incumbent.first_nonconst_table ||
      !SameEstimatedSemanticWork(candidate, incumbent) ||
      candidate.uncertain_fanout_tables == incumbent.uncertain_fanout_tables) {
    return SemanticFanoutRiskPreference::kNoPreference;
  }
  return candidate.uncertain_fanout_tables < incumbent.uncertain_fanout_tables
             ? SemanticFanoutRiskPreference::kCandidate
             : SemanticFanoutRiskPreference::kIncumbent;
}

}  // namespace vipersql

#endif  // SQL_SEMANTIC_PLAN_TIEBREAK_POLICY_H_
