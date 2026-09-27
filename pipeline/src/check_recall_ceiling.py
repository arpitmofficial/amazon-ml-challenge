"""
Run this FIRST, before training anything.

Computes the blocking recall ceiling on the train split: what fraction of true matches
in train_ground_truth.tsv actually survive into candidate_pairs.tsv (train)?

This number upper-bounds everything downstream. If it's below ~0.95-0.97, no amount of
classifier tuning will get you to a high F_0.5 -- go back to blocking.py and raise TOP_K /
lower MIN_TFIDF_SCORE / add more blocking keys instead.
"""

import argparse
from utils import read_id_list_tsv


def main(args):
    ground_truth = read_id_list_tsv(args.ground_truth, "source1_entity_id", "matched_entity_ids")
    candidates = read_id_list_tsv(args.candidates, "source1_entity_id", "candidate_entity_ids")

    total_true_matches = 0
    recovered_matches = 0
    entities_with_matches = 0
    entities_fully_recovered = 0
    total_candidates = 0

    for s1_id, true_set in ground_truth.items():
        cand_set = candidates.get(s1_id, frozenset())
        total_candidates += len(cand_set)
        if not true_set:
            continue
        entities_with_matches += 1
        total_true_matches += len(true_set)
        found = len(true_set & cand_set)
        recovered_matches += found
        if found == len(true_set):
            entities_fully_recovered += 1

    recall_ceiling = recovered_matches / total_true_matches if total_true_matches else 1.0
    entity_recall = entities_fully_recovered / entities_with_matches if entities_with_matches else 1.0
    avg_candidates_per_entity = total_candidates / len(ground_truth) if ground_truth else 0

    print("=== Blocking Recall Ceiling Report ===")
    print(f"S1 entities with >=1 true match: {entities_with_matches}")
    print(f"Pair-level recall ceiling:       {recall_ceiling:.4f}  ({recovered_matches}/{total_true_matches})")
    print(f"Entity-level full-recovery rate: {entity_recall:.4f}  ({entities_fully_recovered}/{entities_with_matches})")
    print(f"Avg candidates per S1 entity:    {avg_candidates_per_entity:.1f}")
    print()
    if recall_ceiling < 0.95:
        print("WARNING: recall ceiling below 0.95 -- fix blocking.py before training a model.")
        print("  Try: raise TOP_K, lower MIN_TFIDF_SCORE, or inspect missed entities directly.")
    else:
        print("Recall ceiling looks healthy. Safe to proceed to feature engineering / training.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--candidates", required=True, help="candidate_pairs.tsv computed on TRAIN sources")
    args = parser.parse_args()
    main(args)
