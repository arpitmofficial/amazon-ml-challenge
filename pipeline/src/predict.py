"""
Stage 4: Predict on test candidates and produce matching_results.tsv.

Steps:
  1. Load test candidate_pairs.tsv, compute the same features as training.
  2. Score with the trained model.
  3. Keep only pairs with probability >= saved threshold.
  4. One-to-one consistency pass: a single S2/S3 record should not be confidently claimed by
     multiple S1 entities. Resolve greedily by descending probability (an S1 entity CAN still
     match multiple S2/S3 records -- only the candidate side is de-duplicated).
  5. Ensure every S1 test entity has exactly one output row (empty list if no match survives).
"""

import argparse
import json
import joblib
import pandas as pd
from collections import defaultdict

from utils import read_source_file, read_id_list_tsv, write_id_list_tsv
from features import build_record_indices, compute_features, FEATURE_COLUMNS
from train import load_candidate_pairs_long  # reuse the same TSV->long-format loader


def greedy_one_to_one_on_candidate_side(scored_df: pd.DataFrame) -> pd.DataFrame:
    """
    scored_df has columns [source1_entity_id, candidate_entity_id, prob], already filtered
    to >= threshold. Sort by prob desc; assign each candidate_entity_id to at most one
    source1_entity_id (the highest-confidence claim). source1 entities keep all their
    surviving, still-available candidates.
    """
    scored_df = scored_df.sort_values("prob", ascending=False)
    used_candidates = set()
    kept_rows = []
    for row in scored_df.itertuples(index=False):
        if row.candidate_entity_id in used_candidates:
            continue
        used_candidates.add(row.candidate_entity_id)
        kept_rows.append(row)
    return pd.DataFrame(kept_rows, columns=scored_df.columns)


def main(args):
    print("Loading test candidate pairs...")
    pairs_df = load_candidate_pairs_long(args.candidates)

    print("Building feature indices...")
    s1_index, cand_index = build_record_indices(args.source1, args.source2, args.source3)

    print("Computing features...")
    feat_df = compute_features(pairs_df, s1_index, cand_index)

    print("Loading model and threshold...")
    model = joblib.load(args.model)
    with open(args.threshold) as f:
        threshold = json.load(f)["threshold"]
    print(f"Using threshold = {threshold}")

    probs = model.predict_proba(feat_df[FEATURE_COLUMNS])[:, 1]
    feat_df["prob"] = probs

    above = feat_df[feat_df["prob"] >= threshold][
        ["source1_entity_id", "candidate_entity_id", "prob"]
    ]
    print(f"{len(above)} / {len(feat_df)} pairs survive threshold")

    resolved = greedy_one_to_one_on_candidate_side(above)
    print(f"{len(resolved)} pairs survive one-to-one consistency resolution")

    result_map = defaultdict(set)
    for row in resolved.itertuples(index=False):
        result_map[row.source1_entity_id].add(row.candidate_entity_id)

    # Ensure every S1 test entity has a row, even if empty (singleton prediction)
    all_s1_ids = read_source_file(args.source1)["entity_id"].tolist()
    for s1_id in all_s1_ids:
        result_map.setdefault(s1_id, set())

    write_id_list_tsv(result_map, args.output, "source1_entity_id", "matched_entity_ids")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source1", required=True)
    parser.add_argument("--source2", required=True)
    parser.add_argument("--source3", required=True)
    parser.add_argument("--candidates", required=True, help="candidate_pairs.tsv for TEST split")
    parser.add_argument("--model", default="model.joblib")
    parser.add_argument("--threshold", default="threshold.json")
    parser.add_argument("--output", default="output/matching_results.tsv")
    args = parser.parse_args()
    main(args)
