"""
Stage 3: Train the pairwise matcher and pick the F_0.5-optimal decision threshold.

Pipeline:
  1. Load candidate_pairs.tsv (train) -- these are the *hard negatives* + true positives.
  2. Load train_ground_truth.tsv, label every candidate pair 1/0.
  3. Split by SOURCE1 ENTITY ID (not by row) into train/val, so no entity leaks across the split.
  4. Compute features, train a LightGBM binary classifier with scale_pos_weight for imbalance.
  5. Sweep probability thresholds 0.05-0.95, pick the one that maximizes macro F_0.5 on the
     validation split -- using the REAL competition metric (utils.macro_f_beta), not accuracy/AUC.
  6. Save model + chosen threshold to disk for predict.py.
"""

import argparse
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib
from collections import defaultdict

from utils import read_id_list_tsv, macro_f_beta
from features import build_record_indices, compute_features, FEATURE_COLUMNS


def load_candidate_pairs_long(candidate_path: str) -> pd.DataFrame:
    """candidate_pairs.tsv -> long dataframe [source1_entity_id, candidate_entity_id]."""
    cand_map = read_id_list_tsv(candidate_path, "source1_entity_id", "candidate_entity_ids")
    rows = []
    for s1_id, cands in cand_map.items():
        for c in cands:
            rows.append((s1_id, c))
    return pd.DataFrame(rows, columns=["source1_entity_id", "candidate_entity_id"])


def label_pairs(pairs_df: pd.DataFrame, ground_truth_map: dict) -> pd.DataFrame:
    labels = []
    for row in pairs_df.itertuples(index=False):
        true_set = ground_truth_map.get(row.source1_entity_id, frozenset())
        labels.append(1 if row.candidate_entity_id in true_set else 0)
    out = pairs_df.copy()
    out["label"] = labels
    return out


def split_by_entity(entity_ids, val_frac=0.2, seed=42):
    rng = np.random.default_rng(seed)
    unique_ids = np.array(sorted(set(entity_ids)))
    rng.shuffle(unique_ids)
    n_val = int(len(unique_ids) * val_frac)
    val_ids = set(unique_ids[:n_val])
    train_ids = set(unique_ids[n_val:])
    return train_ids, val_ids


def sweep_threshold(val_df, val_probs, ground_truth_map, all_val_entity_ids):
    """
    Try thresholds 0.05..0.95, build predicted match sets, score with the exact
    competition macro F_0.5 (including singletons with empty predictions).
    """
    best_thresh, best_score = 0.5, -1.0
    true_map = {eid: ground_truth_map.get(eid, frozenset()) for eid in all_val_entity_ids}

    for thresh in np.arange(0.05, 0.96, 0.01):
        pred_map = defaultdict(set)
        mask = val_probs >= thresh
        for s1_id, cand_id in zip(val_df["source1_entity_id"].values[mask],
                                   val_df["candidate_entity_id"].values[mask]):
            pred_map[s1_id].add(cand_id)
        score = macro_f_beta(true_map, pred_map, beta=0.5)
        if score > best_score:
            best_score, best_thresh = score, thresh

    return best_thresh, best_score


def main(args):
    print("Loading candidate pairs and ground truth...")
    pairs_df = load_candidate_pairs_long(args.candidates)
    ground_truth_map = read_id_list_tsv(args.ground_truth, "source1_entity_id", "matched_entity_ids")

    print("Labeling pairs...")
    labeled_df = label_pairs(pairs_df, ground_truth_map)
    print(f"  {len(labeled_df)} pairs, {labeled_df['label'].sum()} positive")

    print("Building feature indices...")
    s1_index, cand_index = build_record_indices(args.source1, args.source2, args.source3)

    print("Computing features...")
    feat_df = compute_features(labeled_df[["source1_entity_id", "candidate_entity_id"]], s1_index, cand_index)
    feat_df["label"] = labeled_df["label"].values

    all_entity_ids = list(s1_index.keys())
    train_ids, val_ids = split_by_entity(all_entity_ids, val_frac=args.val_frac)

    train_mask = feat_df["source1_entity_id"].isin(train_ids)
    val_mask = feat_df["source1_entity_id"].isin(val_ids)

    X_train = feat_df.loc[train_mask, FEATURE_COLUMNS]
    y_train = feat_df.loc[train_mask, "label"]
    X_val = feat_df.loc[val_mask, FEATURE_COLUMNS]

    pos = max(y_train.sum(), 1)
    neg = max(len(y_train) - y_train.sum(), 1)
    scale_pos_weight = neg / pos
    print(f"Train pairs: {len(X_train)} (pos={pos}, neg={neg}, scale_pos_weight={scale_pos_weight:.2f})")

    model = lgb.LGBMClassifier(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=20,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
    )
    model.fit(X_train, y_train)

    print("Scoring validation set with real F_0.5 threshold sweep...")
    val_probs = model.predict_proba(X_val)[:, 1]
    val_df = feat_df.loc[val_mask, ["source1_entity_id", "candidate_entity_id"]].reset_index(drop=True)
    val_probs = np.asarray(val_probs)

    best_thresh, best_score = sweep_threshold(val_df, val_probs, ground_truth_map, val_ids)
    print(f"Best threshold: {best_thresh:.2f}  ->  validation macro F_0.5 = {best_score:.4f}")

    joblib.dump(model, args.model_out)
    with open(args.threshold_out, "w") as f:
        json.dump({"threshold": float(best_thresh), "val_f0_5": float(best_score)}, f, indent=2)
    print(f"Saved model to {args.model_out}, threshold to {args.threshold_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source1", required=True)
    parser.add_argument("--source2", required=True)
    parser.add_argument("--source3", required=True)
    parser.add_argument("--candidates", required=True, help="candidate_pairs.tsv for TRAIN split")
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--model-out", default="model.joblib")
    parser.add_argument("--threshold-out", default="threshold.json")
    args = parser.parse_args()
    main(args)
