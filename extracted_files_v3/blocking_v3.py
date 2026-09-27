"""
Stage 1 (v3 -- scalable to millions of records): Candidate generation (blocking).

Combines two good ideas that were previously split across two disconnected scripts:
  1. Country partitioning BEFORE vectorizing -- this is the single biggest lever for
     scaling to 12M+ records. Instead of one N x M problem over the whole dataset,
     you get several much smaller N_c x M_c problems, one per country.
  2. sparse_dot_topn (sp_matmul_topn) for the actual top-K sparse cosine similarity --
     a purpose-built, battle-tested library for exactly this ("fuzzy match millions of
     strings") problem. Much faster and more memory-disciplined than a manual chunked
     scipy sparse matmul loop.

IMPORTANT: unlike the earlier submission-only script, this module:
  - Outputs the standard candidate_pairs.tsv schema (source1_entity_id, candidate_entity_ids)
    so it plugs directly into features.py / train.py / predict.py / check_recall_ceiling.py
    with no changes needed anywhere else.
  - Does NOT make the final match decision here. Blocking only produces CANDIDATES.
    The trained LightGBM model + F_0.5-tuned threshold (train.py / predict.py) still makes
    the actual match/no-match call. Do not thresholds cosine similarity directly as your
    final answer -- that throws away everything the model/threshold-sweep step buys you.
  - Adds a cross-country SAFETY NET: entities that get zero candidates within their own
    country partition (likely due to noisy/inconsistent country labels across sources)
    get a second pass against the FULL S2+S3 pool. This set is normally small, so the
    extra cost is bounded even though it isn't country-partitioned.

Run on TRAIN sources first (to measure recall ceiling with check_recall_ceiling.py),
then on TEST sources for the real submission.
"""

import argparse
import gc
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn

from utils import (
    read_source_file,
    normalize_name,
    normalize_address,
    write_id_list_tsv,
)

CANDIDATE_K = 20            # top-N candidates to keep per S1 entity, per pass
CANDIDATE_THRESHOLD = 0.10  # floor cosine score to even consider a pair (kept low; the
                             # model, not this threshold, decides the real match/no-match)
MAX_CANDIDATES = 30         # final cap per S1 entity after combining both passes


def build_combined_text(df: pd.DataFrame) -> pd.Series:
    name = df["business_name"].map(normalize_name)
    addr = df["business_address"].map(normalize_address)
    return name + " " + addr


def topn_within_group(s1_sub: pd.DataFrame, other_sub: pd.DataFrame) -> dict:
    """
    Run TF-IDF + sparse_dot_topn top-K cosine similarity between s1_sub and other_sub.
    Returns dict: s1_entity_id -> list[(candidate_id, score)]
    """
    results = {}
    if len(s1_sub) == 0:
        return results
    if len(other_sub) == 0:
        for eid in s1_sub["entity_id"]:
            results[eid] = []
        return results

    s1_text = build_combined_text(s1_sub)
    other_text = build_combined_text(other_sub)

    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2, dtype=np.float32)
    vectorizer.fit(pd.concat([s1_text, other_text], ignore_index=True))

    query_matrix = vectorizer.transform(s1_text).tocsr()
    corpus_matrix = vectorizer.transform(other_text).tocsr()

    sim = sp_matmul_topn(
        query_matrix, corpus_matrix.T,
        top_n=CANDIDATE_K, threshold=CANDIDATE_THRESHOLD,
    )  # (N, M) sparse, at most CANDIDATE_K nonzero entries per row

    q_ids = s1_sub["entity_id"].values
    c_ids = other_sub["entity_id"].values
    indptr, indices, data = sim.indptr, sim.indices, sim.data

    for i in range(len(q_ids)):
        start, end = indptr[i], indptr[i + 1]
        if start == end:
            results[q_ids[i]] = []
            continue
        row_idx = indices[start:end]
        row_score = data[start:end]
        order = np.argsort(-row_score)
        results[q_ids[i]] = [(c_ids[row_idx[j]], float(row_score[j])) for j in order]

    del vectorizer, query_matrix, corpus_matrix, sim
    gc.collect()
    return results


def run_blocking(source1_path, source2_path, source3_path, output_path):
    print("Loading sources...")
    s1_df = read_source_file(source1_path)
    s2_df = read_source_file(source2_path)
    s3_df = read_source_file(source3_path)
    other_df = pd.concat([s2_df, s3_df], ignore_index=True)

    s1_df["country_norm"] = s1_df["country"].astype(str).str.strip().str.lower()
    other_df["country_norm"] = other_df["country"].astype(str).str.strip().str.lower()

    countries = s1_df["country_norm"].unique()
    print(f"Countries found in S1: {list(countries)}")

    all_results = {}
    for country in countries:
        s1_sub = s1_df[s1_df["country_norm"] == country]
        other_sub = other_df[other_df["country_norm"] == country]
        print(f"  Country '{country}': {len(s1_sub)} S1 rows, {len(other_sub)} S2+S3 rows")
        country_results = topn_within_group(s1_sub, other_sub)
        all_results.update(country_results)

    # Safety net: entities with zero candidates from their own country partition
    # (likely noisy/inconsistent country labels) get one more pass against the FULL pool.
    zero_candidate_ids = [eid for eid, cands in all_results.items() if not cands]
    if zero_candidate_ids:
        print(f"Safety net: {len(zero_candidate_ids)} entities had zero in-country candidates; "
              f"re-checking against the full S2+S3 pool...")
        s1_leftover = s1_df[s1_df["entity_id"].isin(zero_candidate_ids)]
        fallback_results = topn_within_group(s1_leftover, other_df)
        all_results.update(fallback_results)

    # Cap final candidate count and drop the score, write in standard schema
    final_map = {}
    for eid, cands in all_results.items():
        ranked = sorted(cands, key=lambda x: -x[1])[:MAX_CANDIDATES]
        final_map[eid] = [c for c, _ in ranked]

    write_id_list_tsv(final_map, output_path, "source1_entity_id", "candidate_entity_ids")
    print(f"Wrote {output_path} ({len(final_map)} S1 entities)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source1", required=True)
    parser.add_argument("--source2", required=True)
    parser.add_argument("--source3", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run_blocking(args.source1, args.source2, args.source3, args.output)
