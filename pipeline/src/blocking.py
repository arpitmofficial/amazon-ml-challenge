"""
Stage 1: Candidate generation (blocking).

Strategy (high recall, then capped for precision-friendly downstream sizing):
  1. TF-IDF char n-gram (2-4) cosine similarity, top-K nearest neighbors per S1 entity,
     computed via chunked sparse matrix multiplication (fast, no external ANN library needed
     at this dataset scale; swap in FAISS if the dataset is very large).
  2. Token-overlap inverted index blocking (catches heavy word-reordering / partial name matches
     that char n-grams sometimes under-rank).
  3. Union of (1) and (2), deduped, capped at MAX_CANDIDATES per S1 entity, ranked by TF-IDF score
     (token-only hits get a lower synthetic score so they don't crowd out strong matches).

Country is used as a SOFT feature later, not a hard filter here -- cross-source country label
noise would otherwise silently cap recall.

Run standalone to generate candidate_pairs.tsv for train (to measure recall ceiling) or test.
"""

import argparse
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from utils import (
    read_source_file,
    normalize_name,
    normalize_address,
    name_tokens,
    write_id_list_tsv,
)

TOP_K = 20                 # neighbors to pull from TF-IDF search
MAX_CANDIDATES = 30        # final cap per S1 entity after union
MIN_TFIDF_SCORE = 0.03     # floor to strip pure-noise matches
CHUNK_SIZE = 500           # rows of S1 processed per matmul chunk (memory control)
MAX_VOCAB = 100_000        # cap TF-IDF vocabulary size to bound memory on low-RAM machines


def build_combined_text(df: pd.DataFrame) -> pd.Series:
    name = df["business_name"].map(normalize_name)
    addr = df["business_address"].map(normalize_address)
    return name + " " + addr


def tfidf_topk_candidates(s1_df, s2_df, s3_df, vectorizer):
    """
    Return dict: s1_entity_id -> list[(candidate_id, score)] via TF-IDF cosine top-K.

    MEMORY-SAFE VERSION: the similarity matrix is kept SPARSE the entire time.
    Never call .toarray() on a chunk -- for large datasets that densifies a
    (chunk_size x total_other_records) matrix, which can be many GB even for a
    modest chunk size. TF-IDF vectors barely overlap for most pairs, so the
    sparse product is tiny by comparison; we extract top-k straight from each
    sparse row's own nonzero entries.
    """
    other_df = pd.concat([s2_df, s3_df], ignore_index=True)
    other_text = build_combined_text(other_df)
    s1_text = build_combined_text(s1_df)

    other_matrix = vectorizer.transform(other_text).tocsr()   # (M, V), already L2-normalized
    s1_matrix = vectorizer.transform(s1_text).tocsr()         # (N, V)
    other_matrix_T = other_matrix.T.tocsr()                   # (V, M), built once

    other_ids = other_df["entity_id"].values
    results = defaultdict(list)

    n_rows = s1_matrix.shape[0]
    for start in range(0, n_rows, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, n_rows)
        chunk = s1_matrix[start:end]                          # (c, V) sparse
        sims = chunk.dot(other_matrix_T).tocsr()               # (c, M) SPARSE -- no .toarray()
        for i in range(sims.shape[0]):
            row = sims.getrow(i)
            if row.nnz == 0:
                continue
            data = row.data
            indices = row.indices
            if data.max() <= MIN_TFIDF_SCORE:
                continue
            k = min(TOP_K, len(data))
            top_local = np.argpartition(-data, k - 1)[:k]
            top_local = top_local[np.argsort(-data[top_local])]
            s1_id = s1_df.iloc[start + i]["entity_id"]
            for local_idx in top_local:
                score = data[local_idx]
                if score <= MIN_TFIDF_SCORE:
                    continue
                cand_idx = indices[local_idx]
                results[s1_id].append((other_ids[cand_idx], float(score)))
    return results


def token_overlap_candidates(s1_df, s2_df, s3_df):
    """Return dict: s1_entity_id -> list[(candidate_id, synthetic_score)] via inverted index."""
    other_df = pd.concat([s2_df, s3_df], ignore_index=True)
    other_norm_names = other_df["business_name"].map(normalize_name)
    other_tokens = other_norm_names.map(name_tokens)
    other_ids = other_df["entity_id"].values

    inverted = defaultdict(list)
    for idx, toks in enumerate(other_tokens):
        for tok in toks:
            if len(tok) < 3:   # skip very short/common tokens
                continue
            inverted[tok].append(idx)

    s1_norm_names = s1_df["business_name"].map(normalize_name)
    s1_tokens = s1_norm_names.map(name_tokens)

    results = defaultdict(list)
    for s1_idx, toks in enumerate(s1_tokens):
        s1_id = s1_df.iloc[s1_idx]["entity_id"]
        counts = defaultdict(int)
        for tok in toks:
            if len(tok) < 3:
                continue
            for other_idx in inverted.get(tok, []):
                counts[other_idx] += 1
        if not counts:
            continue
        # synthetic score in [0, ~0.3] range so it ranks below strong TF-IDF hits by default
        max_possible = max(len(toks), 1)
        for other_idx, cnt in counts.items():
            synthetic_score = 0.3 * (cnt / max_possible)
            results[s1_id].append((other_ids[other_idx], synthetic_score))
    return results


def merge_candidates(tfidf_map, token_map, max_candidates=MAX_CANDIDATES):
    merged = {}
    all_ids = set(tfidf_map.keys()) | set(token_map.keys())
    for s1_id in all_ids:
        best = {}
        for cand_id, score in tfidf_map.get(s1_id, []):
            best[cand_id] = max(best.get(cand_id, 0.0), score)
        for cand_id, score in token_map.get(s1_id, []):
            best[cand_id] = max(best.get(cand_id, 0.0), score)
        ranked = sorted(best.items(), key=lambda x: -x[1])[:max_candidates]
        merged[s1_id] = [cand_id for cand_id, _ in ranked]
    return merged


def run_blocking(source1_path, source2_path, source3_path, output_path):
    s1_df = read_source_file(source1_path)
    s2_df = read_source_file(source2_path)
    s3_df = read_source_file(source3_path)

    all_text = pd.concat([
        build_combined_text(s1_df),
        build_combined_text(s2_df),
        build_combined_text(s3_df),
    ], ignore_index=True)

    vectorizer = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(2, 4), min_df=2,
        max_features=MAX_VOCAB, dtype=np.float32,
    )
    vectorizer.fit(all_text)

    print("Running TF-IDF nearest-neighbor blocking...")
    tfidf_map = tfidf_topk_candidates(s1_df, s2_df, s3_df, vectorizer)

    print("Running token-overlap blocking...")
    token_map = token_overlap_candidates(s1_df, s2_df, s3_df)

    print("Merging candidate sets...")
    merged = merge_candidates(tfidf_map, token_map)

    # Ensure every S1 entity has a row, even with zero candidates found
    for s1_id in s1_df["entity_id"]:
        merged.setdefault(s1_id, [])

    write_id_list_tsv(merged, output_path, "source1_entity_id", "candidate_entity_ids")
    print(f"Wrote {output_path} ({len(merged)} S1 entities)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source1", required=True)
    parser.add_argument("--source2", required=True)
    parser.add_argument("--source3", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run_blocking(args.source1, args.source2, args.source3, args.output)
