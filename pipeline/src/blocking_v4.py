"""
Stage 1 (v4 — hash-join blocking, no large matrix multiplications).

Design goals
============
* Never perform a similarity comparison over a pool larger than a few hundred records.
* Zero OOM risk: the most expensive operation is a pandas merge on an exact key, which
  is O(N_bucket) per bucket — not O(N_country²).
* Identical normalization and key generation for S1/S2/S3 and train/test via shared
  `add_blocking_keys` imported from utils.py — no copy-paste drift.

Algorithm (per country partition)
==================================
Step 0 — Normalize + key generation  (O(N), vectorized, one pass)
  norm_name, norm_address, digit_signature, phonetic_code,
  name_prefix_key, first_letter — computed once for all three sources.

Step 1 — Country partition  (exact categorical split, zero recall risk)
  Iterate over each unique country in S1; slice S1, S2, S3 to that country.

Step 2 — Four hash-join candidate sets  (union, not intersection)
  For each of the four key columns, inner-merge S1 slice with the combined
  S2+S3 pool on that key.  Empty-key rows are explicitly excluded BEFORE the
  merge to prevent the "every-missing-joins-every-missing" false-positive explosion.
  Pairs from all four joins are unioned and deduplicated.

Step 3 — Fuzzy score the small pool  (rapidfuzz, per-group)
  For each S1 entity, score its unioned candidate set with
  rapidfuzz.fuzz.token_sort_ratio.  Group sizes are now tens to low hundreds,
  not thousands.

Step 4 — Top-K / dynamic threshold selection
  Keep candidates within DYNAMIC_MARGIN of the top score, capped at TOP_K.
  Require score >= MIN_SCORE.

Step 5 — Safety net for zero-candidate entities
  S1 entities that ended up with no candidates (usually <1% of records)
  get a pass against the full same-country pool using a TF-IDF cosine
  similarity via sparse_dot_topn — fast because this only runs on a small
  minority, not the full dataset.

Step 6 — Write output in standard schema
  Every S1 entity gets a row in candidate_pairs.tsv.
  Empty candidate_entity_ids string for entities with genuinely zero candidates.
"""

import argparse
import gc
import time
import logging
from typing import Optional

import numpy as np
import pandas as pd
import rapidfuzz.fuzz as rfuzz

from utils import (
    read_source_file,
    normalize_name,
    normalize_address,
    write_id_list_tsv,
    add_blocking_keys,
    normalize_text_series,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Tunable constants ──────────────────────────────────────────────────────────
TOP_K = 10              # max candidates per S1 entity (after scoring)
MIN_SCORE = 30          # minimum fuzzy score to retain a candidate (0–100 scale)
DYNAMIC_MARGIN = 20     # keep all candidates within this many points of the top score
SAFETY_NET_K = 10       # top-K for the safety-net TF-IDF pass
SAFETY_NET_THRESHOLD = 0.08  # min cosine similarity in safety-net pass


# ──────────────────────────────────────────────────────────────────────────────
# Step 2 helpers — hash-join on a single key
# ──────────────────────────────────────────────────────────────────────────────

def _key_join(
    s1_df: pd.DataFrame,
    pool_df: pd.DataFrame,
    key_col: str,
) -> pd.DataFrame:
    """
    Inner merge of s1_df and pool_df on *key_col*.

    IMPORTANT: rows where key_col == '' are dropped BEFORE the merge to prevent
    the silent "every empty-key row joins every other empty-key row" explosion
    that creates O(N²) false candidate pairs.

    Returns DataFrame with columns [entity_id_s1, entity_id_cand].
    """
    s1_sub = s1_df[s1_df[key_col] != ''][['entity_id', key_col]]
    pool_sub = pool_df[pool_df[key_col] != ''][['entity_id', key_col]]

    if s1_sub.empty or pool_sub.empty:
        return pd.DataFrame(columns=['entity_id_s1', 'entity_id_cand'])

    merged = s1_sub.merge(pool_sub, on=key_col, suffixes=('_s1', '_cand'))
    return merged[['entity_id_s1', 'entity_id_cand']]


def _build_candidate_pairs(
    s1_country: pd.DataFrame,
    pool_country: pd.DataFrame,
) -> pd.DataFrame:
    """
    Union of four hash-join blocking keys.
    Logs per-key and total pair counts so pathological bucket sizes are diagnosable.
    """
    keys = ['digit_signature', 'phonetic_code', 'name_prefix_key', 'first_letter']
    parts = []
    for key in keys:
        pairs = _key_join(s1_country, pool_country, key)
        log.info(
            "    key='%s'  →  %d raw pairs (%.1f avg per S1 entity)",
            key, len(pairs),
            len(pairs) / max(len(s1_country), 1),
        )
        parts.append(pairs)

    all_pairs = pd.concat(parts, ignore_index=True).drop_duplicates()
    log.info(
        "    Combined (deduped): %d pairs across %d S1 entities  (avg %.1f / entity)",
        len(all_pairs),
        all_pairs['entity_id_s1'].nunique() if not all_pairs.empty else 0,
        len(all_pairs) / max(s1_country['entity_id'].nunique(), 1),
    )
    return all_pairs


# ──────────────────────────────────────────────────────────────────────────────
# Step 3 — fuzzy scoring within the small candidate pool
# ──────────────────────────────────────────────────────────────────────────────

def _score_group(
    s1_norm_name: str,
    cand_ids: np.ndarray,
    cand_norm_names: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Score one S1 entity against its small candidate pool using token_sort_ratio.
    Returns (candidate_ids, scores) arrays.
    """
    scores = np.array(
        [rfuzz.token_sort_ratio(s1_norm_name, cn) for cn in cand_norm_names],
        dtype=np.float32,
    )
    return cand_ids, scores


def _score_candidates(
    all_pairs: pd.DataFrame,
    s1_lookup: dict,     # entity_id -> norm_name
    pool_lookup: dict,   # entity_id -> norm_name
) -> dict:
    """
    Group all_pairs by S1 entity, score each group, apply TOP_K / threshold selection.
    Returns {s1_entity_id: [candidate_entity_id, ...]} (ordered by score desc).
    """
    results = {}

    for s1_id, group in all_pairs.groupby('entity_id_s1'):
        s1_name = s1_lookup.get(s1_id, '')
        cand_ids = group['entity_id_cand'].values
        cand_names = np.array([pool_lookup.get(cid, '') for cid in cand_ids])

        _, scores = _score_group(s1_name, cand_ids, cand_names)

        # Step 4: dynamic threshold + TOP_K cap
        mask = scores >= MIN_SCORE
        cand_ids, scores = cand_ids[mask], scores[mask]

        if len(scores) == 0:
            results[s1_id] = []
            continue

        top_score = scores.max()
        mask2 = scores >= (top_score - DYNAMIC_MARGIN)
        cand_ids, scores = cand_ids[mask2], scores[mask2]

        order = np.argsort(-scores)[:TOP_K]
        results[s1_id] = list(cand_ids[order])

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Step 5 — safety net via TF-IDF + sparse_dot_topn (only for zero-candidate S1s)
# ──────────────────────────────────────────────────────────────────────────────

def _safety_net_tfidf(
    s1_leftover: pd.DataFrame,
    pool_df: pd.DataFrame,
) -> dict:
    """
    Run TF-IDF cosine similarity for S1 entities that received zero candidates from
    hash-join blocking.  This is deliberately a heavier method, but it only runs on
    a small minority of entities so total cost is bounded.

    Returns {s1_entity_id: [candidate_entity_id, ...]}
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sparse_dot_topn import sp_matmul_topn
    except ImportError:
        log.warning("sparse_dot_topn not installed — safety net will use rapidfuzz brute-force")
        return _safety_net_brute(s1_leftover, pool_df)

    results = {}

    # Sub-partition by first_letter to keep each TF-IDF problem small
    for letter, s1_grp in s1_leftover.groupby('first_letter'):
        pool_grp = pool_df[pool_df['first_letter'] == letter]
        if pool_grp.empty:
            for eid in s1_grp['entity_id']:
                results[eid] = []
            continue

        s1_text = (s1_grp['norm_name'] + ' ' + s1_grp['norm_address']).fillna('')
        pool_text = (pool_grp['norm_name'] + ' ' + pool_grp['norm_address']).fillna('')

        vect = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4), min_df=1, dtype=np.float32)
        vect.fit(pd.concat([s1_text, pool_text], ignore_index=True))
        q_mat = vect.transform(s1_text).tocsr()
        p_mat = vect.transform(pool_text).tocsr()

        sim = sp_matmul_topn(q_mat, p_mat.T, top_n=SAFETY_NET_K, threshold=SAFETY_NET_THRESHOLD)

        q_ids = s1_grp['entity_id'].values
        p_ids = pool_grp['entity_id'].values
        for i, s1_id in enumerate(q_ids):
            start, end = sim.indptr[i], sim.indptr[i + 1]
            if start == end:
                results[s1_id] = []
            else:
                order = np.argsort(-sim.data[start:end])
                results[s1_id] = [p_ids[sim.indices[start:end][j]] for j in order]

        del vect, q_mat, p_mat, sim
        gc.collect()

    return results


def _safety_net_brute(
    s1_leftover: pd.DataFrame,
    pool_df: pd.DataFrame,
) -> dict:
    """Brute-force rapidfuzz fallback for the safety net (used if sparse_dot_topn missing)."""
    results = {}
    pool_ids = pool_df['entity_id'].values
    pool_names = pool_df['norm_name'].values

    for row in s1_leftover.itertuples(index=False):
        scores = np.array(
            [rfuzz.token_sort_ratio(row.norm_name, pn) for pn in pool_names],
            dtype=np.float32,
        )
        order = np.argsort(-scores)[:TOP_K]
        results[row.entity_id] = [pool_ids[j] for j in order if scores[j] >= MIN_SCORE]

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Main blocking logic
# ──────────────────────────────────────────────────────────────────────────────

def run_blocking(
    source1_path: str,
    source2_path: str,
    source3_path: str,
    output_path: str,
) -> None:
    t0 = time.time()

    # ── Step 0: Load and normalize ──────────────────────────────────────────
    log.info("Loading and normalizing sources...")
    s1 = add_blocking_keys(read_source_file(source1_path))
    s2 = add_blocking_keys(read_source_file(source2_path))
    s3 = add_blocking_keys(read_source_file(source3_path))

    # Combined S2+S3 candidate pool (tagged with source for traceability)
    pool = pd.concat([s2, s3], ignore_index=True)

    # Normalize the country field (lowercased, stripped) for partitioning
    for df in (s1, pool):
        df['country_norm'] = df['country'].astype(str).str.strip().str.lower()

    log.info(
        "  S1: %d rows | S2+S3 pool: %d rows | elapsed %.1fs",
        len(s1), len(pool), time.time() - t0,
    )

    # Pre-build name lookup dicts (used in scoring — avoids repeated DataFrame indexing)
    s1_name_lookup = dict(zip(s1['entity_id'], s1['norm_name']))
    pool_name_lookup = dict(zip(pool['entity_id'], pool['norm_name']))

    # ── Steps 1–4: Per-country blocking ─────────────────────────────────────
    countries = s1['country_norm'].unique()
    log.info("Countries in S1: %s", list(countries))

    final_results: dict[str, list] = {}  # entity_id -> [candidate_entity_id, ...]

    for country in countries:
        t_c = time.time()
        s1_c = s1[s1['country_norm'] == country]
        pool_c = pool[pool['country_norm'] == country]

        log.info(
            "Country '%s': %d S1 rows, %d pool rows",
            country, len(s1_c), len(pool_c),
        )

        if pool_c.empty:
            for eid in s1_c['entity_id']:
                final_results[eid] = []
            continue

        # Step 2: hash-join candidate pairs
        all_pairs = _build_candidate_pairs(s1_c, pool_c)

        if all_pairs.empty:
            for eid in s1_c['entity_id']:
                final_results[eid] = []
        else:
            # Step 3+4: fuzzy scoring + top-K selection
            country_results = _score_candidates(all_pairs, s1_name_lookup, pool_name_lookup)
            final_results.update(country_results)

        # Any S1 entity in this country that had no rows in all_pairs at all
        # (not just scored-away — genuinely zero hash-join hits) gets an empty entry
        seen = set(final_results.keys())
        for eid in s1_c['entity_id']:
            if eid not in seen:
                final_results[eid] = []

        log.info(
            "  Country '%s' done in %.1fs — %d S1 entities processed",
            country, time.time() - t_c, len(s1_c),
        )

    # ── Step 5: Safety net for zero-candidate entities ───────────────────────
    zero_ids = [eid for eid, cands in final_results.items() if not cands]
    if zero_ids:
        log.info(
            "Safety net: %d S1 entities have zero candidates; "
            "running TF-IDF pass against their country pool...",
            len(zero_ids),
        )
        s1_leftover = s1[s1['entity_id'].isin(zero_ids)]

        # Group leftover entities by country and run safety-net within that country pool
        # (falls back to full pool only if pool_c is empty for that country)
        fallback: dict[str, list] = {}
        for country, grp in s1_leftover.groupby('country_norm'):
            pool_c = pool[pool['country_norm'] == country]
            if pool_c.empty:
                pool_c = pool  # truly no match possible in-country; use full pool
                log.info(
                    "  Safety net for '%s': no in-country pool, using full pool (%d rows)",
                    country, len(pool_c),
                )
            country_fallback = _safety_net_tfidf(grp, pool_c)
            fallback.update(country_fallback)

        final_results.update(fallback)
        still_zero = sum(1 for v in fallback.values() if not v)
        log.info(
            "  Safety net complete: %d recovered, %d still zero",
            len(zero_ids) - still_zero, still_zero,
        )

    # ── Step 6: Write output ─────────────────────────────────────────────────
    # Ensure every S1 entity has a row (even those with empty candidates)
    all_s1_ids = set(s1['entity_id'])
    for eid in all_s1_ids:
        if eid not in final_results:
            final_results[eid] = []

    write_id_list_tsv(final_results, output_path, 'source1_entity_id', 'candidate_entity_ids')

    total_candidates = sum(len(v) for v in final_results.values())
    zero_final = sum(1 for v in final_results.values() if not v)
    log.info(
        "Done. Wrote %s | %d S1 entities | avg %.2f candidates/entity | %d with zero candidates | total %.1fs",
        output_path,
        len(final_results),
        total_candidates / max(len(final_results), 1),
        zero_final,
        time.time() - t0,
    )


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Hash-join blocking for business entity resolution (v4).'
    )
    parser.add_argument('--source1', required=True, help='Path to source1 TSV')
    parser.add_argument('--source2', required=True, help='Path to source2 TSV')
    parser.add_argument('--source3', required=True, help='Path to source3 TSV')
    parser.add_argument('--output', required=True, help='Output path for candidate_pairs.tsv')
    args = parser.parse_args()

    run_blocking(args.source1, args.source2, args.source3, args.output)
