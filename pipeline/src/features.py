"""
Stage 2: Pairwise feature engineering for (S1, candidate) pairs.

Speed note: rapidfuzz's scoring functions are implemented in C and are fast even called
per-pair in a list comprehension -- the slowness in naive pipelines almost always comes from
fuzzywuzzy (pure Python) or pandas .apply() with a Python-level string function per row.
This module avoids both: it precomputes per-record fields once into dicts (O(N) not O(N*K)),
then builds all features with vectorized/list-comprehension passes over the (already small)
candidate-pairs table.
"""

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

from utils import (
    read_source_file,
    normalize_name,
    normalize_address,
    extract_digit_tokens,
    name_tokens,
)


def _build_record_index(df: pd.DataFrame) -> dict:
    """entity_id -> dict of precomputed fields, built once per source file."""
    idx = {}
    for row in df.itertuples(index=False):
        eid = row.entity_id
        norm_name = normalize_name(row.business_name)
        norm_addr = normalize_address(row.business_address)
        idx[eid] = {
            "norm_name": norm_name,
            "norm_addr": norm_addr,
            "name_tokens": name_tokens(norm_name),
            "addr_tokens": frozenset(norm_addr.split()) if norm_addr else frozenset(),
            "digits": extract_digit_tokens(row.business_address),
            "country": (row.country or "").strip().lower(),
            "name_len": len(norm_name),
        }
    return idx


def build_record_indices(source1_path, source2_path, source3_path):
    s1 = read_source_file(source1_path)
    s2 = read_source_file(source2_path)
    s3 = read_source_file(source3_path)
    combined_23 = pd.concat([s2, s3], ignore_index=True)
    return _build_record_index(s1), _build_record_index(combined_23)


def compute_features(pairs_df: pd.DataFrame, s1_index: dict, cand_index: dict) -> pd.DataFrame:
    """
    pairs_df: DataFrame with columns [source1_entity_id, candidate_entity_id]
    Returns a new DataFrame with the same key columns plus feature columns.
    """
    s1_ids = pairs_df["source1_entity_id"].values
    cand_ids = pairs_df["candidate_entity_id"].values
    n = len(pairs_df)

    name_ratio = np.zeros(n)
    name_jaro = np.zeros(n)
    name_token_sort = np.zeros(n)
    name_token_set = np.zeros(n)
    name_jaccard = np.zeros(n)
    addr_ratio = np.zeros(n)
    addr_jaccard = np.zeros(n)
    name_len_delta = np.zeros(n)
    digit_exact_match = np.zeros(n)
    digit_conflict = np.zeros(n)
    country_match = np.zeros(n)
    both_names_present = np.zeros(n)
    both_addrs_present = np.zeros(n)

    for i in range(n):
        a = s1_index.get(s1_ids[i])
        b = cand_index.get(cand_ids[i])
        if a is None or b is None:
            continue  # leave zeros; shouldn't normally happen

        na, nb = a["norm_name"], b["norm_name"]
        aa, ab = a["norm_addr"], b["norm_addr"]

        both_names_present[i] = float(bool(na) and bool(nb))
        both_addrs_present[i] = float(bool(aa) and bool(ab))

        if na and nb:
            name_ratio[i] = fuzz.ratio(na, nb) / 100.0
            name_jaro[i] = fuzz.WRatio(na, nb) / 100.0  # weighted ratio, robust general-purpose
            name_token_sort[i] = fuzz.token_sort_ratio(na, nb) / 100.0
            name_token_set[i] = fuzz.token_set_ratio(na, nb) / 100.0

        ta, tb = a["name_tokens"], b["name_tokens"]
        if ta or tb:
            union = ta | tb
            name_jaccard[i] = len(ta & tb) / len(union) if union else 0.0

        if aa and ab:
            addr_ratio[i] = fuzz.token_set_ratio(aa, ab) / 100.0

        ada, adb = a["addr_tokens"], b["addr_tokens"]
        if ada or adb:
            union = ada | adb
            addr_jaccard[i] = len(ada & adb) / len(union) if union else 0.0

        name_len_delta[i] = abs(a["name_len"] - b["name_len"])

        da, db = a["digits"], b["digits"]
        if da and db:
            if da == db:
                digit_exact_match[i] = 1.0
            elif not (da & db):
                digit_conflict[i] = 1.0  # both have digits but share none -> likely different address

        if a["country"] and b["country"]:
            country_match[i] = float(a["country"] == b["country"])

    out = pairs_df.copy()
    out["name_ratio"] = name_ratio
    out["name_wratio"] = name_jaro
    out["name_token_sort_ratio"] = name_token_sort
    out["name_token_set_ratio"] = name_token_set
    out["name_jaccard"] = name_jaccard
    out["addr_token_set_ratio"] = addr_ratio
    out["addr_jaccard"] = addr_jaccard
    out["name_len_delta"] = name_len_delta
    out["digit_exact_match"] = digit_exact_match
    out["digit_conflict"] = digit_conflict
    out["country_match"] = country_match
    out["both_names_present"] = both_names_present
    out["both_addrs_present"] = both_addrs_present
    return out


FEATURE_COLUMNS = [
    "name_ratio", "name_wratio", "name_token_sort_ratio", "name_token_set_ratio",
    "name_jaccard", "addr_token_set_ratio", "addr_jaccard", "name_len_delta",
    "digit_exact_match", "digit_conflict", "country_match",
    "both_names_present", "both_addrs_present",
]
