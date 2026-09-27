"""
Shared utilities for the entity resolution pipeline:
- text normalization (names, addresses)
- TSV I/O for the ID-list files (candidate_pairs.tsv, matching_results.tsv, train_ground_truth.tsv)
- F_0.5 scoring (per-entity + macro average), matching the competition's exact metric
"""

import re
import pandas as pd


# ---------------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------------

# Canonical mapping for legal-suffix variants -> single token.
# We do NOT delete these; we replace them with a canonical form and also
# expose a separate "suffix present" signal downstream in feature engineering.
_SUFFIX_MAP = {
    r"\bcorporation\b": "corp",
    r"\bcorp\.?\b": "corp",
    r"\bincorporated\b": "inc",
    r"\binc\.?\b": "inc",
    r"\blimited\b": "ltd",
    r"\bltd\.?\b": "ltd",
    r"\bprivate\b": "pvt",
    r"\bpvt\.?\b": "pvt",
    r"\bcompany\b": "co",
    r"\bco\.?\b": "co",
    r"\bllc\b": "llc",
    r"&": " and ",
}

_ADDR_ABBR_MAP = {
    r"\broad\b": "rd",
    r"\bstreet\b": "st",
    r"\bavenue\b": "ave",
    r"\bboulevard\b": "blvd",
    r"\blane\b": "ln",
    r"\bdrive\b": "dr",
    r"\bapartment\b": "apt",
    r"\bfloor\b": "fl",
    r"\bnear\b": "near",
}

_PUNCT_RE = re.compile(r"[^\w\s]")
_MULTI_SPACE_RE = re.compile(r"\s+")
_DIGIT_RE = re.compile(r"\d+")


def _base_clean(text: str) -> str:
    if text is None or (isinstance(text, float)):  # NaN
        return ""
    text = str(text).lower().strip()
    return text


def normalize_name(text: str) -> str:
    """Lowercase, canonicalize legal suffixes, strip punctuation, collapse whitespace."""
    text = _base_clean(text)
    if not text:
        return ""
    for pattern, repl in _SUFFIX_MAP.items():
        text = re.sub(pattern, repl, text)
    text = _PUNCT_RE.sub(" ", text)
    text = _MULTI_SPACE_RE.sub(" ", text).strip()
    return text


def normalize_address(text: str) -> str:
    """Lowercase, canonicalize common address abbreviations, strip punctuation."""
    text = _base_clean(text)
    if not text:
        return ""
    for pattern, repl in _ADDR_ABBR_MAP.items():
        text = re.sub(pattern, repl, text)
    text = _PUNCT_RE.sub(" ", text)
    text = _MULTI_SPACE_RE.sub(" ", text).strip()
    return text


def extract_digit_tokens(text: str) -> frozenset:
    """Extract all standalone digit runs (house numbers, PIN/ZIP, unit numbers)."""
    text = _base_clean(text)
    if not text:
        return frozenset()
    return frozenset(_DIGIT_RE.findall(text))


def name_tokens(normalized_name: str) -> frozenset:
    """Token set for a normalized name, used for token-overlap blocking / Jaccard."""
    if not normalized_name:
        return frozenset()
    return frozenset(normalized_name.split())


# ---------------------------------------------------------------------------
# TSV I/O for ID-list files
# ---------------------------------------------------------------------------

def read_source_file(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    return df


def read_id_list_tsv(path: str, id_col: str, list_col: str) -> dict:
    """
    Read a two-column TSV (entity_id <TAB> comma,separated,ids) into
    {entity_id: frozenset(ids)}. Empty string -> empty frozenset.
    """
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    out = {}
    for _, row in df.iterrows():
        key = row[id_col]
        raw = row[list_col]
        ids = frozenset(x for x in raw.split(",") if x) if raw else frozenset()
        out[key] = ids
    return out


def write_id_list_tsv(mapping: dict, path: str, id_col: str, list_col: str) -> None:
    """
    Write {entity_id: iterable_of_ids} to a two-column TSV.
    Deduplicates and sorts each id list for determinism.
    """
    rows = []
    for key in sorted(mapping.keys()):
        ids = sorted(set(mapping[key]))
        rows.append({id_col: key, list_col: ",".join(ids)})
    out_df = pd.DataFrame(rows, columns=[id_col, list_col])
    out_df.to_csv(path, sep="\t", index=False)


# ---------------------------------------------------------------------------
# Scoring: exact competition metric
# ---------------------------------------------------------------------------

def f_beta_entity(true_set: frozenset, pred_set: frozenset, beta: float = 0.5) -> float:
    """
    Per-entity F_beta as defined in the problem statement.
    - Singleton correctly predicted empty -> 1.0
    - Singleton with any false positive prediction -> 0.0
    - Otherwise standard precision/recall F_beta, 0.0 if precision+recall == 0
    """
    if not true_set and not pred_set:
        return 1.0
    if not pred_set:
        # missed everything (or correctly predicted empty when it wasn't) -> recall 0 or true was empty handled above
        return 0.0
    tp = len(true_set & pred_set)
    precision = tp / len(pred_set) if pred_set else 0.0
    recall = tp / len(true_set) if true_set else 0.0
    if precision == 0.0 and recall == 0.0:
        return 0.0
    beta2 = beta * beta
    denom = (beta2 * precision) + recall
    if denom == 0:
        return 0.0
    return (1 + beta2) * precision * recall / denom


def macro_f_beta(true_map: dict, pred_map: dict, beta: float = 0.5) -> float:
    """
    Macro-average F_beta over all S1 entity ids present in true_map.
    pred_map entries missing for an id are treated as empty predictions.
    """
    scores = []
    for entity_id, true_set in true_map.items():
        pred_set = pred_map.get(entity_id, frozenset())
        scores.append(f_beta_entity(true_set, pred_set, beta))
    return sum(scores) / len(scores) if scores else 0.0
