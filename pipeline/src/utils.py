"""
Shared utilities for the entity resolution pipeline:
- text normalization (names, addresses)
- TSV I/O for the ID-list files (candidate_pairs.tsv, matching_results.tsv, train_ground_truth.tsv)
- F_0.5 scoring (per-entity + macro average), matching the competition's exact metric
"""

import re
import pandas as pd
import jellyfish


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
# Vectorized normalization + blocking-key generation (used by blocking_v4)
# ---------------------------------------------------------------------------

# Legal suffix expansions for Series-level normalization.
# Direction is the OPPOSITE of _SUFFIX_MAP above (which collapses to short form):
# here we expand abbreviations → full words so char n-gram similarity is maximised
# across sources that write "corp" vs "corporation", etc.
_EXPAND_SUFFIX = {
    r'\bcorp\b': 'corporation',
    r'\bpvt\b': 'private',
    r'\bltd\b': 'limited',
    r'\binc\b': 'incorporated',
    r'\bco\b': 'company',
    r'\brd\b': 'road',
    r'\bst\b': 'street',
}


def normalize_text_series(series: pd.Series) -> pd.Series:
    """
    Vectorized, O(N) text normalization over a pandas Series.

    Steps (identical to the scalar normalize_name but operating on the whole
    column in one pass — no Python-level row loop):
      1. Fill NaN → ""
      2. Lowercase
      3. Strip punctuation (replace non-word, non-space chars with space)
      4. Collapse multiple whitespace runs → single space, strip edges
      5. Expand common legal-suffix abbreviations

    Used identically for business_name across S1, S2, S3, train and test —
    import this function; do not copy-paste it.
    """
    s = series.fillna("").str.lower()
    s = s.str.replace(r'[^\w\s]', ' ', regex=True)
    s = s.str.replace(r'\s+', ' ', regex=True).str.strip()
    for pat, rep in _EXPAND_SUFFIX.items():
        s = s.str.replace(pat, rep, regex=True)
    return s


def add_blocking_keys(df: pd.DataFrame) -> pd.DataFrame:
    """
    Stamp four hash-join blocking key columns onto *df* (in-place copy).

    Columns added:
      norm_name        — normalized business_name (from normalize_text_series)
      norm_address     — normalized business_address (from normalize_text_series)
      digit_signature  — all digit runs from business_address joined into one string
                         (house numbers, PINs, ZIPs).  Empty string when no digits.
      phonetic_code    — Metaphone code of the first token of norm_name.
                         Empty string for empty/non-alpha first tokens.
      name_prefix_key  — First 4 characters of norm_name (after normalization).
      first_letter     — First character of norm_name.

    Call this once per source file.  The result is deterministic across sources
    and across train/test; no comparisons happen here.
    """
    df = df.copy()

    df['norm_name'] = normalize_text_series(df['business_name'])
    df['norm_address'] = normalize_text_series(
        df['business_address'] if 'business_address' in df.columns
        else pd.Series([''] * len(df), dtype=str)
    )

    # digit_signature: join all digit runs from the raw address field
    df['digit_signature'] = (
        df['business_address']
        .fillna('')
        .str.findall(r'\d+')
        .str.join('')
    )

    # phonetic_code via Metaphone on the first token — O(N) over short strings
    first_tokens = df['norm_name'].str.split().str[0].fillna('')
    df['phonetic_code'] = first_tokens.apply(
        lambda t: jellyfish.metaphone(t) if t and t.isalpha() else ''
    )

    df['name_prefix_key'] = df['norm_name'].str[:4]
    df['first_letter'] = df['norm_name'].str[:1]

    return df


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
