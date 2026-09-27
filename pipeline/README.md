# Business Entity Resolution Pipeline

Run everything from this directory. Assumes the standard `dataset/train/` and `dataset/test/`
layout from the challenge's `student_resource/` package sits alongside this folder (adjust
paths as needed).

## 0. Install

```bash
pip install -r requirements.txt
```

## 1. Blocking (candidate generation) -- run on TRAIN first

```bash
cd src
python blocking.py \
  --source1 ../../dataset/train/train_source1.tsv \
  --source2 ../../dataset/train/train_source2.tsv \
  --source3 ../../dataset/train/train_source3.tsv \
  --output  ../train_candidate_pairs.tsv
```

## 2. Check recall ceiling -- DO NOT SKIP THIS

```bash
python check_recall_ceiling.py \
  --ground-truth ../../dataset/train/train_ground_truth.tsv \
  --candidates   ../train_candidate_pairs.tsv
```

If the reported recall ceiling is below ~0.95, go back and widen blocking (`blocking.py`:
raise `TOP_K`, lower `MIN_TFIDF_SCORE`) before doing anything else. No downstream model can
recover a true match that blocking never proposed.

## 3. Train the matcher + tune threshold

```bash
python train.py \
  --source1 ../../dataset/train/train_source1.tsv \
  --source2 ../../dataset/train/train_source2.tsv \
  --source3 ../../dataset/train/train_source3.tsv \
  --candidates   ../train_candidate_pairs.tsv \
  --ground-truth ../../dataset/train/train_ground_truth.tsv \
  --model-out     ../model.joblib \
  --threshold-out ../threshold.json
```

This prints the validation macro F_0.5 at the best threshold found. That number is your
honest estimate of leaderboard performance.

## 4. Blocking on TEST

```bash
python blocking.py \
  --source1 ../../dataset/test/test_source1.tsv \
  --source2 ../../dataset/test/test_source2.tsv \
  --source3 ../../dataset/test/test_source3.tsv \
  --output  ../output/candidate_pairs.tsv
```

## 5. Predict -> matching_results.tsv

```bash
python predict.py \
  --source1 ../../dataset/test/test_source1.tsv \
  --source2 ../../dataset/test/test_source2.tsv \
  --source3 ../../dataset/test/test_source3.tsv \
  --candidates ../output/candidate_pairs.tsv \
  --model      ../model.joblib \
  --threshold  ../threshold.json \
  --output     ../output/matching_results.tsv
```

## 6. Validate before submitting

```bash
cd ../..   # back to student_resource/
python3 utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test
```

## Design notes & Pipeline Approach (Methodology)

Our overall architecture leverages a 3-stage pipeline explicitly optimized for high recall and memory efficiency to run locally on an 8GB RAM machine. Every script includes `time` tracking to monitor stage durations.

1.  **Stage 1 - Blocking & Candidate Generation (`blocking.py`)**
    *   **Goal**: Maximize recall to $ \ge 95\% $.
    *   **Strategy**: We compute a hybrid candidate pool per `Source 1` entity.
        1.  *Sparse TF-IDF Cosine Similarity*: Character n-grams (2-4) are vectorized. To avoid Out-of-Memory (OOM) errors, we never densify the similarity matrix (`.toarray()`). We use chunked `scipy.sparse.csr_matrix` `.dot()` operations and directly query `row.data` / `row.indices`.
        2.  *Token-Overlap Inverted Index*: Catches extreme word re-ordering or partial overlaps that char n-grams might under-rank.
    *   **Result**: The union of these two candidate sets is deduped and capped at a maximum of 30 candidates per entity. Hard country filtering is strictly avoided here because cross-source country label noise would silently cap recall (e.g. test set adds France).

2.  **Stage 2 - Feature Engineering (`features.py` & `train.py`)**
    *   **Goal**: Build powerful tabular features for the classifier to distinguish true matches from hard negatives.
    *   **Strategy**: All candidate pairs are featurized in batched/vectorized operations (avoiding slow row-by-row python loops).
    *   **Features Used**:
        *   RapidFuzz string similarities (ratio, WRatio, token sort, token set).
        *   Jaccard similarity on name/address tokens.
        *   Digit-set exact-match and conflict flags (extremely high precision signals for house numbers / PINs).
        *   Country-match soft flags and missing-field indicators.

3.  **Stage 3 - Training & Optimization (`train.py`)**
    *   **Goal**: Attain Macro $F_{0.5} \ge 0.98$.
    *   **Strategy**: We train a `LightGBM` binary classifier, using `scale_pos_weight` to address class imbalance. We train exclusively against hard negatives (candidates that survived blocking but aren't matches).
    *   **Thresholding**: Instead of relying on default probability cutoffs, the script performs a hyperparameter sweep over thresholds (0.05 to 0.95), directly optimizing the actual competition metric (Macro $F_{0.5}$) on an entity-grouped validation split.

4.  **Stage 4 - Consistency & Resolution (`predict.py`)**
    *   **Goal**: Ensure one-to-one logical consistency.
    *   **Strategy**: A greedy assignment pass ensures that a given `Source 2` or `Source 3` record is claimed by *at most one* `Source 1` entity, wiping out an entire class of false positives.

## Current State & Next Steps (As of Latest Run)
*   **Pipeline Phase:** We are currently executing **Stage 1 (Blocking)** on the massive `train` dataset.
*   **The Challenge:** The combined training dataset contains over 12.5 million rows (`S1`: 2.2M, `S2`: 5.0M, `S3`: 5.2M). 
*   **Current Blocker:** While the sparse `.dot()` matrix multiplication bug was resolved, the initial `TfidfVectorizer.fit(all_text)` step is currently causing an Out-of-Memory (OOM) crash (allocating >28GB of Virtual Memory) on the 8GB AWS instance.
*   **Next Step:** Awaiting a further-reduced-memory variant for the `blocking.py` script that either fits the vectorizer incrementally (using `HashingVectorizer`) or processes `Source 2` and `Source 3` in entirely separate sequential passes to halve the memory footprint. Once blocking successfully generates `candidate_pairs.tsv`, we will immediately proceed to `check_recall_ceiling.py` and Model Training (`train.py`).
