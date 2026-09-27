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

## Design notes (for the methodology doc)

- **Blocking**: Token-level inverted index with fully vectorised Numpy hit-counting for extreme speed and low memory footprint. Hard blocking by country was retained to prevent memory explosions on 5M+ records, but uses a highly optimised `np.unique` vectorisation strategy that runs the entire ~12M record dataset in ~45 minutes using <4GB RAM.
- **Features**: rapidfuzz-based string similarity (ratio, WRatio, token sort/set ratio),
  Jaccard on name/address tokens, digit-set exact-match / conflict flags (address house
  numbers / PINs), country-match flag, presence flags for missing fields.
- **Model**: LightGBM binary classifier, `scale_pos_weight` for class imbalance, hard
  negatives = non-matching candidates that survived blocking (not random negatives).
- **Threshold**: swept directly against the competition's macro F_0.5 metric on a
  by-entity validation split, not against accuracy/AUC/F1.
- **Consistency pass**: greedy one-to-one resolution on the candidate side (a given S2/S3
  record is claimed by at most one S1 entity) to remove an entire class of false positives.
