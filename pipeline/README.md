# Amazon ML Challenge 2026: Entity Resolution Pipeline

## Overview
This repository contains our pipeline for the Amazon ML Challenge 2026. The objective is to match business entities across three noisy sources without common identifiers. The metric for this challenge is $F_{0.5}$, which severely penalizes false positives (merging different businesses).

## Target Score
**Target**: $F_{0.5} \ge 0.97$

## Current Approach (v1.0 - Baseline)

### Stage 1: Candidate Generation (Blocking)
We use a high-recall blocking strategy to reduce the search space from $O(N \times M)$ to a small candidate set per Source 1 entity. 

**Implementation Details (`src/blocking.py`)**:
- **Preprocessing**: Convert text columns (`business_name`, `business_address`) to lowercase and handle missing values.
- **Combined Text**: Concatenate `business_name` and `business_address` into a single string.
- **Hard Blocking**: We block exactly by the `country` column. Source 1 entities in a country are only compared to Source 2/3 candidates in the same country.
- **TF-IDF Vectorization**: We extract character n-grams (sizes 2 to 4) using `TfidfVectorizer(analyzer='char_wb')`. This handles typos, transliterations, and abbreviations efficiently.
- **Cosine Similarity**: We calculate pairwise cosine similarity between Source 1 queries and Source 2/3 candidates using sparse matrix multiplication.
- **Top-K Selection**: We retrieve the top $K=5$ candidates that have a cosine similarity $> 0.1$.
- **Output**: `candidate_pairs.tsv` containing the narrowed-down search space.

### Estimated $F_{0.5}$ Score for Current Implementation
The current script **only** performs Candidate Generation. If we were to submit this raw output (assuming all candidates are true matches), our **Precision would be extremely low**, and our $F_{0.5}$ score would be poor (likely < 0.20) because $F_{0.5}$ heavily penalizes false positives, and our Top-5 approach generates many false positives (up to 5 candidates per entity). 

To reach $0.97$, this blocking step is just the **first phase** (designed for high *Recall*, not high Precision). 

### Stage 2: Feature Engineering (`src/feature_engineering.py`)
To push our candidates into a >0.98 precision zone, we compute deep comparative features between Source 1 and Source 2/3 candidates:
- **String Similarity Algorithms**: Levenshtein Distance (for typos), Jaro-Winkler (great for prefix-heavy names), and Fuzzy Token Set Ratio (handles word order swaps like "Amazon Inc" vs "Inc Amazon").
- **Address Digits Matching**: Addresses are highly dependent on numbers. We extract digits from both addresses. If they match, we flag it. **Crucially**, if they have conflicting numbers (e.g. 123 Main St vs 125 Main St), we heavily penalize it, as text-only similarity algorithms easily miss this.
- **Length Deltas**: Character length differences.
- **Labels generation**: We inject the `train_ground_truth.tsv` labels to convert this into a supervised classification problem.

### Stage 3: Supervised Classification & F_0.5 Optimization (`src/train.py`)
To achieve the target **$F_{0.5} \ge 0.98$**:
1. We train an **XGBoost Classifier** on the engineered features.
2. Because $F_{0.5}$ weights Precision 2x over Recall, the default 0.5 probability threshold will fail (it predicts too many false positives).
3. We perform **Custom Threshold Tuning**: We split the training data (80/20) and scan probability thresholds from 0.1 to 0.95. We pick the exact threshold that maximizes the $F_{0.5}$ score on the validation set.
4. **Aggressive Precision Backoff**: If the model validation organic $F_{0.5}$ score drops below 0.98, the code automatically forces the threshold to an extreme high ($>0.90$). This means the model will strictly predict a match *only* if it is 90%+ certain. Otherwise, it safely predicts a singleton (which gives a full 1.0 score if correct, avoiding the harsh false-positive penalty).
