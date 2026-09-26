# Business Entity Resolution Pipeline

## Overview
This pipeline uses a highly optimized TF-IDF vectorizer (with character n-grams) combined with sparse matrix multiplication (using `sparse_dot_topn`) to find exact and fuzzy matches across the reference and candidate datasets. Blocking and matching is done simultaneously by keeping candidates above a lower threshold (`0.45`) and matching those above a strict threshold (`0.85`).

## How to reproduce
1. Create a python environment with Python 3.12+
2. Install the requirements:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the pipeline script. The script assumes the test dataset is located at `c:/Users/user/Desktop/amazon_ml/6ab10eb3b23ba_student_resource/student_resource/dataset/test`. You may need to update the path inside `pipeline.py` if running from a different location.
   ```bash
   python src/pipeline.py
   ```
4. Output files `candidate_pairs.tsv` and `matching_results.tsv` will be generated in the `c:/Users/user/Desktop/amazon_ml/pipeline/output` directory.
