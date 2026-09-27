# AWS EC2 Setup Guide (corrected for 12M-record scale)

The previous version of this doc recommended a `t3.large` (8GB RAM) and referenced
file/flag names that no longer exist in this codebase. At 12M total records, an 8GB
instance is not viable regardless of code optimizations -- this doc reflects what
actually works.

## 1. Launch (or resize) an EC2 Instance

- **Instance type**: `r6i.2xlarge` (8 vCPU, 64 GB RAM) as a baseline, `r6i.4xlarge`
  (16 vCPU, 128 GB RAM) for comfortable headroom. Memory-optimized (`r` family), not
  `t`/`m` family -- this workload is RAM-bound, not CPU-bound.
- Cost: roughly $0.50-$1/hr on-demand. A few hours of compute costs a few dollars --
  don't try to save money by under-provisioning RAM; that's what caused the crashes.
- If you already have an instance: **Stop** it (your EBS volume persists), **Change
  instance type** to one of the above, **Start** it again. No data loss, ~5 minutes.
- **Storage**: at least 50 GB EBS (gp3) -- 12M records plus intermediate candidate/feature
  files add up. Check with `df -h` after loading data; resize the volume live if tight.

## 2. Connect

```bash
chmod 400 my-ml-key.pem
ssh -i "my-ml-key.pem" ubuntu@<your-instance-public-dns>
```

## 3. Swap (extra safety margin only -- not a substitute for enough RAM)

```bash
sudo fallocate -l 15G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
free -h
```

## 4. Get the dataset onto the instance

```bash
wget -O dataset.zip "https://cdn.unstop.com/files/6ab10eb3b23ba_student_resource.zip"
sudo apt update && sudo apt install unzip -y
unzip dataset.zip
```

## 5. Python environment

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda
source ~/miniconda/bin/activate
conda init && source ~/.bashrc

conda create -n amazon_ml python=3.10 -y
conda activate amazon_ml

# Matches requirements.txt exactly -- do not substitute fuzzywuzzy/xgboost, the
# current code uses rapidfuzz + lightgbm + sparse_dot_topn.
pip install -r pipeline/requirements.txt
```

## 6. Sync code to the instance

```bash
# Run locally, not on the AWS box
rsync -avz -e "ssh -i my-ml-key.pem" pipeline/ ubuntu@<your-instance-public-dns>:~/pipeline/
```

## 7. Run the pipeline (correct flags, correct order)

All commands below assume you're in `~/pipeline/src/` on the AWS instance, and the
dataset unzipped to `~/student_resource/dataset/`. Adjust paths if different.

```bash
conda activate amazon_ml
cd ~/pipeline/src

# 1. Blocking on TRAIN (use blocking_v3.py -- country-partitioned + sparse_dot_topn,
#    scales to millions of records; the plain blocking.py TF-IDF approach does not)
python blocking_v3.py \
  --source1 ~/student_resource/dataset/train/train_source1.tsv \
  --source2 ~/student_resource/dataset/train/train_source2.tsv \
  --source3 ~/student_resource/dataset/train/train_source3.tsv \
  --output  ~/train_candidate_pairs.tsv

# 2. Check recall ceiling -- do not skip this
python check_recall_ceiling.py \
  --ground-truth ~/student_resource/dataset/train/train_ground_truth.tsv \
  --candidates   ~/train_candidate_pairs.tsv

# 3. Train the matcher + tune threshold against real F_0.5
python train.py \
  --source1 ~/student_resource/dataset/train/train_source1.tsv \
  --source2 ~/student_resource/dataset/train/train_source2.tsv \
  --source3 ~/student_resource/dataset/train/train_source3.tsv \
  --candidates   ~/train_candidate_pairs.tsv \
  --ground-truth ~/student_resource/dataset/train/train_ground_truth.tsv \
  --model-out     ~/model.joblib \
  --threshold-out ~/threshold.json

# 4. Blocking on TEST
python blocking_v3.py \
  --source1 ~/student_resource/dataset/test/test_source1.tsv \
  --source2 ~/student_resource/dataset/test/test_source2.tsv \
  --source3 ~/student_resource/dataset/test/test_source3.tsv \
  --output  ~/output/candidate_pairs.tsv

# 5. Predict
python predict.py \
  --source1 ~/student_resource/dataset/test/test_source1.tsv \
  --source2 ~/student_resource/dataset/test/test_source2.tsv \
  --source3 ~/student_resource/dataset/test/test_source3.tsv \
  --candidates ~/output/candidate_pairs.tsv \
  --model      ~/model.joblib \
  --threshold  ~/threshold.json \
  --output     ~/output/matching_results.tsv

# 6. Validate before uploading
cd ~/student_resource
python3 utils/validate_submission.py \
  --matching ~/output/matching_results.tsv \
  --candidate ~/output/candidate_pairs.tsv \
  --test-dir dataset/test
```

## Notes

- Do **not** use `submission/code/business_entity_resolution/src/pipeline.py` as-is for
  your real run -- it has hardcoded local Windows paths and makes match decisions from a
  raw, unvalidated cosine-similarity threshold instead of the trained model. The useful
  ideas from it (country partitioning, `sparse_dot_topn`) are already folded into
  `blocking_v3.py` above, wired back into the real train/predict pipeline.
- Watch memory live while running: `watch -n 2 free -h`. If it climbs sharply during
  blocking, stop and tell me which country partition it happened on -- one country's
  partition may simply be too large on its own (e.g. if the US split is still several
  million rows) and may need a further sub-split (e.g. by first letter of name) inside
  that partition.
