# AWS EC2 Setup Guide for Team Members

To replicate the training pipeline and generate predictions, you will need a cloud environment with enough RAM and computing power. Follow these steps to set up your own AWS EC2 instance.

## 1. Launching an EC2 Instance
1. Log into your AWS Console and go to **EC2**.
2. Click **Launch Instance**.
3. **OS**: Select **Ubuntu 24.04 LTS** (or 26.04 if available).
4. **Instance Type**: Select an instance with decent RAM. A `t3.large` or `t3.xlarge` is recommended. (Minimum 8GB RAM + Swap).
5. **Key Pair**: Create a new Key Pair (e.g., `my-ml-key.pem`) and download it.
6. **Storage**: Allocate at least **30 GB** of EBS storage (gp3) because the dataset is large when unzipped.
7. Launch the instance.

## 2. Connect to Your Instance
Open your local terminal and restrict permissions on your key file, then SSH into the instance:
```bash
# Mac/Linux only
chmod 400 my-ml-key.pem

# Connect (replace the URL with your instance's Public IPv4 DNS)
ssh -i "my-ml-key.pem" ubuntu@ec2-XX-XX-XX-XX.compute-1.amazonaws.com
```

## 3. Creating Swap Space (Crucial for Memory)
Our Candidate Generation (Blocking) step relies heavily on TF-IDF sparse matrices. To prevent Out-Of-Memory (OOM) crashes on standard instances, we need to create a 15GB Swap file. Run this on your AWS terminal:

```bash
# Create a 15GB swap file
sudo fallocate -l 15G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Verify the swap is active
free -h
```

## 4. Download and Extract the Dataset
You need to download the challenge dataset directly to your AWS instance.

```bash
# Download the dataset (Update URL if necessary)
wget -O dataset.zip "https://cdn.unstop.com/files/6ab10eb3b23ba_student_resource.zip"

# Unzip it
sudo apt update && sudo apt install unzip -y
unzip dataset.zip
```

## 5. Setup the Conda ML Environment
We use Miniconda to manage Python dependencies cleanly.

```bash
# Download and install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda
source ~/miniconda/bin/activate

# Initialize conda
conda init
source ~/.bashrc

# Create the environment for our pipeline
conda create -n amazon_ml python=3.10 -y
conda activate amazon_ml

# Install required libraries
pip install pandas numpy scikit-learn tqdm fuzzywuzzy python-Levenshtein joblib xgboost lightgbm pyarrow
```

## 6. Sync Code to AWS
Now that your AWS instance is ready, open a **NEW local terminal** (on your computer, where the code repo is) and send the code to AWS:

```bash
# Run this locally!
rsync -avz -e "ssh -i my-ml-key.pem" pipeline/ ubuntu@ec2-XX-XX-XX-XX.compute-1.amazonaws.com:~/pipeline/
```

## 7. Run the Pipeline
Go back to your AWS terminal and run the pipeline sequence:

```bash
conda activate amazon_ml

# 1. Blocking (Candidate Generation)
python pipeline/src/blocking.py --data_dir ~/6ab10eb3b23ba_student_resource/student_resource/dataset/train/

# 2. Feature Engineering
python pipeline/src/feature_engineering.py --data_dir ~/6ab10eb3b23ba_student_resource/student_resource/dataset/train/ --candidates_file candidate_pairs.tsv --ground_truth ~/6ab10eb3b23ba_student_resource/student_resource/dataset/train/train_ground_truth.tsv

# 3. Model Training
python pipeline/src/train.py --features_file features.csv --output_file matching_results.tsv
```
