import pandas as pd
import sys

def explore():
    path = "c:/Users/user/Desktop/amazon_ml/6ab10eb3b23ba_student_resource/student_resource/dataset/train"
    s1 = pd.read_csv(f"{path}/train_source1.tsv", sep="\t", nrows=5)
    print("Source 1:")
    print(s1.head())
    
    s2 = pd.read_csv(f"{path}/train_source2.tsv", sep="\t", nrows=5)
    print("Source 2:")
    print(s2.head())
    
    gt = pd.read_csv(f"{path}/train_ground_truth.tsv", sep="\t", nrows=5)
    print("Ground Truth:")
    print(gt.head())

if __name__ == "__main__":
    explore()
