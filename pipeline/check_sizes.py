import pandas as pd
import sys

def check_counts():
    path_train = "c:/Users/user/Desktop/amazon_ml/6ab10eb3b23ba_student_resource/student_resource/dataset/train"
    path_test = "c:/Users/user/Desktop/amazon_ml/6ab10eb3b23ba_student_resource/student_resource/dataset/test"
    
    import os
    print("Train sizes:")
    print("S1:", os.path.getsize(f"{path_train}/train_source1.tsv") / 1024**2, "MB")
    print("S2:", os.path.getsize(f"{path_train}/train_source2.tsv") / 1024**2, "MB")
    print("S3:", os.path.getsize(f"{path_train}/train_source3.tsv") / 1024**2, "MB")
    
    print("\nTest sizes:")
    print("S1:", os.path.getsize(f"{path_test}/test_source1.tsv") / 1024**2, "MB")
    print("S2:", os.path.getsize(f"{path_test}/test_source2.tsv") / 1024**2, "MB")
    print("S3:", os.path.getsize(f"{path_test}/test_source3.tsv") / 1024**2, "MB")

    # line counts
    def count_lines(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            return sum(1 for _ in f)

    print("\nTrain Lines:")
    print("S1:", count_lines(f"{path_train}/train_source1.tsv"))
    print("S2:", count_lines(f"{path_train}/train_source2.tsv"))
    print("S3:", count_lines(f"{path_train}/train_source3.tsv"))
    print("Test Lines:")
    print("S1:", count_lines(f"{path_test}/test_source1.tsv"))
    print("S2:", count_lines(f"{path_test}/test_source2.tsv"))
    print("S3:", count_lines(f"{path_test}/test_source3.tsv"))


if __name__ == "__main__":
    check_counts()
