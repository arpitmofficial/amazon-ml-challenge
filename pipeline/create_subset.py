import pandas as pd
import numpy as np
import os
import gc

def main():
    train_dir = "../6ab10eb3b23ba_student_resource/student_resource/dataset/train"
    out_dir = "dataset_subset/train"
    os.makedirs(out_dir, exist_ok=True)
    
    print("Reading ground truth...")
    gt = pd.read_csv(os.path.join(train_dir, "train_ground_truth.tsv"), sep="\t").fillna("")
    
    # Take a sample of 20000 source1 entities
    np.random.seed(42)
    s1_sample_ids = set(np.random.choice(gt['source1_entity_id'], size=20000, replace=False))
    
    # Get the true matches for these
    true_s23_ids = set()
    gt_sample = gt[gt['source1_entity_id'].isin(s1_sample_ids)]
    
    for matches in gt_sample['matched_entity_ids']:
        if matches:
            true_s23_ids.update(matches.split(","))
            
    print(f"Sampled {len(s1_sample_ids)} S1 entities and found {len(true_s23_ids)} matching S2/S3 entities.")
    
    gt_sample.to_csv(os.path.join(out_dir, "train_ground_truth.tsv"), sep="\t", index=False)
    del gt, gt_sample; gc.collect()
    
    def process_file(in_path, out_path, filter_ids=None, sample_fraction=0.01):
        print(f"Processing {in_path}...")
        chunksize = 100000
        first = True
        for chunk in pd.read_csv(in_path, sep="\t", chunksize=chunksize):
            if filter_ids is not None:
                # Keep if in filter_ids OR randomly sample some negative examples
                mask = chunk['entity_id'].isin(filter_ids) | (np.random.rand(len(chunk)) < sample_fraction)
                filtered = chunk[mask]
            else:
                mask = np.random.rand(len(chunk)) < sample_fraction
                filtered = chunk[mask]
            
            filtered.to_csv(out_path, sep="\t", index=False, mode='a' if not first else 'w', header=first)
            first = False
            
    process_file(os.path.join(train_dir, "train_source1.tsv"), os.path.join(out_dir, "train_source1.tsv"), s1_sample_ids, 0.0)
    process_file(os.path.join(train_dir, "train_source2.tsv"), os.path.join(out_dir, "train_source2.tsv"), true_s23_ids, 0.05)
    process_file(os.path.join(train_dir, "train_source3.tsv"), os.path.join(out_dir, "train_source3.tsv"), true_s23_ids, 0.05)
    
    print("Done!")

if __name__ == "__main__":
    main()
