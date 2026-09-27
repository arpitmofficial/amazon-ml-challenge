import pandas as pd
import numpy as np

def calculate_f05(pred_df, gt_df):
    gt_dict = dict(zip(gt_df['source1_entity_id'], gt_df['matched_entity_ids'].fillna('')))
    
    total_f05 = 0
    count = 0
    
    for _, row in pred_df.iterrows():
        s1_id = row['source1_entity_id']
        pred_matches = row['matched_entity_ids']
        if pd.isna(pred_matches): pred_matches = ""
        
        if s1_id not in gt_dict:
            continue
            
        true_matches = gt_dict[s1_id]
        
        pred_set = set(pred_matches.split(",")) if pred_matches else set()
        true_set = set(true_matches.split(",")) if true_matches else set()
        
        if not pred_set and not true_set:
            f05 = 1.0
        elif not pred_set or not true_set:
            f05 = 0.0
        else:
            tp = len(pred_set.intersection(true_set))
            fp = len(pred_set - true_set)
            fn = len(true_set - pred_set)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            if precision == 0 and recall == 0:
                f05 = 0.0
            else:
                f05 = (1.25 * precision * recall) / (0.25 * precision + recall)
                
        total_f05 += f05
        count += 1
        
    return total_f05 / count if count > 0 else 0

if __name__ == "__main__":
    pred = pd.read_csv("output/matching_results.tsv", sep="\t")
    gt = pd.read_csv("dataset_subset/train/train_ground_truth.tsv", sep="\t")
    
    score = calculate_f05(pred, gt)
    print(f"F0.5 Score: {score:.5f}")
