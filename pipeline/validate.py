import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
from sparse_dot_topn import sp_matmul_topn
import gc
import os

def f0_5_score(predictions, ground_truth):
    precisions = []
    recalls = []
    for qid in ground_truth:
        true_matches = set(ground_truth[qid])
        pred_matches = set(predictions.get(qid, []))
        
        if len(pred_matches) == 0 and len(true_matches) == 0:
            precisions.append(1.0)
            recalls.append(1.0)
            continue
        if len(pred_matches) == 0 or len(true_matches) == 0:
            precisions.append(0.0)
            recalls.append(0.0)
            continue
            
        tp = len(pred_matches & true_matches)
        fp = len(pred_matches - true_matches)
        fn = len(true_matches - pred_matches)
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        precisions.append(prec)
        recalls.append(rec)
        
    prec = np.mean(precisions)
    rec = np.mean(recalls)
    print(f"Macro Precision: {prec:.4f}")
    print(f"Macro Recall: {rec:.4f}")
    if prec == 0 and rec == 0: return 0.0
    f05 = (1.25 * prec * rec) / (0.25 * prec + rec)
    return f05

def run_validation():
    path_train = "c:/Users/user/Desktop/amazon_ml/6ab10eb3b23ba_student_resource/student_resource/dataset/train"
    
    print("Loading validation data...")
    # Using a 10% sample of S1 for validation to save time
    s1 = pd.read_csv(f"{path_train}/train_source1.tsv", sep="\t").fillna("").sample(frac=0.1, random_state=42)
    s2 = pd.read_csv(f"{path_train}/train_source2.tsv", sep="\t").fillna("")
    s3 = pd.read_csv(f"{path_train}/train_source3.tsv", sep="\t").fillna("")
    gt_df = pd.read_csv(f"{path_train}/train_ground_truth.tsv", sep="\t").fillna("")
    
    # Preprocessing
    def clean_text(df):
        text = df['business_name'].astype(str) + " " + df['business_address'].astype(str)
        text = text.str.lower()
        text = text.str.replace(r'[^a-z0-9\s]', ' ', regex=True)
        text = text.str.replace(r'\s+', ' ', regex=True).str.strip()
        return text

    s1['text'] = clean_text(s1)
    s2['text'] = clean_text(s2)
    s3['text'] = clean_text(s3)
    
    s23 = pd.concat([s2, s3], ignore_index=True)
    
    s1['country'] = s1['country'].astype(str).str.strip()
    s23['country'] = s23['country'].astype(str).str.strip()
    
    countries = s1['country'].unique()
    
    all_predictions = {}
    
    CANDIDATE_K = 10
    MATCH_THRESHOLD = 0.85 
    
    for country in countries:
        print(f"Processing country: {country}")
        s1_c = s1[s1['country'] == country]
        s23_c = s23[s23['country'] == country]
        
        if len(s1_c) == 0: continue
        q_ids = s1_c['entity_id'].values
        
        if len(s23_c) == 0:
            for q_id in q_ids:
                all_predictions[q_id] = []
            continue
            
        c_ids = s23_c['entity_id'].values
        
        print("TF-IDF Vectorization...")
        vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 4), min_df=2)
        vectorizer.fit(pd.concat([s1_c['text'], s23_c['text']]))
        
        query_features = vectorizer.transform(s1_c['text'])
        corpus_features = vectorizer.transform(s23_c['text'])
        
        print("Sparse dot product top-n...")
        sim_matrix = sp_matmul_topn(query_features, corpus_features.T, top_n=CANDIDATE_K, threshold=MATCH_THRESHOLD)
        
        indptr = sim_matrix.indptr
        indices = sim_matrix.indices
        
        for i in range(len(q_ids)):
            q_id = q_ids[i]
            start_idx = indptr[i]
            end_idx = indptr[i+1]
            if start_idx == end_idx:
                all_predictions[q_id] = []
                continue
            row_indices = indices[start_idx:end_idx]
            match_list = [c_ids[idx] for idx in row_indices]
            all_predictions[q_id] = match_list
            
        del query_features, corpus_features, sim_matrix, vectorizer
        gc.collect()

    print("Evaluating...")
    gt_dict = {}
    for _, row in gt_df.iterrows():
        matches = row['matched_entity_ids']
        if not matches:
            gt_dict[row['source1_entity_id']] = []
        else:
            gt_dict[row['source1_entity_id']] = str(matches).split(',')
            
    val_gt = {k: gt_dict[k] for k in all_predictions.keys() if k in gt_dict}
    val_pred = {k: all_predictions[k] for k in val_gt.keys()}
    
    f05 = f0_5_score(val_pred, val_gt)
    print(f"F_0.5 Score: {f05:.4f}")

if __name__ == "__main__":
    run_validation()
