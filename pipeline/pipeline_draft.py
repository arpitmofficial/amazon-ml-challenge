import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
import gc
from tqdm import tqdm
import os

def get_top_k_matches(query_features, corpus_features, query_ids, corpus_ids, k=10, threshold=0.7, chunk_size=1000):
    """
    Computes sparse dot product in chunks to find top matches above threshold.
    Returns a dict mapping query_id -> list of matched corpus_ids.
    """
    matches = {qid: [] for qid in query_ids}
    
    # Transpose corpus for dot product
    corpus_T = corpus_features.T.tocsr()
    
    n_queries = query_features.shape[0]
    for start_idx in tqdm(range(0, n_queries, chunk_size), desc="Matching chunks"):
        end_idx = min(start_idx + chunk_size, n_queries)
        query_chunk = query_features[start_idx:end_idx]
        
        # Dot product: query_chunk x corpus_T
        sim_matrix = query_chunk.dot(corpus_T)
        
        # Keep only top k per query to avoid memory explosion
        for i in range(sim_matrix.shape[0]):
            q_idx = start_idx + i
            q_id = query_ids[q_idx]
            row = sim_matrix.getrow(i)
            
            if row.nnz == 0:
                continue
                
            # Get indices and data
            indices = row.indices
            data = row.data
            
            # Filter by threshold
            valid_mask = data >= threshold
            valid_indices = indices[valid_mask]
            valid_data = data[valid_mask]
            
            if len(valid_data) == 0:
                continue
                
            # Sort by similarity descending
            sort_idx = np.argsort(-valid_data)
            
            # Take top k
            top_indices = valid_indices[sort_idx[:k]]
            
            matches[q_id] = [corpus_ids[idx] for idx in top_indices]
            
    return matches

def run_pipeline():
    path_test = "c:/Users/user/Desktop/amazon_ml/6ab10eb3b23ba_student_resource/student_resource/dataset/test"
    output_dir = "c:/Users/user/Desktop/amazon_ml/pipeline/output"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading test data...")
    # Fill NaN with empty string
    s1 = pd.read_csv(f"{path_test}/test_source1.tsv", sep="\t").fillna("")
    s2 = pd.read_csv(f"{path_test}/test_source2.tsv", sep="\t").fillna("")
    s3 = pd.read_csv(f"{path_test}/test_source3.tsv", sep="\t").fillna("")
    
    # Combine text for S1, S2, S3
    s1['text'] = s1['business_name'].astype(str) + " " + s1['business_address'].astype(str)
    s2['text'] = s2['business_name'].astype(str) + " " + s2['business_address'].astype(str)
    s3['text'] = s3['business_name'].astype(str) + " " + s3['business_address'].astype(str)
    
    # Lowercase and clean a bit
    for df in [s1, s2, s3]:
        df['text'] = df['text'].str.lower()
        
    s23 = pd.concat([s2, s3], ignore_index=True)
    
    countries = s1['country'].unique()
    
    all_candidates = {}
    all_matches = {}
    
    for country in countries:
        if not country: continue
        print(f"Processing country: {country}")
        
        # Filter by country
        s1_c = s1[s1['country'] == country].copy()
        s23_c = s23[s23['country'] == country].copy()
        
        if len(s1_c) == 0 or len(s23_c) == 0:
            for q_id in s1_c['entity_id'].values:
                all_candidates[q_id] = []
                all_matches[q_id] = []
            continue
            
        print("Vectorizing...")
        vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 4), min_df=2, max_df=0.8)
        
        # Fit on both query and corpus
        vectorizer.fit(pd.concat([s1_c['text'], s23_c['text']]))
        
        query_features = vectorizer.transform(s1_c['text'])
        corpus_features = vectorizer.transform(s23_c['text'])
        
        q_ids = s1_c['entity_id'].values
        c_ids = s23_c['entity_id'].values
        
        # Candidate Generation (low threshold, e.g. 0.4)
        print("Candidate Generation...")
        # To save time, we do it all in one pass but use two thresholds
        candidates_dict = get_top_k_matches(query_features, corpus_features, q_ids, c_ids, k=20, threshold=0.4, chunk_size=2000)
        
        # Matching (high threshold, e.g. 0.75)
        print("Matching...")
        matches_dict = {qid: [c for c in candidates if True] for qid, candidates in candidates_dict.items()} # We will filter in one go
        # Wait, the get_top_k_matches already does everything, let's just re-filter
        
        # Actually let's just do one pass and split based on score
        # Let's modify the flow to get scores
        pass
        
    # We will refine the script
