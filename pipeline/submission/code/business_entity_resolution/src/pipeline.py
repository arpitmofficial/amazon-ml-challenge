import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
from sparse_dot_topn import sp_matmul_topn
import gc
import os

def run_pipeline():
    path_test = "c:/Users/user/Desktop/amazon_ml/6ab10eb3b23ba_student_resource/student_resource/dataset/test"
    output_dir = "c:/Users/user/Desktop/amazon_ml/pipeline/output"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading test data...")
    s1 = pd.read_csv(f"{path_test}/test_source1.tsv", sep="\t").fillna("")
    s2 = pd.read_csv(f"{path_test}/test_source2.tsv", sep="\t").fillna("")
    s3 = pd.read_csv(f"{path_test}/test_source3.tsv", sep="\t").fillna("")
    
    # Preprocessing
    def clean_text(df):
        text = df['business_name'].astype(str) + " " + df['business_address'].astype(str)
        # simple cleaning
        text = text.str.lower()
        text = text.str.replace(r'[^a-z0-9\s]', ' ', regex=True)
        text = text.str.replace(r'\s+', ' ', regex=True).str.strip()
        return text

    s1['text'] = clean_text(s1)
    s2['text'] = clean_text(s2)
    s3['text'] = clean_text(s3)
    
    s23 = pd.concat([s2, s3], ignore_index=True)
    
    # The countries in test are US, India, and France. Wait, country might be empty.
    s1['country'] = s1['country'].astype(str).str.strip()
    s23['country'] = s23['country'].astype(str).str.strip()
    
    countries = s1['country'].unique()
    print(f"Found countries in S1: {countries}")
    
    all_candidates = []
    all_matches = []
    
    CANDIDATE_K = 15
    CANDIDATE_THRESHOLD = 0.45
    MATCH_THRESHOLD = 0.85 # High precision needed
    
    for country in countries:
        print(f"Processing country: {country}")
        
        s1_c = s1[s1['country'] == country]
        s23_c = s23[s23['country'] == country]
        
        if len(s1_c) == 0: continue
        
        q_ids = s1_c['entity_id'].values
        
        if len(s23_c) == 0:
            for q_id in q_ids:
                all_candidates.append(f"{q_id}\t")
                all_matches.append(f"{q_id}\t")
            continue
            
        c_ids = s23_c['entity_id'].values
        
        print(f"S1 size: {len(s1_c)}, S23 size: {len(s23_c)}")
        
        print("TF-IDF Vectorization...")
        vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 4), min_df=2)
        # To avoid memory issues, we can fit on a sample if it's too big, but let's try fitting on all for the country
        vectorizer.fit(pd.concat([s1_c['text'], s23_c['text']]))
        
        query_features = vectorizer.transform(s1_c['text'])
        corpus_features = vectorizer.transform(s23_c['text'])
        
        print("Sparse dot product top-n...")
        # Get top K matches
        # query_features is shape (N, V), corpus_features is shape (M, V)
        # we want top K from query_features @ corpus_features.T
        sim_matrix = sp_matmul_topn(query_features, corpus_features.T, top_n=CANDIDATE_K, threshold=CANDIDATE_THRESHOLD)
        
        print("Processing results...")
        
        # Iterate row by row
        indptr = sim_matrix.indptr
        indices = sim_matrix.indices
        data = sim_matrix.data
        
        for i in range(len(q_ids)):
            q_id = q_ids[i]
            
            start_idx = indptr[i]
            end_idx = indptr[i+1]
            
            if start_idx == end_idx:
                all_candidates.append(f"{q_id}\t")
                all_matches.append(f"{q_id}\t")
                continue
                
            row_indices = indices[start_idx:end_idx]
            row_data = data[start_idx:end_idx]
            
            # Sort just in case
            sort_idx = np.argsort(-row_data)
            row_indices = row_indices[sort_idx]
            row_data = row_data[sort_idx]
            
            # Candidates
            cand_list = [c_ids[idx] for idx in row_indices]
            # Matches
            match_list = [c_ids[idx] for idx, score in zip(row_indices, row_data) if score >= MATCH_THRESHOLD]
            
            all_candidates.append(f"{q_id}\t" + ",".join(cand_list))
            all_matches.append(f"{q_id}\t" + ",".join(match_list))
            
        del query_features, corpus_features, sim_matrix, vectorizer
        gc.collect()

    print("Writing candidate_pairs.tsv...")
    with open(os.path.join(output_dir, "candidate_pairs.tsv"), "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        f.write("\n".join(all_candidates))
        f.write("\n")
        
    print("Writing matching_results.tsv...")
    with open(os.path.join(output_dir, "matching_results.tsv"), "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        f.write("\n".join(all_matches))
        f.write("\n")
        
    print("Done!")

if __name__ == "__main__":
    run_pipeline()
