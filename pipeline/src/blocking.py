import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import argparse
import os

def load_data(data_dir):
    print("Loading datasets...")
    df_s1 = pd.read_csv(os.path.join(data_dir, "train_source1.tsv"), sep="\t")
    df_s2 = pd.read_csv(os.path.join(data_dir, "train_source2.tsv"), sep="\t")
    df_s3 = pd.read_csv(os.path.join(data_dir, "train_source3.tsv"), sep="\t")
    return df_s1, df_s2, df_s3

def preprocess_text(df, columns):
    """Fills NaN and lowercases text columns for TF-IDF"""
    for col in columns:
        df[col] = df[col].fillna("").astype(str).str.lower()
    # Combine name and address into a single feature for blocking
    df['combined_text'] = df['business_name'] + " " + df['business_address']
    return df

def generate_candidates(df_source1, df_source2_or_3, k_candidates=5):
    """
    Blocks by country to reduce search space, then calculates TF-IDF cosine similarity.
    Returns a dataframe of candidate pairs.
    """
    candidates = []
    
    countries = df_source1['country'].unique()
    
    for country in countries:
        print(f"Processing country: {country}")
        s1_subset = df_source1[df_source1['country'] == country].reset_index(drop=True)
        s23_subset = df_source2_or_3[df_source2_or_3['country'] == country].reset_index(drop=True)
        
        if len(s1_subset) == 0 or len(s23_subset) == 0:
            continue
            
        # Fit TF-IDF on both combined
        vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4), min_df=2)
        
        # Fit on candidate database
        tfidf_s23 = vectorizer.fit_transform(s23_subset['combined_text'])
        # Transform queries
        tfidf_s1 = vectorizer.transform(s1_subset['combined_text'])
        
        print(f"  Calculating similarity for {len(s1_subset)} S1 entities against {len(s23_subset)} candidates...")
        # Calculate cosine similarity (sparse matrix multiplication)
        # Note: For very large datasets, we might need chunking here to avoid OOM
        similarities = cosine_similarity(tfidf_s1, tfidf_s23)
        
        # Get top K indices for each S1 entity
        for i in range(len(s1_subset)):
            s1_id = s1_subset.loc[i, 'entity_id']
            # Get indices of top k scores
            top_k_idx = similarities[i].argsort()[-k_candidates:][::-1]
            
            cand_list = []
            for idx in top_k_idx:
                score = similarities[i, idx]
                if score > 0.1: # Minimum similarity threshold to be a candidate
                    cand_list.append(s23_subset.loc[idx, 'entity_id'])
            
            if cand_list:
                candidates.append({
                    'source1_entity_id': s1_id,
                    'candidate_entity_ids': ",".join(cand_list)
                })
                
    return pd.DataFrame(candidates)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the dataset directory")
    parser.add_argument("--output_file", type=str, default="candidate_pairs.tsv")
    args = parser.parse_args()
    
    df_s1, df_s2, df_s3 = load_data(args.data_dir)
    
    df_s1 = preprocess_text(df_s1, ['business_name', 'business_address'])
    df_s2 = preprocess_text(df_s2, ['business_name', 'business_address'])
    df_s3 = preprocess_text(df_s3, ['business_name', 'business_address'])
    
    print("Generating candidates from Source 2...")
    cands_s2 = generate_candidates(df_s1, df_s2, k_candidates=5)
    
    print("Generating candidates from Source 3...")
    cands_s3 = generate_candidates(df_s1, df_s3, k_candidates=5)
    
    # Combine S2 and S3 candidates
    all_cands = pd.concat([cands_s2, cands_s3])
    
    # Group by source1_entity_id and merge the comma-separated strings
    final_cands = all_cands.groupby('source1_entity_id')['candidate_entity_ids'].apply(
        lambda x: ','.join(filter(None, x))
    ).reset_index()
    
    final_cands.to_csv(args.output_file, sep='\t', index=False)
    print(f"Candidate pairs saved to {args.output_file}")
