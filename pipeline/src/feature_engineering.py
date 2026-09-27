import pandas as pd
import numpy as np
from fuzzywuzzy import fuzz
import Levenshtein
import re
import argparse
import os

def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', ' ', text)
    # Standardize common abbreviations
    replacements = {
        r'\bcorp\b': 'corporation',
        r'\bpvt\b': 'private',
        r'\bltd\b': 'limited',
        r'\binc\b': 'incorporated',
        r'\bllc\b': 'limited liability company',
        r'\bst\b': 'street',
        r'\brd\b': 'road',
        r'\bave\b': 'avenue',
        r'\bblvd\b': 'boulevard'
    }
    for k, v in replacements.items():
        text = re.sub(k, v, text)
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_numbers(text):
    if pd.isna(text):
        return set()
    return set(re.findall(r'\d+', str(text)))

def generate_features(df_pairs, df_s1, df_cands):
    print("Merging features...")
    # df_pairs has columns: source1_entity_id, candidate_entity_id, label (if train)
    
    # Merge S1 features
    df = df_pairs.merge(df_s1, left_on='source1_entity_id', right_on='entity_id', how='left')
    df = df.rename(columns={'business_name': 'name1', 'business_address': 'addr1', 'country': 'country1'})
    
    # Merge Candidate features
    df = df.merge(df_cands, left_on='candidate_entity_id', right_on='entity_id', how='left')
    df = df.rename(columns={'business_name': 'name2', 'business_address': 'addr2', 'country': 'country2'})
    
    print("Cleaning text...")
    for col in ['name1', 'name2', 'addr1', 'addr2']:
        df[f'{col}_clean'] = df[col].apply(clean_text)
        
    print("Calculating similarities...")
    
    # Pre-allocate arrays for speed
    n = len(df)
    name_jw = np.zeros(n)
    name_lev = np.zeros(n)
    name_token_set = np.zeros(n)
    addr_jw = np.zeros(n)
    addr_token_set = np.zeros(n)
    num_match = np.zeros(n)
    num_conflict = np.zeros(n)
    
    for i in range(n):
        n1 = df.at[i, 'name1_clean']
        n2 = df.at[i, 'name2_clean']
        a1 = df.at[i, 'addr1_clean']
        a2 = df.at[i, 'addr2_clean']
        
        # Name similarities
        if n1 and n2:
            name_jw[i] = Levenshtein.jaro_winkler(n1, n2)
            name_lev[i] = Levenshtein.ratio(n1, n2)
            name_token_set[i] = fuzz.token_set_ratio(n1, n2) / 100.0
            
        # Address similarities
        if a1 and a2:
            addr_jw[i] = Levenshtein.jaro_winkler(a1, a2)
            addr_token_set[i] = fuzz.token_set_ratio(a1, a2) / 100.0
            
            # Digit matching (Crucial for addresses to avoid false positives!)
            nums1 = extract_numbers(a1)
            nums2 = extract_numbers(a2)
            if nums1 and nums2:
                intersection = nums1.intersection(nums2)
                if intersection:
                    num_match[i] = 1
                if nums1 != nums2 and not nums1.issubset(nums2) and not nums2.issubset(nums1):
                    # conflicting numbers usually mean different address (e.g. 123 vs 125 main st)
                    num_conflict[i] = 1

    df['name_jw'] = name_jw
    df['name_lev'] = name_lev
    df['name_token_set'] = name_token_set
    df['addr_jw'] = addr_jw
    df['addr_token_set'] = addr_token_set
    df['num_match'] = num_match
    df['num_conflict'] = num_conflict
    
    # Character length differences
    df['name_len_diff'] = abs(df['name1_clean'].str.len() - df['name2_clean'].str.len())
    df['addr_len_diff'] = abs(df['addr1_clean'].str.len() - df['addr2_clean'].str.len())
    
    # Exact matches
    df['exact_name_match'] = (df['name1_clean'] == df['name2_clean']).astype(int)
    
    features = [
        'name_jw', 'name_lev', 'name_token_set', 'addr_jw', 'addr_token_set',
        'num_match', 'num_conflict', 'name_len_diff', 'addr_len_diff', 'exact_name_match'
    ]
    
    return df, features

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--candidates_file", type=str, required=True)
    parser.add_argument("--ground_truth", type=str, default=None)
    parser.add_argument("--output_file", type=str, default="features.csv")
    args = parser.parse_args()
    
    print("Loading data for feature engineering...")
    df_s1 = pd.read_csv(os.path.join(args.data_dir, "train_source1.tsv"), sep="\t")
    df_s2 = pd.read_csv(os.path.join(args.data_dir, "train_source2.tsv"), sep="\t")
    df_s3 = pd.read_csv(os.path.join(args.data_dir, "train_source3.tsv"), sep="\t")
    df_cands = pd.concat([df_s2, df_s3])
    
    candidates = pd.read_csv(args.candidates_file, sep="\t")
    
    # Expand candidates from comma-separated list to rows
    expanded_pairs = []
    for _, row in candidates.iterrows():
        s1 = row['source1_entity_id']
        cands = str(row['candidate_entity_ids']).split(',')
        for c in cands:
            if c and c.strip():
                expanded_pairs.append({'source1_entity_id': s1, 'candidate_entity_id': c.strip()})
                
    df_pairs = pd.DataFrame(expanded_pairs)
    
    # If ground truth exists, attach labels
    if args.ground_truth:
        gt = pd.read_csv(args.ground_truth, sep="\t")
        true_pairs = set()
        for _, row in gt.iterrows():
            s1 = row['source1_entity_id']
            matches = str(row['matched_entity_ids']).split(',')
            for m in matches:
                if m and m.strip():
                    true_pairs.add((s1, m.strip()))
                    
        df_pairs['label'] = df_pairs.apply(lambda x: 1 if (x['source1_entity_id'], x['candidate_entity_id']) in true_pairs else 0, axis=1)
    
    df_features, feature_cols = generate_features(df_pairs, df_s1, df_cands)
    
    # Save the dataframe
    cols_to_save = ['source1_entity_id', 'candidate_entity_id'] + feature_cols
    if 'label' in df_features.columns:
        cols_to_save.append('label')
        
    df_features[cols_to_save].to_csv(args.output_file, index=False)
    print(f"Features saved to {args.output_file}")
