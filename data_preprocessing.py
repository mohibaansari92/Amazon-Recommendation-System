import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob  # Basic sentiment; run: python -m textblob.download_corpora
import joblib  # For saving models/vectorizers
from sentence_transformers import SentenceTransformer  # For BERT; optional
from sklearn.decomposition import PCA  # Optional for BERT dim reduction
import gc  # For garbage collection to free memory

def preprocess_data(use_bert=True, subsample_frac=1.0, reduce_bert_dims=True, force_full_bert=False, tfidf_max_features=500):
    # Load the dataset
    try:
        df = pd.read_csv('MultipleFiles/Reviews.csv')
        print("Dataset loaded successfully.")
        print(f"Dataset shape: {df.shape}")
        print("\nFirst few rows:")
        print(df.head())
    except FileNotFoundError:
        print("Error: 'MultipleFiles/Reviews.csv' not found. Ensure the file is in the correct location.")
        return None

    # Optional: Subsample for faster processing/testing (applies to entire pipeline)
    if subsample_frac < 1.0:
        df = df.sample(frac=subsample_frac, random_state=42).reset_index(drop=True)
        print(f"Subsampled to {len(df)} rows for testing.")

    # Initial data exploration
    print("\nDataset Info:")
    df.info()
    print("\nMissing values before cleaning:")
    print(df.isnull().sum())

    # Handle missing values
    initial_rows = len(df)
    df.dropna(subset=['Score'], inplace=True)
    print(f"\nDropped {initial_rows - len(df)} rows with missing 'Score'.")

    df['ProfileName'] = df['ProfileName'].fillna('Anonymous')
    df['Summary'] = df['Summary'].fillna('')
    df['Text'] = df['Text'].fillna('')

    print("\nMissing values after cleaning:")
    print(df.isnull().sum())

    # Feature Engineering: Encode UserId and ProductId (standard Amazon: 'User Id' no space)
    user_encoder = LabelEncoder()
    product_encoder = LabelEncoder()
    df['user_id_encoded'] = user_encoder.fit_transform(df['UserId'].astype(str))
    df['product_id_encoded'] = product_encoder.fit_transform(df['ProductId'].astype(str))

    print("\nEncoded IDs sample:")
    print(df[['UserId', 'user_id_encoded', 'ProductId', 'product_id_encoded', 'Score']].head())

    # Feature Engineering: Helpfulness Ratio
    df['HelpfulnessRatio'] = df.apply(
        lambda row: row['HelpfulnessNumerator'] / row['HelpfulnessDenominator'] if row['HelpfulnessDenominator'] > 0 else 0,
        axis=1
    )
    print("\nHelpfulnessRatio sample:")
    print(df[['HelpfulnessNumerator', 'HelpfulnessDenominator', 'HelpfulnessRatio']].head())

    # NEW: Review Length (word count)
    df['review_length'] = df['Text'].apply(lambda x: len(str(x).split()))
    print("\nReview Length sample:")
    print(df[['review_length']].head())

    # NEW: Basic Sentiment Score (TextBlob polarity: -1 negative to +1 positive)
    try:
        df['sentiment_score'] = df['Text'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
    except Exception as e:
        print(f"TextBlob failed (run 'python -m textblob.download_corpora'): {e}. Using 0 for sentiment.")
        df['sentiment_score'] = 0.0
    print("\nSentiment Score sample:")
    print(df[['sentiment_score']].head())

    # NEW: User and Product Average Scores (collaborative filtering)
    df['user_avg_score'] = df.groupby('user_id_encoded')['Score'].transform('mean')
    df['product_avg_score'] = df.groupby('product_id_encoded')['Score'].transform('mean')
    print("\nUser  / Product Avg Scores sample:")
    print(df[['user_id_encoded', 'user_avg_score', 'product_id_encoded', 'product_avg_score', 'Score']].head())

    # Feature Engineering: Time-based features
    df['Time'] = pd.to_datetime(df['Time'], unit='s')
    df['review_year'] = df['Time'].dt.year
    df['review_month'] = df['Time'].dt.month
    df['review_dayofweek'] = df['Time'].dt.dayofweek

    print("\nTime-based features sample:")
    print(df[['Time', 'review_year', 'review_month', 'review_dayofweek']].head())

    # TF-IDF: Fit vectorizer but DON'T densify/add to df (saves memory; apply in training)
    print(f"\nFitting TF-IDF vectorizer on Text (max_features={tfidf_max_features})...")
    tfidf = TfidfVectorizer(max_features=tfidf_max_features, stop_words='english', ngram_range=(1, 2))
    tfidf_matrix = tfidf.fit_transform(df['Text'])  # Keep sparse
    joblib.dump(tfidf, 'tfidf_vectorizer.pkl')
    print(f"TF-IDF fitted: sparse shape {tfidf_matrix.shape}. Vectorizer saved—no columns added to CSV to save memory.")
    del tfidf_matrix  # Free memory
    gc.collect()  # Garbage collection

    # OPTIONAL: BERT Embeddings (semantic text features; heavy—set use_bert=False to skip)
    bert_used = False
    if use_bert:
        print("\nGenerating BERT embeddings from Text...")
        try:
            # Auto-subsample for BERT if full data (to avoid memory issues; override with force_full_bert=True)
            bert_frac = subsample_frac
            if subsample_frac == 1.0 and not force_full_bert:
                bert_frac = 0.1  # Default: 10% for BERT on full data
                df_bert = df.sample(frac=bert_frac, random_state=42).reset_index(drop=True)
                print(f"Auto-subsampling to {len(df_bert)} rows for BERT (set force_full_bert=True to use full).")
            else:
                df_bert = df
            print(f"BERT on {len(df_bert)} rows (may take 5–20 mins)...")

            # Lightweight SentenceTransformer
            bert_model = SentenceTransformer('all-MiniLM-L6-v2')
            embeddings = bert_model.encode(
                df_bert['Text'].tolist(), 
                batch_size=64,  # Adjust lower (32) if OOM
                show_progress_bar=True
            )
            
            # Optional: Reduce dims with PCA (uncomment for 50 dims)
            # pca = PCA(n_components=50, random_state=42)
            # embeddings = pca.fit_transform(embeddings)
            # joblib.dump(pca, 'bert_pca.pkl')
            
            # Truncate to 10 dims for simplicity (set reduce_bert_dims=False to keep all ~384)
            if reduce_bert_dims:
                n_dims = min(10, embeddings.shape[1])
                embedding_cols = [f'bert_emb_{i}' for i in range(n_dims)]
                embedding_df = pd.DataFrame(embeddings[:, :n_dims], columns=embedding_cols, index=df_bert.index)
            else:
                embedding_cols = [f'bert_emb_{i}' for i in range(embeddings.shape[1])]
                embedding_df = pd.DataFrame(embeddings, columns=embedding_cols, index=df_bert.index)
            
            # Align indices and concat to main df
            df = df.merge(embedding_df, left_index=True, right_index=True, how='left')
            # Fill NaN for non-subsampled rows (if auto-subsampled)
            for col in embedding_cols:
                df[col] = df[col].fillna(0.0)
            
            joblib.dump(bert_model, 'bert_model.pkl')
            bert_used = True
            print(f"BERT embeddings added: original shape {embeddings.shape}, used dims: {len(embedding_cols)}")
            del embeddings  # Free memory
            gc.collect()
        except ImportError:
            print("BERT failed: Install 'sentence-transformers' with 'pip install sentence-transformers'.")
        except Exception as e:
            print(f"BERT failed (other error): {e}. Falling back to no BERT.")
    else:
        print("Skipping BERT embeddings.")

    # Save processed data (lightweight: base features + Text/Summary for TF-IDF in training)
    output_file = 'processed_data.csv.gz'  # Gzipped to save space
    # Base columns + text (exclude large non-essential like ProfileName if wanted)
    base_cols = ['UserId', 'ProductId', 'HelpfulnessNumerator', 'HelpfulnessDenominator', 'Score', 'Time', 
                 'Summary', 'Text', 'user_id_encoded', 'product_id_encoded', 'HelpfulnessRatio', 
                 'review_length', 'sentiment_score', 'user_avg_score', 'product_avg_score', 
                 'review_year', 'review_month', 'review_dayofweek']
    if bert_used:
        base_cols += [col for col in df.columns if col.startswith('bert_emb_')]
    df_base = df[base_cols]
    df_base.to_csv(output_file, index=False, compression='gzip')
    print(f"\nData preprocessing complete. Processed data saved to '{output_file}' (gzipped, with Text for TF-IDF).")
    print(f"Final dataset shape (saved): {df_base.shape}")
    print(f"BERT used: {bert_used}")
    print(f"TF-IDF vectorizer saved separately for training.")

    return df  # Return full df if needed, but saved is base + text

if __name__ == "__main__":
    # Recommended for first run: subsample_frac=0.1, use_bert=False (fast test, no memory issue)
    # For full: subsample_frac=1.0, use_bert=True (BERT auto-subsamples to 10%)
    preprocess_data(use_bert=False, subsample_frac=0.1, tfidf_max_features=500)