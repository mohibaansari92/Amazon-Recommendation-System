# main.py - Runs the entire pipeline
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Suppress TensorFlow warnings (optional)

from data_preprocessing import preprocess_data
from data_visualization import visualize_data
from model_training import train_model

if __name__ == "__main__":
    try:
        print("=== Step 1: Data Preprocessing ===")
        # Safe defaults: 10% subsample, no BERT, 300 TF-IDF features (avoids memory issues)
        preprocess_data(use_bert=False, subsample_frac=0.1, tfidf_max_features=300)
        
        print("\n=== Step 2: Data Visualization ===")
        visualize_data()
        
        print("\n=== Step 3: Model Training and Evaluation ===")
        # Safe defaults: XGBoost, no SMOTE (avoids OOM), 20% tuning sample
        train_model(use_xgboost=True, use_smote=False, sample_frac=0.2)
        
        print("\n=== Project Complete! Check outputs, plots in 'visualizations/', and saved models. ===")
    except Exception as e:
        print(f"Pipeline failed: {e}. Check dependencies and data files.")