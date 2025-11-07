import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier  # Fallback
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from xgboost import XGBClassifier  # Default: pip install xgboost
from imblearn.over_sampling import SMOTE  # For imbalance: pip install imbalanced-learn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from scipy.sparse import hstack, csr_matrix
from sklearn.preprocessing import StandardScaler  # Optional (not needed for trees)
import warnings
warnings.filterwarnings('ignore')
import gc

def train_model(use_xgboost=True, use_smote=True, sample_frac=0.3):
    # Load processed data (gzipped)
    try:
        df = pd.read_csv('processed_data.csv.gz', compression='gzip')
        print("Processed data loaded for model training.")
        print(f"Dataset shape: {df.shape}")
    except FileNotFoundError:
        print("Error: 'processed_data.csv.gz' not found. Run 'data_preprocessing.py' first.")
        return

    # Load TF-IDF vectorizer and apply on-the-fly (sparse)
    try:
        tfidf = joblib.load('tfidf_vectorizer.pkl')
        X_tfidf = tfidf.transform(df['Text'])  # Sparse, no memory issue
        print(f"TF-IDF applied: sparse shape {X_tfidf.shape}")
    except FileNotFoundError:
        print("TF-IDF vectorizer not found. Run preprocessing first.")
        return

    # Define features
    numeric_features = ['user_id_encoded', 'product_id_encoded', 'HelpfulnessRatio',
                        'review_year', 'review_month', 'review_dayofweek',
                        'review_length', 'sentiment_score', 'user_avg_score', 'product_avg_score']
    bert_features = [col for col in df.columns if col.startswith('bert_emb_')]
    if bert_features:
        numeric_features += bert_features
        print(f"BERT features included: {len(bert_features)} dims")
    target = 'Score'

    # Check features exist
    missing_features = [f for f in numeric_features if f not in df.columns]
    if missing_features:
        print(f"Warning: Missing features {missing_features}. Using available ones.")
        numeric_features = [f for f in numeric_features if f in df.columns]

    # Prepare X_numeric (dense)
    X_numeric = df[numeric_features]

    # Combine numeric (dense) + TF-IDF (sparse)
    X = hstack([X_numeric, X_tfidf])
    y = df[target].astype(int) - 1  # FIX: Map 1-5 to 0-4 for XGBoost/scikit-learn compatibility

    # Split data (stratified for imbalance)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    # Reset indices to consecutive 0-based for safe positional subsampling
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    print(f"\nTraining data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")
    print(f"Label range: {y.min()} to {y.max()} (0-4 mapped from 1-5)")

    # Optional: SMOTE for imbalance (with aggressive subsampling to avoid OOM)
    smote_applied = False
    if use_smote:
        print("Applying SMOTE for class balance (with memory safeguards)...")
        try:
            smote = SMOTE(random_state=42, k_neighbors=5)
            # Aggressive subsample for SMOTE: Cap at 10k rows to prevent OOM
            max_smote_size = 10000
            if X_train.shape[0] > max_smote_size:
                smote_indices = np.random.choice(X_train.shape[0], max_smote_size, replace=False)
                X_train_smote = X_train[smote_indices].toarray()
                y_train_smote = y_train.iloc[smote_indices].values  # Positional iloc
                print(f"Subsampled train to {max_smote_size} rows for SMOTE.")
            else:
                X_train_smote = X_train.toarray()
                y_train_smote = y_train.values  # To numpy (0-4)
            X_train_res, y_train_res = smote.fit_resample(X_train_smote, y_train_smote)
            # If post-SMOTE still too large (>50k), skip and warn
            if X_train_res.shape[0] > 50000:
                print("Post-SMOTE too large; skipping to avoid OOM. Using class weights instead.")
            else:
                X_train = csr_matrix(X_train_res) if X_train_res.shape[0] < 20000 else X_train_res
                y_train = pd.Series(y_train_res)  # Back to Series (0-4)
                y_train = y_train.reset_index(drop=True)
                smote_applied = True
                print(f"After SMOTE: {X_train.shape}, class distribution: {np.bincount(y_train)}")
            gc.collect()
        except MemoryError:
            print("Memory error in SMOTE; skipping. Using class weights instead.")
        except Exception as e:
            print(f"SMOTE failed: {e}. Skipping.")
    else:
        print("SMOTE disabled; using class weights for imbalance.")

    # Subsample for tuning (use dense for compatibility; positional iloc)
    n_samples = int(len(y_train) * sample_frac)
    train_indices = np.random.choice(len(y_train), n_samples, replace=False)
    if hasattr(X_train, 'toarray'):
        X_train_sample = X_train[train_indices].toarray()
    else:
        X_train_sample = X_train[train_indices]
    y_train_sample = y_train.iloc[train_indices]  # Positional iloc (0-4)
    print(f"Using subsample for tuning: {X_train_sample.shape}")

    # Model: XGBoost (default) or RandomForest
    if use_xgboost:
        base_model = XGBClassifier(
            random_state=42, n_jobs=2, verbosity=0,
            scale_pos_weight='balanced' if not smote_applied else 1,  # Adjust if SMOTE applied
            reg_alpha=0.1, reg_lambda=1.0  # Regularization
        )
        param_dist = {
            'n_estimators': [200, 300, 500],
            'max_depth': [6, 10, 15],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }
        scoring = 'f1_weighted'
        model_name = 'XGBoost'
    else:
        base_model = RandomForestClassifier(random_state=42, n_jobs=2, class_weight='balanced')
        param_dist = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None]
        }
        scoring = 'f1_weighted'
        model_name = 'Random Forest'

    print(f"\nStarting hyperparameter tuning ({model_name})...")
    random_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_dist,
        n_iter=15,
        cv=3,
        verbose=2,
        random_state=42,
        n_jobs=2,
        scoring=scoring
    )

    try:
        random_search.fit(X_train_sample, y_train_sample)
        print("\nBest Parameters Found:")
        print(random_search.best_params_)
    except Exception as e:  # Broader catch for any tuning error
        print(f"Tuning failed: {e}. Using default model.")
        if use_xgboost:
            best_params = {'n_estimators': 300, 'max_depth': 10, 'learning_rate': 0.05, 'subsample': 0.8, 'colsample_bytree': 0.8}
        else:
            best_params = {'n_estimators': 200, 'max_depth': 10, 'min_samples_split': 5, 'min_samples_leaf': 2, 'max_features': 'sqrt'}

    # Refit on full training data (use best_params if tuning failed)
    if use_xgboost:
        best_model = XGBClassifier(
            **(random_search.best_params_ if 'random_search' in locals() and random_search.best_params_ else best_params),
            random_state=42, n_jobs=2, verbosity=0,
            scale_pos_weight='balanced' if not smote_applied else 1, reg_alpha=0.1, reg_lambda=1.0
        )
    else:
        best_model = RandomForestClassifier(
            **(random_search.best_params_ if 'random_search' in locals() and random_search.best_params_ else best_params),
            random_state=42, n_jobs=2, class_weight='balanced'
        )
    
    # Fit on full (use dense if sparse post-SMOTE is tricky)
    if hasattr(X_train, 'toarray'):
        best_model.fit(X_train.toarray(), y_train)
    else:
        best_model.fit(X_train, y_train)

    # Predictions (0-4)
    if hasattr(X_test, 'toarray'):
        y_pred = best_model.predict(X_test.toarray())
        y_pred_proba = best_model.predict_proba(X_test.toarray())
    else:
        y_pred = best_model.predict(X_test)
        y_pred_proba = best_model.predict_proba(X_test)

    # Map back to 1-5 for display/reporting
    y_test_display = y_test + 1
    y_pred_display = y_pred + 1

    # Metrics (use original 1-5 for report)
    accuracy = accuracy_score(y_test_display, y_pred_display)
    f1 = f1_score(y_test_display, y_pred_display, average='weighted')
    print(f"\n{model_name} Results (SMOTE applied: {smote_applied}):")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1-Score (weighted): {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test_display, y_pred_display, target_names=['1-Star', '2-Star', '3-Star', '4-Star', '5-Star']))

    # Confusion Matrix (labels 1-5)
    cm = confusion_matrix(y_test_display, y_pred_display)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(1,6), yticklabels=range(1,6))
    plt.title(f'{model_name} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    # Cross-Validation by Rating Group (Stratified; positional iloc; 0-4 labels)
    print(f"\nCross-Validation ({model_name}, Stratified 5-Fold):")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    # Use subsample for CV to avoid memory
    cv_size = min(10000, X_train.shape[0])  # Smaller cap
    cv_indices = np.random.choice(X_train.shape[0], cv_size, replace=False)
    if hasattr(X_train, 'toarray'):
        cv_X = X_train[cv_indices].toarray()
    else:
        cv_X = X_train[cv_indices]
    cv_y = y_train.iloc[cv_indices]  # Positional iloc (0-4)
    cv_scores = cross_val_score(best_model, cv_X, cv_y, cv=skf, scoring='f1_weighted')
    print(f"CV F1-Score (mean ± std): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Per-group F1 on test set (map back to 1-5 for names)
    per_group_f1 = f1_score(y_test_display, y_pred_display, average=None)
    class_names = ['1-Star', '2-Star', '3-Star', '4-Star', '5-Star']
    for i, (name, score) in enumerate(zip(class_names, per_group_f1)):
        print(f"{name}: F1 = {score:.4f}")

    # Plot per-group performance
    plt.figure(figsize=(8, 5))
    sns.barplot(x=class_names, y=per_group_f1, palette='viridis')
    plt.title(f'{model_name} Per-Rating Group F1-Scores')
    plt.ylabel('F1-Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Feature Importances (top 20; map to names)
    if hasattr(best_model, 'feature_importances_'):
        # Create feature names: numeric (incl. BERT) + TF-IDF terms
        tfidf_feature_names = tfidf.get_feature_names_out()
        full_feature_names = numeric_features + list(tfidf_feature_names)
        feature_importances = pd.Series(best_model.feature_importances_, index=full_feature_names).sort_values(ascending=False)
        top_features = feature_importances.head(20)
        print(f"\nTop 20 Feature Importances ({model_name}):")
        print(top_features)
        
        plt.figure(figsize=(10, 8))
        sns.barplot(x=top_features.values, y=top_features.index)
        plt.title(f'{model_name} Top Feature Importances')
        plt.xlabel('Importance')
        plt.tight_layout()
        plt.show()
    else:
        print("Feature importances not available for this model.")

    # Save model and predictions (save original 1-5 for predictions CSV)
    model_filename = f'best_model_{"xgboost" if use_xgboost else "rf"}_{"smote" if smote_applied else "noweight"}.pkl'
    joblib.dump(best_model, model_filename)
    print(f"Best model saved to '{model_filename}'.")
    
    predictions_df = pd.DataFrame({
        'Actual': y_test_display.values, 
        'Predicted': y_pred_display, 
        'Predicted_Prob': np.max(y_pred_proba, axis=1)
    })
    predictions_df.to_csv('model_predictions.csv', index=False)
    print("Predictions saved to 'model_predictions.csv'.")

    print(f"\nTraining complete! SMOTE applied: {smote_applied}, Model: {model_name}, Sample Frac: {sample_frac}")
    gc.collect()

if __name__ == "__main__":
    # Customize: use_xgboost=True for default, use_smote=False to avoid memory issues, sample_frac=0.2 for speed
    train_model(use_xgboost=True, use_smote=False, sample_frac=0.2)