import pandas as pd
import numpy as np
import pickle
import datetime as dt

def preprocess_and_predict(raw_df, pipeline_path="churn_prediction_pipeline.pkl"):
    """
    Loads a saved pipeline to preprocess raw customer data and predict churn probability.

    Args:
        raw_df (pd.DataFrame): A DataFrame containing new, raw customer data.
                               It must have columns consistent with the original training data.
        pipeline_path (str): The file path to the saved .pkl pipeline.

    Returns:
        pd.DataFrame: The input DataFrame with a new 'churn_probability' column.
    """
    # 1. Load the complete pipeline object
    try:
        with open(pipeline_path, 'rb') as f:
            pipeline = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: Pipeline file not found at '{pipeline_path}'")
        return None

    model = pipeline['model']
    scaler = pipeline['scaler']
    oh_cols = pipeline['one_hot_columns']
    freq_maps = pipeline['frequency_maps']
    num_feats = pipeline['numeric_features']
    oh_feats = pipeline['one_hot_features']
    freq_feats = pipeline['frequency_features']
    final_feature_order = pipeline['final_feature_order']

    df = raw_df.copy()

    # --- 2. Feature Engineering (Crucial Step) ---
    # a. Create date-based features
    for c in ['signup_date', 'last_purchase_date']:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors='coerce')
            
    TODAY = pd.Timestamp.today().normalize()
    if 'last_purchase_date' in df.columns:
        df['days_since_last_purchase'] = (TODAY - df['last_purchase_date']).dt.days
    if 'signup_date' in df.columns:
        df['tenure_days'] = (TODAY - df['signup_date']).dt.days

    # b. **FIX**: Ensure 'churn_score_formula' exists BEFORE it is accessed.
    # The model was trained with this feature. We add it here with a neutral default
    # value if it's not already in the input data.
    if 'churn_score_formula' not in df.columns:
        df['churn_score_formula'] = 0.5 # Neutral default

    # c. Now that all expected numeric columns exist, fill any NaNs.
    # This was the line that previously caused the error.
    df[num_feats] = df[num_feats].fillna(0)
    
    # --- 3. Apply Saved Encodings ---
    # a. One-Hot Encoding
    if oh_feats:
        df_oh = pd.get_dummies(df[oh_feats], dummy_na=False)
        df_oh = df_oh.reindex(columns=oh_cols, fill_value=0)
    else:
        df_oh = pd.DataFrame(index=df.index)

    # b. Frequency Encoding
    for col in freq_feats:
        df[col + '_freq'] = df[col].astype(str).map(freq_maps.get(col, {})).fillna(0)

    # --- 4. Assemble the Final Feature Matrix ---
    features_for_model = [c for c in num_feats if c in df.columns] + [f + '_freq' for f in freq_feats]
    df_processed = pd.concat([df[features_for_model].reset_index(drop=True), df_oh.reset_index(drop=True)], axis=1)
    df_processed = df_processed.reindex(columns=final_feature_order, fill_value=0)

    # --- 5. Scale Numeric Features ---
    if scaler and num_feats:
        cols_to_scale = [c for c in num_feats if c in df_processed.columns]
        if cols_to_scale:
            df_processed[cols_to_scale] = scaler.transform(df_processed[cols_to_scale])

    # --- 6. Make Predictions ---
    predictions = model.predict_proba(df_processed)[:, 1]

    raw_df['churn_probability'] = predictions
    return raw_df

# ==========================================================
# Example Usage
# ==========================================================
if __name__ == '__main__':
    new_customer_data = pd.DataFrame({
        'customer_id': ['CUST-001', 'CUST-002', 'CUST-003'],
        'age': [34, 25, 55],
        'gender': ['Female', 'Male', 'Male'],
        'category': ['Electronics', 'Books', 'Home Goods'], # FIX: Renamed 'top_category' to 'category' to match the trained model's expectation
        'top_product_id': ['PROD-123', 'PROD-456', 'PROD-123'],
        'total_spend': [250.75, 1200.00, 85.50],
        'purchase_count': [5, 25, 2],
        'total_cancellations': [0, 1, 2],
        'avg_rating': [4.8, 4.1, 2.5],
        'signup_date': ['2023-01-15', '2022-05-20', '2025-08-01'],
        'last_purchase_date': ['2025-09-01', '2025-03-10', '2025-08-05']
    })

    print("--- Input Data ---")
    print(new_customer_data.head())
    print("\n")

    results_df = preprocess_and_predict(new_customer_data)

    if results_df is not None:
        print("--- Churn Predictions ---")
        results_df['churn_probability'] = results_df['churn_probability'].apply(lambda x: f"{x:.2%}")
        print(results_df[['customer_id', 'churn_probability']])

