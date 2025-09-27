import pandas as pd
import psycopg2
import numpy as np
from datetime import datetime
import joblib
import warnings

warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report
from imblearn.over_sampling import SMOTE

# --- Database Connection Details ---
DB_NAME = "hackathon"
DB_USER = "postgres"
DB_PASS = "Post@7070" 
DB_HOST = "localhost"
DB_PORT = "5432"

# --- Main Data Processing and Training Functions ---

def get_aggregated_data():
    """Fetches and aggregates data from the database, similar to the notebook's logic."""
    conn = None
    try:
        conn = psycopg2.connect(
            database=DB_NAME, user=DB_USER, password=DB_PASS, host=DB_HOST, port=DB_PORT
        )
        print("Success: Database connection successful.")
        
        # This SQL query aggregates data for each customer, matching the notebook's intent
        sql_query = """
            SELECT
                c.customer_id,
                c.age,
                c.gender,
                c.country,
                MIN(c.signup_date) as signup_date,
                MAX(o.last_purchase_date) as last_purchase_date,
                COUNT(o.order_id) as purchase_count,
                SUM(o.quantity) as total_items_purchased,
                SUM(o.unit_price * o.quantity) as total_spend,
                AVG(o.ratings) as avg_rating,
                SUM(o.cancellations_count) as total_cancellations,
                MAX(o.subscription_status) as subscription_status -- Takes the most recent status
            FROM
                customers c
            JOIN
                orders o ON c.customer_id = o.customer_id
            GROUP BY
                c.customer_id, c.age, c.gender, c.country;
        """
        
        df = pd.read_sql(sql_query, conn)
        print("Success: Data loaded and aggregated into DataFrame.")
        return df

    except Exception as e:
        print("Error: Could not fetch data.")
        print(e)
        return None
    finally:
        if conn is not None:
            conn.close()

def feature_engineering_and_labeling(df):
    """Applies the feature engineering and churn labeling logic from the Colab notebook."""
    
    # 1. Date handling and Recency/Tenure features
    df['signup_date'] = pd.to_datetime(df['signup_date'], errors='coerce')
    df['last_purchase_date'] = pd.to_datetime(df['last_purchase_date'], errors='coerce')

    # --- CRITICAL FIX: Use the same fixed date as the notebook ---
    TODAY = datetime(2025, 9, 27)
    df['days_since_last_purchase'] = (TODAY - df['last_purchase_date']).dt.days.fillna(9999)
    df['tenure_days'] = (TODAY - df['signup_date']).dt.days.fillna(-1)

    # ... (the rest of the function is the same)
    
    # 2. Fill missing numeric/categorical defaults
    numeric_cols = df.select_dtypes(include=np.number).columns
    for c in numeric_cols:
        df[c] = df[c].fillna(df[c].median())
    
    # 3. Human-like feature engineering
    df['avg_spend_per_order'] = df['total_spend'] / df['purchase_count'].replace(0, 1)
    df['purchases_per_year'] = (df['purchase_count'] * 365) / (df['tenure_days'] + 1)
    
    # 4. Define Churn Label based on notebook rules
    def derive_status(row):
        ss = str(row.get('subscription_status', '')).lower()
        days = row.get('days_since_last_purchase', np.nan)
        if ss == 'cancelled' or (pd.notna(days) and days > 365):
            return 'churned'
        return 'active' # Simplified for a binary outcome

    df['status'] = df.apply(derive_status, axis=1)
    df['churn'] = (df['status'] == 'churned').astype(int)

    print("Success: Feature engineering and labeling complete.")
    print("Churn distribution:\n", df['churn'].value_counts(normalize=True))
    return df

def train_and_save_model(df):
    """Prepares data, trains the Random Forest model, and saves it."""
    
    # 1. Define features and target, excluding identifiers and leak-prone columns
    features_to_use = [
        'age', 'days_since_last_purchase', 'tenure_days', 'purchase_count',
        'total_spend', 'avg_spend_per_order', 'total_cancellations', 'avg_rating',
        'purchases_per_year'
    ]
    categorical_features = ['gender', 'country']
    target = 'churn'

    # 2. One-hot encode categorical features
    df_model = pd.get_dummies(df, columns=categorical_features, drop_first=True)
    
    # Get final list of feature columns after encoding
    final_feature_columns = features_to_use + [col for col in df_model.columns if any(cat in col for cat in categorical_features)]
    
    # Ensure all feature columns exist
    for col in final_feature_columns:
        if col not in df_model.columns:
            df_model[col] = 0 # Add missing column if a category was not in the data
            
    X = df_model[final_feature_columns]
    y = df_model[target]

    # 3. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

    # 4. Scale numeric features
    scaler = StandardScaler()
    X_train[features_to_use] = scaler.fit_transform(X_train[features_to_use])
    X_test[features_to_use] = scaler.transform(X_test[features_to_use])

    # 5. Handle class imbalance with SMOTE
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
    print(f"SMOTE applied. New train shape: {X_train_res.shape}")

    # 6. Train the Random Forest model
    model = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
    model.fit(X_train_res, y_train_res)
    
    # 7. Evaluate the model
    y_prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    print(f"\nModel Evaluation (Random Forest) ROC-AUC: {auc:.4f}")
    print("Classification Report:\n", classification_report(y_test, model.predict(X_test)))

    # 8. Save the model, scaler, and columns
    model_data_package = {
        'model': model,
        'scaler': scaler,
        'numeric_columns': features_to_use,
        'model_columns': final_feature_columns
    }
    joblib.dump(model_data_package, 'churn_model.pkl')
    print("\nSuccess: New Random Forest model saved to 'churn_model.pkl'")

# --- Main Execution Block ---
if __name__ == '__main__':
    # Step 1: Get aggregated data from the database
    customer_df = get_aggregated_data()
    
    if customer_df is not None:
        # Step 2: Apply feature engineering and create churn label
        customer_df_featured = feature_engineering_and_labeling(customer_df)
        
        # Step 3: Train and save the new model
        train_and_save_model(customer_df_featured)