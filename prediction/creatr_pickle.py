# 0. Imports
import numpy as np
import pandas as pd
import datetime as dt
import warnings
import pickle
warnings.filterwarnings('ignore')

# plotting
import matplotlib.pyplot as plt
import seaborn as sns

# ML & preprocessing
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.metrics import precision_score

# Try import XGBoost (optional)
try:
    from xgboost import XGBClassifier
    xgb_available = True
except Exception:
    xgb_available = False

# Try SHAP (optional)
try:
    import shap
    shap_available = True
except Exception:
    shap_available = False

# ==========================================================
# Helper functions
# ==========================================================
def target_encode_kfold(train_series, target_series, test_series=None, n_splits=5, smoothing=10, random_state=42):
    """
    K-fold target encoding (returns enc_train, enc_test)
    enc_train: out-of-fold target-encoded values for train
    enc_test: mapping from full train to test (if test_series provided) else None
    smoothing: larger -> moves categories with few observations to global mean
    """
    df_temp = pd.DataFrame({'col': train_series.astype(str).values, 'target': target_series.values})
    oof = pd.Series(index=df_temp.index, dtype=float)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    global_mean = df_temp['target'].mean()

    for train_idx, val_idx in skf.split(df_temp, df_temp['target']):
        tr = df_temp.iloc[train_idx]
        val = df_temp.iloc[val_idx]
        agg = tr.groupby('col')['target'].agg(['count','mean'])
        counts = agg['count']
        means = agg['mean']
        # smoothing
        smooth = (counts * means + smoothing * global_mean) / (counts + smoothing)
        mapping = smooth.to_dict()
        oof.iloc[val_idx] = val['col'].map(mapping).fillna(global_mean)

    # For test series, compute mapping on full train
    enc_test = None
    if test_series is not None:
        full_agg = df_temp.groupby('col')['target'].agg(['count','mean'])
        counts = full_agg['count']
        means = full_agg['mean']
        smooth = (counts * means + smoothing * global_mean) / (counts + smoothing)
        mapping = smooth.to_dict()
        enc_test = pd.Series(test_series.astype(str).map(mapping).fillna(global_mean).values, index=test_series.index)

    return oof.values, enc_test

def freq_encode(series):
    vc = series.astype(str).value_counts(dropna=False)
    freq_map = (vc / vc.sum()).to_dict()
    return series.astype(str).map(freq_map)

def precision_at_k(y_true, y_score, k=10):
    # y_score: predicted probabilities (higher = more likely churn)
    df = pd.DataFrame({'y_true': y_true, 'y_score': y_score})
    df_sorted = df.sort_values('y_score', ascending=False).head(k)
    return df_sorted['y_true'].sum() / k

# ==========================================================
# 1. Load raw file
# ==========================================================
DATA_PATH = "./customer.xls" # Adjust as needed
df_raw = pd.read_excel(DATA_PATH)

print("Raw shape:", df_raw.shape)

# ==========================================================
# 2. Basic cleaning & cast dates
# ==========================================================
df_raw = df_raw.drop_duplicates().reset_index(drop=True)
for c in ['signup_date','last_purchase_date']:
    if c in df_raw.columns:
        df_raw[c] = pd.to_datetime(df_raw[c], errors='coerce')

# ==========================================================
# 3. Transaction -> Customer aggregation (if needed)
# ==========================================================
customer_col = None
for candidate in ['customer_id','CustomerID','customerID','cust_id','user_id']:
    if candidate in df_raw.columns:
        customer_col = candidate
        break
if customer_col is None:
    raise ValueError("No customer identifier column found.")

if df_raw[customer_col].duplicated().any():
    print("Aggregating to customer-level features.")
    df_raw['order_amount'] = df_raw.get('unit_price', 0) * df_raw.get('quantity', 0)
    agg_funcs = {
        'order_amount': ['sum','mean','count'], 'unit_price': ['mean'], 'quantity': ['sum','mean'],
        'cancellations_count': 'sum', 'Ratings': 'mean', 'signup_date': 'min', 'last_purchase_date': 'max',
        'product_id': lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan,
        'category': lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan,
        'gender': lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan,
    }
    agg_used = {k:v for k,v in agg_funcs.items() if k in df_raw.columns}
    customer_agg = df_raw.groupby(customer_col).agg(agg_used)
    customer_agg.columns = ['_'.join(col).strip() for col in customer_agg.columns.values]
    customer_agg = customer_agg.reset_index()
    rename_map = {
        'order_amount_sum': 'total_spend', 'order_amount_count': 'purchase_count',
        'cancellations_count_sum': 'total_cancellations', 'Ratings_mean': 'avg_rating',
        'last_purchase_date_max': 'last_purchase_date', 'signup_date_min': 'signup_date',
        'product_id_<lambda>': 'top_product_id', 'category_<lambda>': 'top_category', 'gender_<lambda>': 'gender_mode'
    }
    customer_agg = customer_agg.rename(columns=rename_map)
    df = customer_agg.copy()
else:
    print("Each row is already customer-level.")
    df = df_raw.copy()

# ==========================================================
# 4. Canonical date columns & recency/tenure
# ==========================================================
for c in ['signup_date','last_purchase_date']:
    if c in df.columns:
        df[c] = pd.to_datetime(df[c], errors='coerce')

TODAY = pd.Timestamp.today().normalize()
if 'last_purchase_date' in df.columns:
    df['days_since_last_purchase'] = (TODAY - df['last_purchase_date']).dt.days
if 'signup_date' in df.columns:
    df['tenure_days'] = (TODAY - df['signup_date']).dt.days
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
df[num_cols] = df[num_cols].fillna(0)

# ==========================================================
# 5. Define churn target
# ==========================================================
def status_from_row(row):
    days = row.get('days_since_last_purchase', np.nan)
    ss = str(row.get('subscription_status_mode', '')).lower()
    if ss == 'cancelled' or (pd.notna(days) and days > 180):
        return 'churned'
    return 'active' # Simplified for clarity

df['status'] = df.apply(status_from_row, axis=1)
df['churn'] = (df['status'] == 'churned').astype(int)

# ==========================================================
# 6. Human-like feature engineering + formula-based score
# ==========================================================
df['tenure_years'] = df['tenure_days'] / 365.0
df['purchases_per_year'] = df.apply(lambda r: r.get('purchase_count', 1) / r['tenure_years'] if r['tenure_years'] > 0 else 0, axis=1)
df['rating_for_score'] = df.get('avg_rating', df.get('Ratings', 3.0))

norm_cols = [c for c in ['days_since_last_purchase','total_cancellations','purchases_per_year','rating_for_score'] if c in df.columns]
if norm_cols:
    df[[c + "_norm" for c in norm_cols]] = MinMaxScaler().fit_transform(df[norm_cols].fillna(0))

w = {'days_since_last_purchase_norm': 0.5, 'total_cancellations_norm': 0.2, 'purchases_per_year_norm': -0.15, 'rating_for_score_norm': -0.15}
comb = np.zeros(df.shape[0])
for col, weight in w.items():
    if col in df.columns:
        comb += weight * df[col].values
df['churn_score_formula'] = (comb - comb.min()) / (comb.max() - comb.min()) if comb.max() > comb.min() else 0

# ==========================================================
# 7. Encoding strategy & feature list
# ==========================================================
numeric_feats = [c for c in ['age','total_spend','purchase_count','days_since_last_purchase','tenure_days','total_cancellations','avg_rating','churn_score_formula'] if c in df.columns]
cat_onehot = [c for c in ['gender_mode', 'gender', 'top_category', 'category'] if c in df.columns][:2] # Simplified
cat_freq = [c for c in ['top_product_id'] if c in df.columns]
cat_target = [] # Disabling for simplicity in pickle example

# ==========================================================
# 8. Train/test split
# ==========================================================
working = df.copy().reset_index(drop=True)
X_train, X_test, y_train, y_test = train_test_split(working, working['churn'], test_size=0.25, random_state=42, stratify=working['churn'])

# ==========================================================
# 9. Apply encodings
# ==========================================================
if cat_onehot:
    combined_onehot = pd.get_dummies(pd.concat([X_train[cat_onehot], X_test[cat_onehot]], axis=0), dummy_na=False)
    oh_train = combined_onehot.iloc[:X_train.shape[0], :]
    oh_test = combined_onehot.iloc[X_train.shape[0]:, :]
else:
    oh_train = pd.DataFrame(index=X_train.index)
    oh_test = pd.DataFrame(index=X_test.index)

freq_maps = {}
for col in cat_freq:
    freq_maps[col] = X_train[col].astype(str).value_counts(normalize=True).to_dict()
    X_train[col + '_freq'] = X_train[col].astype(str).map(freq_maps[col]).fillna(0)
    X_test[col + '_freq'] = X_test[col].astype(str).map(freq_maps[col]).fillna(0)

features_final = numeric_feats + [c + '_freq' for c in cat_freq]
X_train_enc = pd.concat([X_train[features_final].reset_index(drop=True), oh_train.reset_index(drop=True)], axis=1)
X_test_enc  = pd.concat([X_test[features_final].reset_index(drop=True), oh_test.reset_index(drop=True)], axis=1)
X_train_enc = X_train_enc.loc[:,~X_train_enc.columns.duplicated()]
X_test_enc  = X_test_enc.loc[:,~X_test_enc.columns.duplicated()]

# ==========================================================
# 10. Scale numeric columns
# ==========================================================
encoded_num_cols = [c for c in numeric_feats if c in X_train_enc.columns]
scaler = StandardScaler()
if encoded_num_cols:
    X_train_enc[encoded_num_cols] = scaler.fit_transform(X_train_enc[encoded_num_cols])
    X_test_enc[encoded_num_cols] = scaler.transform(X_test_enc[encoded_num_cols])

# ==========================================================
# 11. Train & evaluate models
# ==========================================================
models = {}
results = {}
lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
lr.fit(X_train_enc, y_train)
models['LogisticRegression'] = lr
results['LogisticRegression'] = roc_auc_score(y_test, lr.predict_proba(X_test_enc)[:,1])

rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
rf.fit(X_train_enc, y_train)
models['RandomForest'] = rf
results['RandomForest'] = roc_auc_score(y_test, rf.predict_proba(X_test_enc)[:,1])

if xgb_available:
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', scale_pos_weight=scale_pos_weight, random_state=42)
    xgb.fit(X_train_enc, y_train)
    models['XGBoost'] = xgb
    results['XGBoost'] = roc_auc_score(y_test, xgb.predict_proba(X_test_enc)[:,1])

best_model_name = max(results, key=results.get)
best_model = models[best_model_name]
print(f"\nSelected best model: {best_model_name} with ROC-AUC: {results[best_model_name]:.4f}")


# ==========================================================
# 17a. NEW: Save the complete prediction pipeline to a .pkl file
# ==========================================================
print("\n--- Creating prediction pipeline for deployment ---")

# Bundle all necessary components into a single dictionary.
# This makes loading and using the pipeline much easier later.
churn_pipeline = {
    'model': best_model,
    'scaler': scaler if encoded_num_cols else None,
    'one_hot_columns': oh_train.columns.tolist(), # The exact column names from one-hot encoding
    'frequency_maps': freq_maps, # The dictionary of frequency encodings
    'numeric_features': numeric_feats, # List of numeric features to be scaled
    'one_hot_features': cat_onehot, # List of features to be one-hot encoded
    'frequency_features': cat_freq, # List of features to be frequency encoded
    'final_feature_order': X_train_enc.columns.tolist() # The final order of columns the model expects
}

# Define the filename for the .pkl file
PKL_FILENAME = "churn_prediction_pipeline.pkl"

# Use 'wb' (write binary) mode to save the file
with open(PKL_FILENAME, 'wb') as file:
    pickle.dump(churn_pipeline, file)

print(f"✅ Successfully saved the complete pipeline to {PKL_FILENAME}")
print("   This file contains the model, scaler, and all encoding information.")

# ==========================================================
# (Original sections 12-18 can still run for analysis)
# ==========================================================
# For brevity, these are omitted but your original code would continue here for analysis.
print("\n--- Original analysis complete ---")