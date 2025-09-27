import pandas as pd
import psycopg2
import joblib
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# --- Database Connection Details ---
DB_NAME = "hackathon"
DB_USER = "postgres"
DB_PASS = "Post@7070" # <-- IMPORTANT: Change this
DB_HOST = "localhost"
DB_PORT = "5432"

# --- Helper Functions (copied from app.py) ---
def get_data_for_modeling():
    conn = psycopg2.connect(database=DB_NAME, user=DB_USER, password=DB_PASS, host=DB_HOST, port=DB_PORT)
    sql_query = """
        SELECT
            c.customer_id, c.age, c.gender, c.country, c.signup_date,
            MAX(o.last_purchase_date) as last_seen,
            COUNT(o.order_id) as total_orders,
            SUM(o.quantity) as total_items_purchased,
            SUM(o.unit_price * o.quantity) as total_spend,
            AVG(o.ratings) as average_rating,
            SUM(o.cancellations_count) as total_cancellations
        FROM customers c JOIN orders o ON c.customer_id = o.customer_id
        GROUP BY c.customer_id;
    """
    df = pd.read_sql(sql_query, conn)
    conn.close()
    return df

def feature_engineering(df):
    df['signup_date'] = pd.to_datetime(df['signup_date'])
    df['last_seen'] = pd.to_datetime(df['last_seen'])
    today = datetime.now()
    df['days_since_last_purchase'] = (today - df['last_seen']).dt.days
    df['customer_tenure'] = (df['last_seen'] - df['signup_date']).dt.days
    df['avg_order_value'] = df['total_spend'] / df['total_orders']
    df.fillna(0, inplace=True)
    df.replace([float('inf'), float('-inf')], 0, inplace=True)
    return df

# --- Main Analysis Function ---
def analyze_and_plot_churn_trends():
    """Analyzes and plots churn trends over time."""
    # 1. Load Model
    model_data = joblib.load('churn_model.pkl')
    churn_model = model_data['model']
    model_columns = model_data['columns']
    
    # 2. Fetch and prepare data
    customer_df = get_data_for_modeling()
    customer_df_featured = feature_engineering(customer_df)
    
    # 3. Predict churn for all customers
    df_predict = pd.get_dummies(customer_df_featured, columns=['gender', 'country'], drop_first=True)
    df_predict_aligned = df_predict.reindex(columns=model_columns, fill_value=0)
    
    # Use predict() to get the final 0 or 1 label
    customer_df_featured['predicted_churn'] = churn_model.predict(df_predict_aligned[model_columns])
    
    # 4. Analyze trends
    # Set the 'last_seen' as the index for time-series analysis
    df_time = customer_df_featured.set_index('last_seen')
    
    # Resample by month and sum the number of churners
    monthly_churn = df_time['predicted_churn'].resample('M').sum()
    
    # 5. Plot the results
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(monthly_churn.index, monthly_churn.values, marker='o', linestyle='-', color='crimson')
    
    # Formatting the plot
    ax.set_title('Monthly Customer Churn Trends', fontsize=16)
    ax.set_ylabel('Number of Churned Customers')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save the plot to a file
    plt.savefig('churn_trends.png')
    print("\nSuccess: Churn trend analysis complete. Plot saved to 'churn_trends.png'")

if __name__ == '__main__':
    analyze_and_plot_churn_trends()