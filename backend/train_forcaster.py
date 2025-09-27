import pandas as pd
import psycopg2
import joblib
import warnings
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings('ignore')

# --- Database Connection Details ---
DB_NAME = "hackathon"
DB_USER = "postgres"
DB_PASS = "Post@7070" # <-- IMPORTANT: Change this
DB_HOST = "localhost"
DB_PORT = "5432"

def get_sales_data():
    """Fetches raw order data to create a sales time-series."""
    conn = None
    try:
        conn = psycopg2.connect(
            database=DB_NAME, user=DB_USER, password=DB_PASS, host=DB_HOST, port=DB_PORT
        )
        print("Success: Database connection successful.")
        
        # This query gets the date and total amount for each order
        sql_query = """
            SELECT
                last_purchase_date,
                (unit_price * quantity) as order_amount
            FROM
                orders;
        """
        
        df = pd.read_sql(sql_query, conn)
        print("Success: Sales data loaded from database.")
        return df

    except Exception as e:
        print(f"Error: Could not fetch sales data. {e}")
        return None
    finally:
        if conn is not None:
            conn.close()

def train_and_save_forecaster(df):
    """Prepares time-series data and trains and saves a SARIMAX model."""
    
    # 1. Prepare the time-series data
    df['last_purchase_date'] = pd.to_datetime(df['last_purchase_date'], errors='coerce')
    df = df.dropna(subset=['last_purchase_date', 'order_amount'])
    
    # Aggregate sales by day
    sales_daily = df.groupby('last_purchase_date')['order_amount'].sum().asfreq('D').fillna(0)
    
    if len(sales_daily) < 14:
        print("Error: Not enough daily data to train the forecasting model.")
        return

    # 2. Train the SARIMAX model (using parameters from the notebook)
    print("Training SARIMAX model... this may take a moment.")
    try:
        # The model uses weekly seasonality (m=7)
        model = SARIMAX(sales_daily, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
        results = model.fit(disp=False)
        
        # 3. Save the trained model
        joblib.dump(results, 'sales_forecaster.pkl')
        print("\nSuccess: Sales forecasting model trained and saved to 'sales_forecaster.pkl'")

    except Exception as e:
        print(f"Error during model training: {e}")

# --- Main Execution Block ---
if __name__ == '__main__':
    sales_df = get_sales_data()
    
    if sales_df is not None:
        train_and_save_forecaster(sales_df)