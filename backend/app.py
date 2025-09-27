import psycopg2
import pandas as pd
import numpy as np
import joblib
from flask import Flask, jsonify, request
from flask_cors import CORS
from decimal import Decimal
import datetime

# --- Database Connection Details ---
DB_NAME = "hackathon"
DB_USER = "postgres"
DB_PASS = "Post@7070" # <-- IMPORTANT: Change this
DB_HOST = "localhost"
DB_PORT = "5432"

# --- Load the saved model package ---
try:
    model_package = joblib.load('churn_model.pkl')
    churn_model = model_package['model']
    scaler = model_package['scaler']
    numeric_columns = model_package['numeric_columns']
    model_columns = model_package['model_columns']
    print("Success: New Random Forest model package loaded.")
except FileNotFoundError:
    print("Error: 'churn_model.pkl' not found. Please run the train_model.py script first.")
    exit()

try:
    sales_forecaster = joblib.load('sales_forecaster.pkl')
    print("Success: SARIMAX (Sales) model loaded.")
except FileNotFoundError:
    print("Warning: 'sales_forecaster.pkl' not found. Sales forecasting will not work.")
    sales_forecaster = None

# Initialize the Flask application
app = Flask(__name__)
CORS(app)

# --- Helper Functions (used by multiple endpoints) ---
def json_converter(obj):
    if isinstance(obj, Decimal): return float(obj)
    if isinstance(obj, (datetime.datetime, datetime.date)): return obj.isoformat()
    raise TypeError ("Type %s not serializable" % type(obj))

def get_aggregated_data():
    conn = psycopg2.connect(database=DB_NAME, user=DB_USER, password=DB_PASS, host=DB_HOST, port=DB_PORT)
    sql_query = """
        SELECT
            c.customer_id, c.age, c.gender, c.country,
            MIN(c.signup_date) as signup_date,
            MAX(o.last_purchase_date) as last_purchase_date,
            COUNT(o.order_id) as purchase_count,
            SUM(o.quantity) as total_items_purchased,
            SUM(o.unit_price * o.quantity) as total_spend,
            AVG(o.ratings) as avg_rating,
            SUM(o.cancellations_count) as total_cancellations,
            MAX(o.subscription_status) as subscription_status
        FROM customers c JOIN orders o ON c.customer_id = o.customer_id
        GROUP BY c.customer_id, c.age, c.gender, c.country;
    """
    df = pd.read_sql(sql_query, conn)
    conn.close()
    return df

def feature_engineering_for_prediction(df):
    """Applies the same feature engineering as the training script."""
    df['signup_date'] = pd.to_datetime(df['signup_date'], errors='coerce')
    df['last_purchase_date'] = pd.to_datetime(df['last_purchase_date'], errors='coerce')
    
    # --- CRITICAL FIX: Use the same fixed date as the notebook ---
    TODAY = datetime.datetime(2025, 9, 27)
    df['days_since_last_purchase'] = (TODAY - df['last_purchase_date']).dt.days.fillna(9999)
    df['tenure_days'] = (TODAY - df['signup_date']).dt.days.fillna(-1)
    df['avg_spend_per_order'] = df['total_spend'] / df['purchase_count'].replace(0, 1)
    df['purchases_per_year'] = (df['purchase_count'] * 365) / (df['tenure_days'] + 1)
    
    # Fill any remaining NaNs in numeric columns
    num_cols = df.select_dtypes(include=np.number).columns
    for c in num_cols:
        df[c] = df[c].fillna(df[c].median())
        
    return df

# --- API Endpoints ---
@app.route('/api/orders', methods=['GET'])
def get_orders():
    # ... (This endpoint is restored) ...
    conn = psycopg2.connect(database=DB_NAME, user=DB_USER, password=DB_PASS, host=DB_HOST, port=DB_PORT)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM orders;")
    orders_data = cursor.fetchall()
    column_names = [desc[0] for desc in cursor.description]
    orders_list = []
    for row in orders_data:
        order_dict = dict(zip(column_names, row))
        for key, value in order_dict.items():
            order_dict[key] = json_converter(value) if isinstance(value, (Decimal, datetime.date)) else value
        orders_list.append(order_dict)
    cursor.close()
    conn.close()
    return jsonify(orders_list)

@app.route('/api/predict_churn', methods=['GET'])
def predict_churn():
    # ... (This is your updated prediction endpoint) ...
    try:
        count = request.args.get('count', default=10, type=int)
        customer_df = get_aggregated_data()
        customer_df_featured = feature_engineering_for_prediction(customer_df)
        df_predict = pd.get_dummies(customer_df_featured, columns=['gender', 'country'], drop_first=True)
        df_predict_aligned = df_predict.reindex(columns=model_columns, fill_value=0)
        df_predict_aligned[numeric_columns] = scaler.transform(df_predict_aligned[numeric_columns])
        churn_probabilities = churn_model.predict_proba(df_predict_aligned[model_columns])[:, 1]
        results_df = customer_df_featured[['customer_id']].copy()
        results_df['churn_probability'] = churn_probabilities
        top_n_churners = results_df.sort_values(by='churn_probability', ascending=False).head(count)
        return jsonify(top_n_churners.to_dict(orient='records'))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/churn_trends', methods=['GET'])
def get_churn_trends():
    # ... (This endpoint is restored) ...
    try:
        customer_df = get_aggregated_data()
        customer_df_featured = feature_engineering_for_prediction(customer_df)
        df_predict = pd.get_dummies(customer_df_featured, columns=['gender', 'country'], drop_first=True)
        df_predict_aligned = df_predict.reindex(columns=model_columns, fill_value=0)
        df_predict_aligned[numeric_columns] = scaler.transform(df_predict_aligned[numeric_columns])
        customer_df_featured['predicted_churn'] = churn_model.predict(df_predict_aligned[model_columns])
        df_time = customer_df_featured.set_index('last_purchase_date')
        monthly_churn = df_time['predicted_churn'].resample('M').sum()
        trend_data = {
            "months": monthly_churn.index.strftime('%Y-%m').tolist(),
            "churn_counts": monthly_churn.values.tolist()
        }
        return jsonify(trend_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/churn_segmentation', methods=['GET'])
def get_churn_segmentation():
    # ... (This endpoint is restored) ...
    try:
        customer_df = get_aggregated_data()
        customer_df_featured = feature_engineering_for_prediction(customer_df)
        df_predict = pd.get_dummies(customer_df_featured, columns=['gender', 'country'], drop_first=True)
        df_predict_aligned = df_predict.reindex(columns=model_columns, fill_value=0)
        df_predict_aligned[numeric_columns] = scaler.transform(df_predict_aligned[numeric_columns])
        churn_probabilities = churn_model.predict_proba(df_predict_aligned[model_columns])[:, 1]
        def assign_segment(prob):
            if prob < 0.3: return 'Low Risk'
            elif prob < 0.7: return 'Medium Risk'
            else: return 'High Risk'
        segments = pd.Series(churn_probabilities).apply(assign_segment)
        segment_counts = segments.value_counts().to_dict()
        return jsonify(segment_counts)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route('/api/sales_forecast', methods=['GET'])
def get_sales_forecast():
    """Generates a sales forecast for a specified number of future days."""
    if sales_forecaster is None:
        return jsonify({"error": "Sales forecasting model not loaded."}), 500
        
    try:
        # Get the number of days to forecast from the URL, default to 30
        days_to_forecast = request.args.get('days', default=30, type=int)

        # Use the loaded model to get the forecast
        forecast_results = sales_forecaster.get_forecast(steps=days_to_forecast)
        
        # Get the mean prediction
        predicted_mean = forecast_results.predicted_mean
        
        # Get the confidence interval to show a range of uncertainty
        confidence_interval = forecast_results.conf_int()

        # Format the data for the frontend chart
        forecast_data = {
            "dates": predicted_mean.index.strftime('%Y-%m-%d').tolist(),
            "predicted_sales": predicted_mean.values.tolist(),
            "confidence_lower": confidence_interval.iloc[:, 0].values.tolist(),
            "confidence_upper": confidence_interval.iloc[:, 1].values.tolist(),
        }
        
        return jsonify(forecast_data)

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

@app.route('/api/top_products', methods=['GET'])
def get_top_products():
    """Calculates the top 10 products with the highest historical sales."""
    conn = None
    try:
        conn = psycopg2.connect(
            database=DB_NAME, user=DB_USER, password=DB_PASS, host=DB_HOST, port=DB_PORT
        )
        
        # This query calculates the total sales for each product and orders them
        sql_query = """
            SELECT
                p.product_name,
                p.category,
                SUM(o.unit_price * o.quantity) as total_sales
            FROM
                products p
            JOIN
                orders o ON p.product_id = o.product_id
            GROUP BY
                p.product_name, p.category
            ORDER BY
                total_sales DESC
            LIMIT 10;
        """
        
        df = pd.read_sql(sql_query, conn)
        
        # Convert the DataFrame to a list of dictionaries for the JSON response
        top_products_list = df.to_dict(orient='records')
        
        return jsonify(top_products_list)

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if conn is not None:
            conn.close()

# ... (add this at the end of your app.py, before the if __name__ == '__main__': line)

@app.route('/api/full_sales_view', methods=['GET'])
def get_full_sales_view():
    """
    Provides the last 180 days of historical sales and a future forecast.
    """
    if sales_forecaster is None:
        return jsonify({"error": "Sales forecasting model not loaded."}), 500
        
    conn = None
    try:
        # Part 1: Fetch Recent Historical Data
        conn = psycopg2.connect(
            database=DB_NAME, user=DB_USER, password=DB_PASS, host=DB_HOST, port=DB_PORT
        )
        
        # --- MODIFIED SQL QUERY ---
        # This query now fetches only the sales from the last 180 days
        # relative to the most recent sale in the database.
        sql_query = """
            SELECT 
                last_purchase_date, 
                SUM(unit_price * quantity) as order_amount
            FROM 
                orders
            WHERE 
                last_purchase_date >= (SELECT MAX(last_purchase_date) - INTERVAL '180 days' FROM orders)
            GROUP BY
                last_purchase_date;
        """
        
        df = pd.read_sql(sql_query, conn)
        df['last_purchase_date'] = pd.to_datetime(df['last_purchase_date'], errors='coerce')
        historical_sales = df.groupby('last_purchase_date')['order_amount'].sum().asfreq('D').fillna(0)

        # Part 2: Generate Forecast
        days_to_forecast = request.args.get('days', default=90, type=int)
        forecast_results = sales_forecaster.get_forecast(steps=days_to_forecast)
        predicted_mean = forecast_results.predicted_mean

        # Part 3: Combine and Format Data
        full_view_data = {
            "historical_dates": historical_sales.index.strftime('%Y-%m-%d').tolist(),
            "historical_sales": historical_sales.values.tolist(),
            "forecast_dates": predicted_mean.index.strftime('%Y-%m-%d').tolist(),
            "forecast_sales": predicted_mean.values.tolist(),
        }
        
        return jsonify(full_view_data)

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if conn is not None:
            conn.close()

if __name__ == '__main__':
    app.run(debug=True, threaded=False)