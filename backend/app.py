import psycopg2
import pandas as pd
import numpy as np
import joblib
from flask import Flask, jsonify, request
from flask_cors import CORS
from decimal import Decimal
import datetime
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from data_importer import insert_data_from_df


# from pyngrok import ngrok

# public_url = ngrok.connect(5000)
# print(public_url)

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


# In your app.py file

@app.route('/api/predict_churn', methods=['GET'])
def predict_churn():
    """Predicts the top N customers likely to churn with additional details."""
    try:
        count = request.args.get('count', default=10, type=int)

        customer_df = get_aggregated_data()
        customer_df_featured = feature_engineering_for_prediction(customer_df)
        
        df_predict = pd.get_dummies(customer_df_featured, columns=['gender', 'country'], drop_first=True)
        df_predict_aligned = df_predict.reindex(columns=model_columns, fill_value=0)
        df_predict_aligned[numeric_columns] = scaler.transform(df_predict_aligned[numeric_columns])
        
        churn_probabilities = churn_model.predict_proba(df_predict_aligned[model_columns])[:, 1]
        
        # --- CRITICAL CHANGE: Select more columns for the results ---
        results_df = customer_df_featured[[
            'customer_id', 
            'last_purchase_date', 
            'total_cancellations', 
            'subscription_status'
        ]].copy()
        results_df['churn_probability'] = churn_probabilities
        
        top_n_churners = results_df.sort_values(by='churn_probability', ascending=False).head(count)
        
        # Convert date to string for JSON compatibility
        top_n_churners['last_purchase_date'] = top_n_churners['last_purchase_date'].dt.strftime('%Y-%m-%d')
        
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

@app.route('/api/sales_kpis', methods=['GET'])
def get_sales_kpis():
    """Analyzes historical sales to find key performance indicators."""
    conn = None
    try:
        conn = psycopg2.connect(
            database=DB_NAME, user=DB_USER, password=DB_PASS, host=DB_HOST, port=DB_PORT
        )
        sql_query = """
            SELECT last_purchase_date, (unit_price * quantity) as order_amount
            FROM orders;
        """
        df = pd.read_sql(sql_query, conn)
        df['last_purchase_date'] = pd.to_datetime(df['last_purchase_date'], errors='coerce')
        df = df.dropna(subset=['last_purchase_date', 'order_amount'])

        # Calculate KPIs
        total_revenue = df['order_amount'].sum()
        avg_daily_sales = df.groupby(df['last_purchase_date'].dt.date)['order_amount'].sum().mean()
        
        # Find best and worst sales month
        monthly_sales = df.set_index('last_purchase_date').resample('M')['order_amount'].sum()
        best_month = monthly_sales.idxmax()
        best_month_sales = monthly_sales.max()
        worst_month = monthly_sales.idxmin()
        worst_month_sales = monthly_sales.min()

        kpis = {
            "total_revenue": total_revenue,
            "average_daily_sales": avg_daily_sales,
            "best_month": best_month.strftime('%B %Y'),
            "best_month_sales": best_month_sales,
            "worst_month": worst_month.strftime('%B %Y'),
            "worst_month_sales": worst_month_sales,
        }
        return jsonify(kpis)

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if conn is not None:
            conn.close()

@app.route('/api/product_demand_forecast', methods=['GET'])
def get_product_demand_forecast():
    """Forecasts demand for the top 5 selling products with a fallback for sparse data."""
    conn = None
    try:
        conn = psycopg2.connect(
            database=DB_NAME, user=DB_USER, password=DB_PASS, host=DB_HOST, port=DB_PORT
        )
        
        # Step 1: Find top 5 products (unchanged)
        top_products_query = """
            SELECT product_id, SUM(quantity) as total_quantity
            FROM orders
            GROUP BY product_id
            ORDER BY total_quantity DESC
            LIMIT 5;
        """
        top_products_df = pd.read_sql(top_products_query, conn)
        top_product_ids = top_products_df['product_id'].tolist()

        if not top_product_ids:
            return jsonify([])

        # Step 2: Fetch sales history (unchanged)
        sales_history_query = f"""
            SELECT o.last_purchase_date, o.product_id, o.quantity, p.product_name
            FROM orders o
            JOIN products p ON o.product_id = p.product_id
            WHERE o.product_id IN ({",".join(["'%s'" % pid for pid in top_product_ids])});
        """
        sales_history_df = pd.read_sql(sales_history_query, conn)
        sales_history_df['last_purchase_date'] = pd.to_datetime(sales_history_df['last_purchase_date'])

        # Step 3: Forecast for each product
        all_forecasts = []
        for product_id in top_product_ids:
            product_sales = sales_history_df[sales_history_df['product_id'] == product_id]
            daily_demand = product_sales.groupby('last_purchase_date')['quantity'].sum().asfreq('D').fillna(0)
            product_name = product_sales['product_name'].iloc[0]
            
            forecasted_demand = 0
            if len(daily_demand[daily_demand > 0]) > 7:
                model = ExponentialSmoothing(daily_demand, trend='add', seasonal=None).fit(smoothing_level=0.2)
                forecast = model.forecast(30)
                forecasted_demand = abs(round(forecast.sum()))
            else:
                total_units = daily_demand.sum()
                days_with_sales = (daily_demand.index.max() - daily_demand.index.min()).days
                if days_with_sales > 0:
                    avg_daily_rate = total_units / days_with_sales
                    forecasted_demand = abs(round(avg_daily_rate * 30))
                else:
                    forecasted_demand = total_units
            
            all_forecasts.append({
                "product_id": product_id,
                "product_name": product_name,
                # --- CRITICAL FIX: Convert the number to a standard Python integer ---
                "forecasted_demand_30_days": int(forecasted_demand)
            })

        return jsonify(all_forecasts)

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if conn is not None:
            conn.close()

@app.route('/api/user_distribution', methods=['GET'])
def get_user_distribution():
    """Calculates the number of users per country."""
    conn = None
    try:
        conn = psycopg2.connect(
            database=DB_NAME, user=DB_USER, password=DB_PASS, host=DB_HOST, port=DB_PORT
        )
        sql_query = """
            SELECT country, COUNT(customer_id) as user_count
            FROM customers
            GROUP BY country
            ORDER BY user_count DESC;
        """
        df = pd.read_sql(sql_query, conn)
        
        # Convert the DataFrame to a list of dictionaries
        country_data = df.to_dict(orient='records')
        return jsonify(country_data)

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if conn is not None:
            conn.close()

@app.route('/api/main_kpis', methods=['GET'])
def get_main_kpis():
    """Calculates the main dashboard KPIs: Revenue, Orders, AOV, and Churn Rate."""
    conn = None
    try:
        conn = psycopg2.connect(
            database=DB_NAME, user=DB_USER, password=DB_PASS, host=DB_HOST, port=DB_PORT
        )
        
        # --- Part 1: Calculate Sales KPIs ---
        sales_query = """
            SELECT
                SUM(unit_price * quantity) as total_revenue,
                COUNT(DISTINCT order_id) as total_orders
            FROM orders;
        """
        sales_df = pd.read_sql(sales_query, conn)
        total_revenue = sales_df['total_revenue'][0]
        total_orders = sales_df['total_orders'][0]
        average_order_value = total_revenue / total_orders if total_orders > 0 else 0

        # --- Part 2: Calculate Churn Rate ---
        customer_df = get_aggregated_data()
        customer_df_featured = feature_engineering_for_prediction(customer_df)
        df_predict = pd.get_dummies(customer_df_featured, columns=['gender', 'country'], drop_first=True)
        df_predict_aligned = df_predict.reindex(columns=model_columns, fill_value=0)
        df_predict_aligned[numeric_columns] = scaler.transform(df_predict_aligned[numeric_columns])
        predictions = churn_model.predict(df_predict_aligned[model_columns])
        churn_rate = (predictions.sum() / len(predictions)) * 100 if len(predictions) > 0 else 0

        # --- Part 3: Combine and CONVERT KPIs ---
        kpis = {
            "total_revenue": float(total_revenue),
            "total_orders": int(total_orders),
            "average_order_value": float(average_order_value),
            "churn_rate": float(churn_rate),
        }
        
        return jsonify(kpis)

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if conn is not None:
            conn.close()


@app.route('/api/upload_data', methods=['POST'])
def upload_data():
    """Receives an Excel file and uses the importer to add it to the database."""
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected for uploading"}), 400

    if file and (file.filename.endswith('.xls') or file.filename.endswith('.xlsx')):
        conn = None
        try:
            df = pd.read_excel(file)
            conn = psycopg2.connect(
                database=DB_NAME, user=DB_USER, password=DB_PASS, host=DB_HOST, port=DB_PORT
            )
            
            # Call the function from the other file
            result = insert_data_from_df(conn, df)
            
            if result['success']:
                return jsonify({"message": f"Successfully processed {result['rows_processed']} rows."})
            else:
                return jsonify({"error": result['error']}), 500

        except Exception as e:
            return jsonify({"error": f"An error occurred: {str(e)}"}), 500
        finally:
            if conn is not None:
                conn.close()
    else:
        return jsonify({"error": "Invalid file type. Please upload an Excel file."}), 400
    
@app.route('/api/sales_by_age', methods=['GET'])
def get_sales_by_age():
    """Calculates total sales revenue for predefined age groups."""
    conn = None
    try:
        conn = psycopg2.connect(
            database=DB_NAME, user=DB_USER, password=DB_PASS, host=DB_HOST, port=DB_PORT
        )

        # SQL query uses a CASE statement to group customers into age buckets
        sql_query = """
            SELECT
                CASE
                    WHEN c.age BETWEEN 18 AND 25 THEN '18-25'
                    WHEN c.age BETWEEN 26 AND 35 THEN '26-35'
                    WHEN c.age BETWEEN 36 AND 45 THEN '36-45'
                    WHEN c.age BETWEEN 46 AND 60 THEN '46-60'
                    -- Assuming anyone 61 or older is 60+
                    ELSE '60+' 
                END AS age_group,
                -- Sum the quantity from the orders table
                SUM(o.quantity) AS total_sales
            FROM customers c
            -- Join customers table with the orders table
            JOIN orders o ON c.customer_id = o.customer_id
            -- Group by the newly created age_group label
            GROUP BY age_group
            ORDER BY total_sales DESC;
        """
        
        # Use pandas to execute the query and fetch the results
        df = pd.read_sql(sql_query, conn)
        
        # Convert the DataFrame to a list of dictionaries for JSON
        # Example output structure: [{"age_group": "26-35", "total_sales": 150000.50}, ...]
        age_data = df.to_dict(orient='records')
        
        return jsonify(age_data)

    except Exception as e:
        # Handle the exception and return a 500 error
        print(f"Database Error in get_sales_by_age: {e}")
        return jsonify({"error": "Failed to fetch sales by age data."}), 500
    finally:
        if conn is not None:
            conn.close()

@app.route('/api/monthly_sales', methods=['GET'])
def get_monthly_sales():
    """Fetches total quantity sold grouped by month."""
    conn = None
    try:
        conn = psycopg2.connect(
            database=DB_NAME, user=DB_USER, password=DB_PASS, host=DB_HOST, port=DB_PORT
        )
        sql_query = """
            SELECT last_purchase_date, quantity
            FROM orders;
        """
        df = pd.read_sql(sql_query, conn)
        df['last_purchase_date'] = pd.to_datetime(df['last_purchase_date'], errors='coerce')
        df = df.dropna(subset=['last_purchase_date', 'quantity'])

        # Group by Year-Month and sum quantities
        monthly_sales = (
            df.groupby(df['last_purchase_date'].dt.to_period("M"))['quantity']
              .sum()
              .reset_index()
        )
        monthly_sales['last_purchase_date'] = monthly_sales['last_purchase_date'].dt.strftime('%B %Y')

        data = [
            {"month": row['last_purchase_date'], "total_quantity": int(row['quantity'])}
            for _, row in monthly_sales.iterrows()
        ]

        return jsonify(data)

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if conn is not None:
            conn.close()

@app.route('/api/yearly_sales', methods=['GET'])
def get_yearly_sales():
    """Fetches total quantity sold grouped by year."""
    conn = None
    try:
        conn = psycopg2.connect(
            database=DB_NAME, user=DB_USER, password=DB_PASS, host=DB_HOST, port=DB_PORT
        )
        sql_query = """
            SELECT last_purchase_date, quantity
            FROM orders;
        """
        df = pd.read_sql(sql_query, conn)
        df['last_purchase_date'] = pd.to_datetime(df['last_purchase_date'], errors='coerce')
        df = df.dropna(subset=['last_purchase_date', 'quantity'])

        yearly_sales = (
            df.groupby(df['last_purchase_date'].dt.year)['quantity']
              .sum()
              .reset_index()
        )
        yearly_sales.rename(columns={"last_purchase_date": "year", "quantity": "total_quantity"}, inplace=True)

        data = [
            {"year": int(row['year']), "total_quantity": int(row['total_quantity'])}
            for _, row in yearly_sales.iterrows()
        ]

        return jsonify(data)

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if conn is not None:
            conn.close()

@app.route('/api/db_stats', methods=['GET'])
def get_db_stats():
    """Returns total entries count and % of cancelled subscriptions."""
    conn = None
    try:
        conn = psycopg2.connect(
            database=DB_NAME, user=DB_USER, password=DB_PASS,
            host=DB_HOST, port=DB_PORT
        )
        cur = conn.cursor()

        # Total entries
        cur.execute("SELECT COUNT(*) FROM orders;")
        total_count = cur.fetchone()[0]

        # Cancelled subscriptions
        cur.execute("""
            SELECT COUNT(*) FROM orders 
            WHERE subscription_status = 'cancelled';
        """)
        cancelled_count = cur.fetchone()[0]

        # Percentage calculation
        cancelled_percentage = (
            (cancelled_count / total_count) * 100 if total_count > 0 else 0
        )

        cur.close()
        return jsonify({
            "total_entries": total_count,
            "cancelled_count": cancelled_count,
            "cancelled_percentage": round(cancelled_percentage, 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if conn:
            conn.close()

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, threaded=False)