import psycopg2
import sys

# --- Fill in these details ---
db_host = "29.91.35.42"  # Your laptop's IP address
db_port = "5432"          # The PostgreSQL port, usually 5432
db_name = "hackathon"    # The name of the database
db_user = "postgres"        # Your database username
db_pass = "Post@7070"    # Your database password
# -----------------------------

try:
    # Establish the connection
    conn = psycopg2.connect(
        host=db_host,
        port=db_port,
        dbname=db_name,
        user=db_user,
        password=db_pass
    )

    # Create a cursor object to interact with the database
    cur = conn.cursor()

    # --- Test the connection by getting the PostgreSQL version ---
    print('PostgreSQL database version:')
    cur.execute('SELECT version()')
    db_version = cur.fetchone()
    print(db_version)
    
    # Close the cursor and connection
    cur.close()
    conn.close()
    print("Database connection closed.")

except psycopg2.OperationalError as e:
    print(f"Could not connect to the database: {e}", file=sys.stderr)
    # This error often means a firewall is blocking the connection,
    # the IP address is wrong, or the database server isn't running.

except psycopg2.Error as e:
    print(f"Database error: {e}", file=sys.stderr)
    # This can happen for various other reasons, like bad credentials.