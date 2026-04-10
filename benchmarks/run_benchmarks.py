import os
import time
import tracemalloc
import duckdb
import pandas as pd
import pyarrow.parquet as pq

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from query_engine.execution import execute_query, QueryContext

# 1. Data Generation
def generate_data_if_needed():
    if os.path.exists('users.parquet') and os.path.exists('transactions.parquet'):
        print("Data already exists. Skipping generation.")
        return
        
    print("Generating 10M row dataset with DuckDB...")
    con = duckdb.connect()
    con.execute("""
        CREATE TABLE users AS 
        SELECT 
            range as id, 
            'Region_' || (range % 5) as region 
        FROM range(1000000)
    """)
    con.execute("COPY users TO 'users.parquet' (FORMAT PARQUET)")
    
    con.execute("""
        CREATE TABLE transactions AS 
        SELECT 
            range as id, 
            (range % 1000000) as user_id, 
            (random() * 100)::FLOAT as amount 
        FROM range(10000000)
    """)
    con.execute("COPY transactions TO 'transactions.parquet' (FORMAT PARQUET)")
    print("Data generation complete.")

def run_duckdb():
    print("Running DuckDB benchmark...")
    con = duckdb.connect()
    
    start_time = time.time()
    tracemalloc.start()
    
    res = con.execute("""
        SELECT 
            users.region, 
            SUM(transactions.amount) as sum_amount
        FROM 'users.parquet' as users
        JOIN 'transactions.parquet' as transactions
        ON users.id = transactions.user_id
        WHERE transactions.amount > 10
        GROUP BY users.region
        ORDER BY users.region
    """).fetchall()
    
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    duration = time.time() - start_time
    
    print(f"DuckDB Results: {res}")
    print(f"DuckDB Latency: {duration:.4f} seconds")
    print(f"DuckDB Peak Memory: {peak / 1024 / 1024:.2f} MB")
    print("-" * 40)

def run_pandas():
    print("Running Pandas benchmark...")
    start_time = time.time()
    tracemalloc.start()
    
    users_df = pd.read_parquet('users.parquet')
    tx_df = pd.read_parquet('transactions.parquet')
    
    # Filter
    tx_df = tx_df[tx_df['amount'] > 10]
    
    # Join
    joined = pd.merge(users_df, tx_df, left_on='id', right_on='user_id')
    
    # Group By + Agg
    grouped = joined.groupby('region')['amount'].sum().reset_index()
    
    # Order By
    result = grouped.sort_values('region')
    
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    duration = time.time() - start_time
    
    print(f"Pandas Results:\n{result.head()}")
    print(f"Pandas Latency: {duration:.4f} seconds")
    print(f"Pandas Peak Memory: {peak / 1024 / 1024:.2f} MB")
    print("-" * 40)

def run_custom_engine():
    print("Running Custom Engine benchmark...")
    ctx = QueryContext(data_sources={
        'users': 'users.parquet',
        'transactions': 'transactions.parquet'
    })
    
    sql = """
    SELECT region, SUM(amount)
    FROM users 
    JOIN transactions ON id = user_id 
    WHERE amount > 10 
    GROUP BY region 
    ORDER BY region
    """
    
    start_time = time.time()
    tracemalloc.start()
    
    chunks = execute_query(sql, ctx)
    
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    duration = time.time() - start_time
    
    if chunks:
        # Just print the first chunk summarizing
        print(f"Custom Engine Results (Cols: {list(chunks[0].keys())}):")
        for i in range(min(5, len(next(iter(chunks[0].values()))))):
            row = {k: v[i] for k, v in chunks[0].items()}
            print(row)
            
    print(f"Custom Engine (Cold) Latency: {duration:.4f} seconds")
    print(f"Custom Engine (Cold) Peak Memory: {peak / 1024 / 1024:.2f} MB")
    
    print("Running Custom Engine Cache benchmark...")
    start_time_cache = time.time()
    tracemalloc.start()
    
    chunks_cached = execute_query(sql, ctx)
    
    current_c, peak_c = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    duration_cache = time.time() - start_time_cache
    
    print(f"Custom Engine (Cached) Latency: {duration_cache:.6f} seconds")
    print(f"Custom Engine (Cached) Peak Memory: {peak_c / 1024 / 1024:.2f} MB")
    print("-" * 40)

if __name__ == "__main__":
    generate_data_if_needed()
    run_duckdb()
    run_pandas()
    run_custom_engine()
