import os
import sqlite3
import pandas as pd
from fastapi import FastAPI
from contextlib import asynccontextmanager

DB_PATH = 'data/inventory.db'
CSV_PATH = 'data/master_inventory.csv'

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize DB
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)

    conn = sqlite3.connect(DB_PATH, timeout=20.0)
    conn.execute('PRAGMA journal_mode=WAL')

    # --- NEW: Initialize Local Operational Configuration JSON Baseline ---
    config_file = "data/store_config.json"
    if not os.path.exists(config_file):
        with open(config_file, "w") as f:
            import json
            json.dump({
                "monthly_rent": 0.0,
                "monthly_electricity": 0.0,
                "monthly_salaries": 0.0,
                "tax_rate_percent": 0.0
            }, f)
    
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='products'")
    table_exists = cursor.fetchone()
    
    # --- THE FIX 1: Only read the CSV if it actually exists ---
    if not table_exists:
        if os.path.exists(CSV_PATH):
            df = pd.read_csv(CSV_PATH)
            df.to_sql('products', conn, if_exists='replace', index=False)

    # --- THE FIX 2: Only check for Units_Sold if the table actually got created ---
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='products'")
    if cursor.fetchone():
        cursor.execute("PRAGMA table_info(products)")
        existing_columns = [info[1] for info in cursor.fetchall()]
        
        if 'Units_Sold' not in existing_columns:
            cursor.execute("ALTER TABLE products ADD COLUMN Units_Sold INTEGER DEFAULT 0")

    # --- Create sales_logs table (Safe to run empty) ---
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS sales_logs (
            Transaction_ID TEXT,
            Timestamp TEXT,
            ProductID TEXT,
            Product_Name TEXT,
            Sale_Price REAL,
            Cost_Price REAL,
            Qty_Sold INTEGER,
            Total_Price REAL,
            Profit REAL,
            Category TEXT,
            Sub_Category TEXT,
            Brand TEXT
        )
    ''')
        
    conn.commit()
    conn.close()
    yield
    # Shutdown

app = FastAPI(lifespan=lifespan)

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from routes.dashboard_routes import router as dashboard_router

app.include_router(dashboard_router, prefix="/api")
