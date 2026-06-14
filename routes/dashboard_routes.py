from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Any, Dict
import sqlite3
import math
import pandas as pd
import os
import uuid
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import itertools
import json
import hashlib

router = APIRouter()
DB_PATH = 'data/inventory.db'

def get_db_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

@router.get("/products", response_model=List[Dict[str, Any]])
def get_products() -> List[Dict[str, Any]]:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM products")
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

@router.get("/search/{query}", response_model=List[Dict[str, Any]])
def search_products(query: str) -> List[Dict[str, Any]]:
    conn = get_db_connection()
    cursor = conn.cursor()
    search_term = f"%{query}%"
    cursor.execute("""
        SELECT * FROM products 
        WHERE product LIKE ? OR categories LIKE ? OR sub_category LIKE ?
    """, (search_term, search_term, search_term))
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


class CheckoutItem(BaseModel):
    ProductID: str
    Quantity_Bought: int

class CheckoutRequest(BaseModel):
    items: List[CheckoutItem]
    payment_method: str = "Cash"

@router.post("/checkout")
def checkout(request: CheckoutRequest) -> Dict[str, Any]:
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # 1. Generate Batch IDs
    tx_id = f"TXN-{uuid.uuid4().hex[:8].upper()}"
    # Formatted exactly as YYYYMMDDHHMMSS
    tx_time = datetime.now().strftime('%Y-%m-%d %H:%M')
    
    try:
        for item in request.items:

            cursor.execute("""
                SELECT Stock, product, categories, sub_category, brand, sale_price, 
                COALESCE(cost_price, sale_price * 0.86) as cost_price 
                FROM products WHERE ProductID = ?
            """, (item.ProductID,))
            row = cursor.fetchone()
            
            if not row:
                raise HTTPException(status_code=404, detail=f"Product {item.ProductID} not found")
                
            s_price = float(row['sale_price'])
            c_price = float(row['cost_price'])
            current_stock = int(row['Stock'])
            
            # 3. Calculate Math
            total_price = s_price * item.Quantity_Bought
            profit = (s_price - c_price) * item.Quantity_Bought
            new_stock = current_stock - item.Quantity_Bought
            
            if new_stock < 0:
                raise HTTPException(status_code=400, detail=f"Insufficient stock for {row['product']}")
            
            # 4. Update Main Inventory
            cursor.execute("UPDATE products SET Stock = ?, Units_Sold = COALESCE(Units_Sold, 0) + ? WHERE ProductID = ?", 
                           (new_stock, item.Quantity_Bought, item.ProductID))
                           
            # 5. Log the Sale
            cursor.execute("""
                INSERT INTO sales_logs 
                (Transaction_ID, Timestamp, ProductID, Product_Name, Sale_Price, Cost_Price, 
                Qty_Sold, Total_Price, Profit, Category, Sub_Category, Brand)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                tx_id, tx_time, item.ProductID, row['product'], s_price, 
                c_price, item.Quantity_Bought, total_price, profit, 
                row['categories'], row['sub_category'], row['brand']
            ))
            
        conn.commit()

    except Exception as e:
        conn.rollback() # Rolls back the entire batch if one item fails
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        conn.close()
        
    return {"message": "Checkout and logging successful", "transaction_id": tx_id}


# =====================================================================
# --- SECURITY, CONFIGURATION, AND ACCESS CONTROL ENDPOINTS ---
# =====================================================================

CONFIG_FILE = "data/store_config.json"

class StoreSettings(BaseModel):
    monthly_rent: float
    monthly_electricity: float
    monthly_salaries: float
    tax_rate_percent: float

# --- 1. CRYPTOGRAPHY HELPER ---
def hash_password(password: str) -> str:
    # Enterprise standard: Never store plain text passwords
    return hashlib.sha256(password.encode()).hexdigest()

# --- 2. UPGRADED CONFIG LOADER ---
def get_store_config() -> Dict[str, Any]:
    default_config = {
        "monthly_rent": 0.0, 
        "monthly_electricity": 0.0, 
        "monthly_salaries": 0.0, 
        "tax_rate_percent": 0.0,
        "admin_password_hash": hash_password("admin123") # Default presentation password
    }
    
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                loaded = json.load(f)
                # Ensure hash exists for older config files
                if "admin_password_hash" not in loaded:
                    loaded["admin_password_hash"] = default_config["admin_password_hash"]
                return loaded
        except Exception:
            pass
    return default_config


@router.get("/settings")
def read_settings():
    return get_store_config()

@router.post("/settings")
def update_settings(settings: StoreSettings):
    os.makedirs("data", exist_ok=True)
    
    # Secure Merge: Preserve the password when updating financial settings
    current_config = get_store_config()
    new_config = settings.model_dump() if hasattr(settings, 'model_dump') else settings.dict()
    new_config["admin_password_hash"] = current_config.get("admin_password_hash")
    
    with open(CONFIG_FILE, "w") as f:
        json.dump(new_config, f)
    return {"message": "Store configuration securely updated!"}


# --- 3. NEW ACCESS CONTROL ENDPOINTS ---
class LoginRequest(BaseModel):
    password: str

class PasswordChangeRequest(BaseModel):
    current_password: str
    new_password: str

@router.post("/auth/login")
def login(req: LoginRequest):
    config = get_store_config()
    if hash_password(req.password) == config.get("admin_password_hash"):
        return {"status": "success", "message": "Authenticated"}
    raise HTTPException(status_code=401, detail="Invalid administrator password")

@router.post("/auth/change_password")
def change_password(req: PasswordChangeRequest):
    config = get_store_config()
    
    # Verify current password
    if hash_password(req.current_password) != config.get("admin_password_hash"):
        raise HTTPException(status_code=401, detail="Current password incorrect")
        
    # Update to new password
    config["admin_password_hash"] = hash_password(req.new_password)
    
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f)
        
    return {"message": "Admin password successfully updated!"}

# =====================================================================


@router.get("/analytics")
def get_analytics() -> Dict[str, Any]:
    try:
        conn = get_db_connection()
        sales_df = pd.read_sql_query("SELECT * FROM sales_logs", conn)
        products_df = pd.read_sql_query("SELECT ProductID, product, Stock FROM products", conn)
        conn.close()
        
        if sales_df.empty:
            return {"status": "waiting", "message": "Waiting for checkout logs"}
            
        sales_df['Timestamp'] = pd.to_datetime(sales_df['Timestamp'], format='mixed', errors='coerce')
        sales_df = sales_df.dropna(subset=['Timestamp'])
        sales_df['Date'] = sales_df['Timestamp'].dt.date.astype(str)
        sales_df['Hour'] = sales_df['Timestamp'].dt.hour
        
        config = get_store_config()
        
        months_active = 1.0
        min_date = sales_df['Timestamp'].min()
        max_date = sales_df['Timestamp'].max()
        
        if pd.notnull(min_date) and pd.notnull(max_date):
            days_active = (max_date - min_date).days
            # Even if it's 5 days, prorate the exact rent down to the decimal
            months_active = max(days_active, 1) / 30.44

        monthly_fixed_costs = (config.get('monthly_rent', 0.0) + 
                               config.get('monthly_electricity', 0.0) + 
                               config.get('monthly_salaries', 0.0))
        
        total_fixed_overhead = monthly_fixed_costs * months_active
        
        total_revenue = float(sales_df['Total_Price'].sum())
        gross_profit = float(sales_df['Profit'].sum())

        
        
        tax_amount = gross_profit * (config.get('tax_rate_percent', 0.0) / 100.0)
        total_overhead = total_fixed_overhead + tax_amount
        net_profit = gross_profit - total_overhead
        
        gross_margin = float((gross_profit / total_revenue) * 100) if total_revenue > 0 else 0.0
        net_margin = float((net_profit / total_revenue) * 100) if total_revenue > 0 else 0.0
        total_units = int(sales_df['Qty_Sold'].sum())
        
        # --- THE FIX: json.loads(df.to_json()) strips all NumPy datatypes so FastAPI doesn't crash ---
        daily_sales = json.loads(sales_df.groupby('Date')['Total_Price'].sum().reset_index().to_json(orient='records'))
        hourly_sales = json.loads(sales_df.groupby('Hour')['Transaction_ID'].nunique().reset_index().to_json(orient='records'))
        
        top_rev = json.loads(sales_df.groupby('Product_Name')['Total_Price'].sum().sort_values(ascending=False).head(5).reset_index().to_json(orient='records'))
        top_vol = json.loads(sales_df.groupby('Product_Name')['Qty_Sold'].sum().sort_values(ascending=False).head(5).reset_index().to_json(orient='records'))
        top_cat = json.loads(sales_df.groupby('Category')['Profit'].sum().sort_values(ascending=False).head(5).reset_index().to_json(orient='records'))
        
        basket_tx = sales_df.groupby('Transaction_ID').filter(lambda x: len(x) > 1)
        formatted_pairs = []
        if not basket_tx.empty:
            baskets = basket_tx.groupby('Transaction_ID')['Product_Name'].apply(list)
            pairs = []
            for basket in baskets:
                pairs.extend(list(itertools.combinations(sorted(basket), 2)))
            if pairs:
                pair_counts = pd.Series(pairs).value_counts().head(5)
                formatted_pairs = [{"Item_A": p[0], "Item_B": p[1], "Times_Bought_Together": int(count)} for p, count in pair_counts.items()]
                
        oldest_sale = sales_df['Timestamp'].min()
        days_running = (pd.Timestamp.now() - oldest_sale).days if pd.notnull(oldest_sale) else 0
        dead_stock = []
        if days_running >= 7:
            sold_pids = sales_df['ProductID'].unique()
            dead_stock_df = products_df[~products_df['ProductID'].isin(sold_pids)]
            dead_stock = json.loads(dead_stock_df[['product', 'Stock']].head(5).to_json(orient='records'))
            
        return {
            "status": "success",
            "kpis": {
                "total_revenue": total_revenue,
                "gross_profit": gross_profit,
                "net_profit": net_profit,
                "gross_margin": gross_margin,
                "net_margin": net_margin,
                "overhead_deducted": total_overhead,
                "total_units": total_units
            },
            "trends": { "daily": daily_sales, "hourly": hourly_sales },
            "performance": { "top_revenue": top_rev, "top_volume": top_vol, "top_categories": top_cat },
            "insights": { "market_basket": formatted_pairs, "dead_stock": dead_stock, "days_running": days_running }
        }
    except Exception as e:
        import traceback
        traceback.print_exc() # Prints exact error to your terminal if it fails again
        raise HTTPException(status_code=500, detail=str(e))

        
@router.get("/export")
def export_logs(start_date: str, end_date: str) -> List[Dict[str, Any]]:
    try:
        conn = get_db_connection()
        query = f"""
            SELECT * FROM sales_logs 
            WHERE Timestamp >= '{start_date} 00:00' AND Timestamp <= '{end_date} 23:59'
            ORDER BY Timestamp DESC
        """
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df.to_dict('records')
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class ImportSalesRequest(BaseModel):
    data: List[Dict[str, Any]]
    mode: str # "append" or "replace"

@router.post("/import_sales")
def import_sales(req: ImportSalesRequest) -> Dict[str, Any]:
    if not req.data:
        raise HTTPException(status_code=400, detail="Empty data payload")
    try:
        conn = get_db_connection()
        df = pd.DataFrame(req.data)
        
        # FIX 3: Stop the ghost multiplier by respecting the "replace" mode
        if req.mode == 'replace':
            df.to_sql('sales_logs', conn, if_exists='replace', index=False)
            final_count = len(df)
        else:
            df.to_sql('sales_logs', conn, if_exists='append', index=False)
            # Fetch to get accurate count
            existing_df = pd.read_sql_query("SELECT * FROM sales_logs", conn)
            final_count = len(existing_df)
            
        conn.commit()
        conn.close()
        return {"message": f"Sales logs {req.mode}d successfully! Total database records: {final_count}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class ImportInventoryRequest(BaseModel):
    data: List[Dict[str, Any]]
    mode: str # "update" or "replace"

class ImportInventoryRequest(BaseModel):
    data: List[Dict[str, Any]]
    mode: str # "update" or "replace"

@router.post("/import_inventory")
def import_inventory(req: ImportInventoryRequest) -> Dict[str, Any]:
    if not req.data:
        raise HTTPException(status_code=400, detail="Empty data payload")
        
    try:
        conn = get_db_connection()
        df = pd.DataFrame(req.data)
        
        # 1. Normalize IDs
        if 'ProductID' in df.columns:
            df['ProductID'] = df['ProductID'].astype(str)
        
        # 2. SQLite Database Merge (Completely bypasses CSV files!)
        if req.mode == "update":
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='products'")
            
            if cursor.fetchone():
                existing_df = pd.read_sql_query("SELECT * FROM products", conn)
                
                if not existing_df.empty and 'ProductID' in existing_df.columns:
                    existing_df['ProductID'] = existing_df['ProductID'].astype(str)
                    
                    # Align indices to perfectly merge chunks
                    existing_df.set_index('ProductID', inplace=True)
                    df.set_index('ProductID', inplace=True)
                    
                    existing_df.update(df)
                    new_rows = df[~df.index.isin(existing_df.index)]
                    final_df = pd.concat([existing_df, new_rows]).reset_index()
                else:
                    final_df = df.reset_index() if 'ProductID' in df.index.names else df
            else:
                final_df = df.reset_index() if 'ProductID' in df.index.names else df
        else:
            final_df = df
            
        # 3. Protect historical sales data during updates
        if 'Units_Sold' not in final_df.columns:
            final_df['Units_Sold'] = 0
        else:
            final_df['Units_Sold'] = final_df['Units_Sold'].fillna(0)
            
        # 4. Save directly back to SQLite Database
        final_df.to_sql('products', conn, if_exists='replace', index=False)
        conn.commit()
        conn.close()
        
        return {"message": f"Inventory chunk processed! Total DB rows: {len(final_df)}"}
        
    except Exception as e:
        # If it ever crashes again, this will print the EXACT reason to your Render logs
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/recommendations/{product_id}")
def get_ml_recommendations(product_id: str, top_n: int = 3) -> List[Dict[str, Any]]:
    conn = get_db_connection()
    cursor = conn.cursor()
    # Fetch core data to build the AI corpus (FIXED COLUMN NAMES)
    cursor.execute("SELECT ProductID, product, categories, sub_category, brand, sale_price, Stock FROM products")
    rows = cursor.fetchall()
    conn.close()
    
    df = pd.DataFrame([dict(row) for row in rows])
    if df.empty or len(df) < 2:
        return []
        
    # BUILD CORPUS: Combine text fields to figure out what the product actually is
    df['content'] = df['product'].fillna('') + ' ' + df['categories'].fillna('') + ' ' + df['sub_category'].fillna('') + ' ' + df['brand'].fillna('')
    
    # Safety gate: If the product isn't found, return nothing
    if product_id not in df['ProductID'].values:
        return []
        
    try:
        # TF-IDF VECTORIZE: Turn text into mathematical weights
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(df['content'])
        
        # COSINE LOOKUP: Find the mathematical angle between the target product and all others
        target_idx = df.index[df['ProductID'] == product_id].tolist()[0]
        cosine_sim = cosine_similarity(tfidf_matrix[target_idx], tfidf_matrix).flatten()
        
        # Get the highest scoring items (excluding the item itself)
        similar_indices = cosine_sim.argsort()[-(top_n+1):-1][::-1]
        
        recommendations = df.iloc[similar_indices].to_dict(orient='records')
        return recommendations
    except Exception as e:
        print(f"ML Engine Error: {e}")
        return []


class ProductUpdate(BaseModel):
    product: str
    categories: str
    sub_category: str
    description: str
    sale_price: float
    Cost_Price: float
    Stock: int
    Capacity: int

@router.put("/products/{product_id}")
def update_product(product_id: str, update_data: ProductUpdate) -> Dict[str, str]:
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            UPDATE products 
            SET product = ?, categories = ?, sub_category = ?, description = ?, 
                sale_price = ?, Cost_Price = ?, Stock = ?, Capacity = ?
            WHERE ProductID = ?
        """, (
            update_data.product, update_data.categories, update_data.sub_category, 
            update_data.description, update_data.sale_price, update_data.Cost_Price, 
            update_data.Stock, update_data.Capacity, product_id
        ))
        
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="Product not found")
            
        conn.commit()
        return {"message": "Product updated successfully"}
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()


class RestockItem(BaseModel):
    ProductID: str
    Qty_Received: int

class RestockRequest(BaseModel):
    items: List[RestockItem]

@router.post("/restock")
def process_restock(request: RestockRequest) -> Dict[str, Any]:
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # --- FAILSAFE: Auto-create the audit table if it doesn't exist ---
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS restock_logs (
            Batch_ID TEXT,
            Timestamp TEXT,
            ProductID TEXT,
            Product_Name TEXT,
            Qty_Added INTEGER
        )
    ''')
    
    batch_id = f"RCV-{uuid.uuid4().hex[:8].upper()}"
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
    
    try:
        for item in request.items:
            # 1. Update the Main Inventory
            cursor.execute("UPDATE products SET Stock = Stock + ? WHERE ProductID = ?", (item.Qty_Received, item.ProductID))
            
            # 2. Get the product name for the log
            cursor.execute("SELECT product FROM products WHERE ProductID = ?", (item.ProductID,))
            row = cursor.fetchone()
            p_name = row['product'] if row else "Unknown Item"
            
            # 3. Write to Audit Trail
            cursor.execute("""
                INSERT INTO restock_logs (Batch_ID, Timestamp, ProductID, Product_Name, Qty_Added)
                VALUES (?, ?, ?, ?, ?)
            """, (batch_id, timestamp, item.ProductID, p_name, item.Qty_Received))
            
        conn.commit()
        return {"message": "Receipt Processed successfully", "batch_id": batch_id}
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()

@router.get("/restock_logs")
def get_restock_logs() -> List[Dict[str, Any]]:
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        # Check if table exists before querying
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='restock_logs'")
        if not cursor.fetchone():
            return []
            
        cursor.execute("SELECT * FROM restock_logs ORDER BY Timestamp DESC LIMIT 150")
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()