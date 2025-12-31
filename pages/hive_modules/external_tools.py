# hive_modules/external_tools.py
import httpx
import sqlite3

@inject_convo_uuid  # Use your existing decorator
def webhook_invoker(url: str, payload: dict, method: str = "POST", convo_uuid: str = None) -> str:
    """
    🜛 Tool: Webhook Invoker
    Agents can trigger external services
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.request(method, url, json=payload)
            return f"Webhook {method} to {url}: {response.status_code}"
    except Exception as e:
        return f"Webhook error: {str(e)}"

@inject_convo_uuid
def database_persist(table: str, data: dict, convo_uuid: str = None) -> str:
    """
    🜛 Tool: Database Persistence
    Write to external DB (separate from memory)
    """
    try:
        # Use a separate DB for external data
        conn = sqlite3.connect("./sandbox/external_data.db")
        cursor = conn.cursor()
        
        # Create table if not exists
        columns = ", ".join([f"{k} TEXT" for k in data.keys()])
        cursor.execute(f"CREATE TABLE IF NOT EXISTS {table} (id INTEGER PRIMARY KEY, {columns})")
        
        # Insert
        placeholders = ", ".join(["?" for _ in data])
        cursor.execute(f"INSERT INTO {table} VALUES (NULL, {placeholders})", list(data.values()))
        conn.commit()
        conn.close()
        
        return f"Persisted to {table}: {data}"
    except Exception as e:
        return f"DB error: {str(e)}"

@inject_convo_uuid
def database_query(query: str, convo_uuid: str = None) -> str:
    """
    🜛 Tool: Database Query
    Read from external DB
    """
    try:
        conn = sqlite3.connect("./sandbox/external_data.db")
        cursor = conn.cursor()
        cursor.execute(query)
        results = cursor.fetchall()
        conn.close()
        return f"Query results: {results}"
    except Exception as e:
        return f"Query error: {str(e)}"

# Auto-register with ToolBridge
def register_external_tools(container):
    container.register_tool(webhook_invoker)
    container.register_tool(database_persist)
    container.register_tool(database_query)

# In your ToolBridge._load_tools(), add:
# from hive_modules.external_tools import register_external_tools
# register_external_tools(container)
