from tiny_sql_expert import create_inmemory_db, ask_with_retries
import sqlite3

with open("schema.sql", "r", encoding="utf-8") as f:
    schema = f.read()
conn = create_inmemory_db(schema)
shots_short = [
    {
        "question": "List all users (name and email) who ordered products in category 'Electronics'.",
        "sql": "SELECT u.name, u.email FROM users u JOIN orders o ON u.user_id = o.user_id JOIN products p ON p.product_id = o.product_id WHERE p.category = 'Electronics';",
    }
]
sql = ask_with_retries(
    "List distinct users who bought electronics",
    schema,
    conn,
    shots_short,
    model_path="./models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
    mock=False,
    n_ctx=512,
    n_threads=2,
)
print(sql)
