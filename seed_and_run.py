from tiny_sql_expert import create_inmemory_db, ask_with_retries
import sqlite3

# Load schema
with open("schema.sql", "r", encoding="utf-8") as f:
    schema_sql = f.read()

conn = create_inmemory_db(schema_sql)

# Seed sample data
cur = conn.cursor()
cur.executescript(
    """
INSERT INTO users (user_id, name, email, signup_date) VALUES
 (1, 'Alice', 'alice@example.com', '2024-01-10'),
 (2, 'Bob', 'bob@example.com', '2024-02-12'),
 (3, 'Carol', 'carol@example.com', '2024-03-05');

INSERT INTO products (product_id, name, category, price) VALUES
 (1, 'Widget A', 'Electronics', 199.99),
 (2, 'Gadget B', 'Home', 29.99),
 (3, 'Gizmo C', 'Electronics', 49.99);

INSERT INTO orders (order_id, user_id, product_id, quantity, order_date, status) VALUES
 (1, 1, 1, 1, '2025-03-02', 'completed'),
 (2, 2, 2, 2, '2025-03-05', 'completed'),
 (3, 3, 3, 1, '2025-03-10', 'pending'),
 (4, 1, 3, 2, '2025-03-15', 'completed');
"""
)
conn.commit()

# Shots for pipeline (keep consistent with tiny_sql_expert)
shots = [
    {
        "question": "List all users (name and email) who ordered products in category 'Electronics'.",
        "sql": "SELECT u.name, u.email FROM users u JOIN orders o ON u.user_id = o.user_id JOIN products p ON p.product_id = o.product_id WHERE p.category = 'Electronics';",
    },
    {
        "question": "Count number of orders per product.",
        "sql": "SELECT p.name, COUNT(*) AS orders_count FROM products p JOIN orders o ON p.product_id = o.product_id GROUP BY p.product_id;",
    },
]

if __name__ == "__main__":
    nl_q = "List all users who bought electronics"
    # Use mock=True to avoid needing a model for quick demo
    final_sql = ask_with_retries(nl_q, schema_sql, conn, shots, mock=True)
    print("=== GENERATED SQL ===")
    print(final_sql)
    print()
    cur = conn.cursor()
    cur.execute(final_sql)
    rows = cur.fetchall()
    print("=== QUERY RESULTS ===")
    for r in rows:
        print(r)
