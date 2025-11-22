from tiny_sql_expert import create_inmemory_db, ask_with_retries


def test_pipeline_mock_success():
    with open("schema.sql", "r", encoding="utf-8") as f:
        schema = f.read()
    conn = create_inmemory_db(schema)
    shots = [
        {
            "question": "List users who bought electronics",
            "sql": "SELECT u.name, u.email FROM users u JOIN orders o ON u.user_id=o.user_id JOIN products p ON p.product_id=o.product_id WHERE p.category='Electronics';",
        }
    ]
    final_sql = ask_with_retries(
        "List users who bought electronics", schema, conn, shots, mock=True
    )
    assert "SELECT" in final_sql.upper()
