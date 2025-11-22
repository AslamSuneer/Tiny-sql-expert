import sqlite3
from tiny_sql_expert import create_inmemory_db, validate_sql_with_sqlite


def test_validate_valid_select():
    with open("schema.sql", "r", encoding="utf-8") as f:
        schema = f.read()
    conn = create_inmemory_db(schema)
    vr = validate_sql_with_sqlite("SELECT 1;", conn)
    assert vr.is_valid


def test_forbidden_word():
    with open("schema.sql", "r", encoding="utf-8") as f:
        schema = f.read()
    conn = create_inmemory_db(schema)
    vr = validate_sql_with_sqlite("DROP TABLE users;", conn)
    assert not vr.is_valid
    assert "Contains forbidden" in vr.error
