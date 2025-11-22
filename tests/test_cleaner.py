from tiny_sql_expert import clean_model_output


def test_clean_model_output_strips_fences():
    raw = "```sql\nSQL: SELECT 1\n```"
    cleaned = clean_model_output(raw)
    assert cleaned.strip().upper().startswith("SELECT 1")
    assert cleaned.endswith(";")
