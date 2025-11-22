# Tiny SQL Expert — NL → SQL using a Local Small LLM (<4B)

**Author:** Aslam Suneer  
**Task chosen:** The "Tiny" SQL Expert (SLM Optimization) — convert English questions into SQL using a small local LLM, with schema awareness and a self-correction/validation loop.

---

## Summary
This project converts natural-language SQL questions into valid SQL statements using a local small model (TinyLlama 1.1B in my demos). The pipeline is schema-aware (DDL injected), uses few-shot prompting, and includes an automated **validation + repair** loop: generated SQL is validated in SQLite (EXPLAIN). If the SQL fails, the script re-prompts the model with the failing SQL and exact sqlite error, asking for a corrected SQL. All attempts are logged to `logs/pipeline.log` as JSONL — showing failure → retry → success.

**Key features**
- Use of SLM < 4B (TinyLlama 1.1B recommended / used in demos)
- Schema injection (users, products, orders) — forces JOINs
- Forbidden-word filter (DROP, DELETE, ALTER, TRUNCATE)
- Validation via SQLite EXPLAIN / EXPLAIN QUERY PLAN
- Repair prompt + retry logic (self-correction loop)
- Unit tests (`pytest`) and metrics runner
- Logging (JSONL) for audit and evidence

---

## Repo contents
- `tiny_sql_expert.py` — main CLI script
- `schema.sql` — sample schema (users, products, orders)
- `seed_and_run.py` — seeds demo data and runs pipeline (mock)
- `metrics.py` — runs many prompts and records success rates
- `tests/` — unit & pipeline mock tests
- `requirements.txt` — runtime deps
- `requirements-dev.txt` — dev deps (black/flake8/pre-commit/pytest)
- `.pre-commit-config.yaml` — formatting & lint hooks
- `demo/` — screenshots or video of the pipeline
- `logs/` — JSONL logs (created during runs)

---

## Quick start (Windows, PowerShell) — minimal (mock)
1. Create & activate venv:
```powershell
cd C:\Users\aslam\Project
python -m venv venv
venv\Scripts\activate
