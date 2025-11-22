from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
import time
from dataclasses import dataclass
from typing import List, Optional

import sqlparse

# Optional import (only used in real-model mode). If not installed, user can use --mock.
try:
    from llama_cpp import Llama  # type: ignore
except Exception:  # pragma: no cover - import failure handled at runtime
    Llama = None  # type: ignore

# Constants and defaults
FORBIDDEN_WORDS = {"drop", "delete", "truncate", "alter"}
RETRY_LIMIT = 3
DEFAULT_MODEL_PATH = "DEFAULT_MODEL_PATH = ./models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
LOG_PATH = "logs/pipeline.log"
PROMPT_SNAPSHOT_MAX = 1200  # chars to store in logs


@dataclass
class ValidationResult:
    is_valid: bool
    error: Optional[str] = None


# --------------------------
# Database / Validation
# --------------------------
def create_inmemory_db(schema_sql: str) -> sqlite3.Connection:
    """Create an in-memory sqlite DB and apply DDL schema_sql."""
    conn = sqlite3.connect(":memory:")
    conn.executescript(schema_sql)
    return conn


def has_forbidden_words(sql: str) -> bool:
    """Return True if SQL contains forbidden words (case-insensitive)."""
    lowered = sql.lower()
    for word in FORBIDDEN_WORDS:
        if re.search(rf"\b{re.escape(word)}\b", lowered):
            return True
    return False


def validate_sql_with_sqlite(sql: str, conn: sqlite3.Connection) -> ValidationResult:
    """Validate SQL using sqlite3 using EXPLAIN / EXPLAIN QUERY PLAN.

    This avoids executing data-modifying statements.
    """
    if has_forbidden_words(sql):
        return ValidationResult(False, "Contains forbidden operation")

    statements = [s.strip() for s in sqlparse.split(sql) if s.strip()]

    try:
        for stmt in statements:
            normalized = stmt.rstrip(";")
            if normalized.lower().startswith("select"):
                conn.execute(f"EXPLAIN QUERY PLAN {normalized}")
            else:
                conn.execute(f"EXPLAIN {normalized}")
    except sqlite3.Error as e:
        return ValidationResult(False, str(e))

    return ValidationResult(True, None)


# --------------------------
# Prompting / Cleaning
# --------------------------
def build_prompt(nl_question: str, schema: str, shots: List[dict]) -> str:
    """Build a few-shot prompt including schema, examples, and a short instruction."""
    shot_texts = []
    for s in shots:
        shot_texts.append(f"### NL: {s['question']}\nSQL: {s['sql']}\n")

    prompt = (
        "You are a SQL generator for the given SQLite schema. "
        "Only output the SQL statement(s) as your final answer (no explanation).\n\n"
        "Instruction: When the NL question requests user information, always include columns "
        "`u.name` and `u.email` in the SELECT list unless the user explicitly requests other fields.\n\n"
        "Schema:\n"
        f"{schema}\n\n"
        "Examples:\n"
        f"{''.join(shot_texts)}\n"
        "### NL: "
        f"{nl_question}\n"
        "SQL:"
    )
    return prompt


def clean_model_output(text: str) -> str:
    """Sanitize the model output into a raw SQL string."""
    text = re.sub(r"```(?:sql)?\n", "", text, flags=re.IGNORECASE)
    text = re.sub(r"```\s*$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^\s*SQL:\s*", "", text, flags=re.IGNORECASE)
    text = text.strip()
    if not text.endswith(";"):
        text += ";"
    return text


# --------------------------
# Postprocessing heuristics
# --------------------------
def postprocess_sql(final_sql: str, nl_question: str) -> str:
    """Heuristics to fix/strengthen model SQL outputs.

    - Enforce DISTINCT when NL asks for distinct/unique.
    - Add u.email to projection if the NL explicitly requests email/contact or mentions users.
    - Make product category equality case-insensitive using COLLATE NOCASE.
    """
    sql = final_sql.strip()
    if not sql.endswith(";"):
        sql += ";"

    lowered_q = nl_question.lower()

    # Enforce DISTINCT if requested
    if ("distinct" in lowered_q or "unique" in lowered_q) and re.search(
        r"(?i)^(\s*select\s+)(?!distinct)", sql
    ):
        sql = re.sub(r"(?i)^(\s*select\s+)", r"\1DISTINCT ", sql, count=1)

    # If user asked for email/contact or asked about users, ensure u.email in projection
    wants_email = (
        "email" in lowered_q
        or "contact" in lowered_q
        or "user" in lowered_q
        or "users" in lowered_q
    )
    if wants_email and "u.email" not in sql.lower():
        sql = re.sub(
            r"(?i)^(\s*select\s+(?:distinct\s+)?)\s*([^\n\r;]+)",
            lambda m: f"{m.group(1)}{m.group(2)}, u.email",
            sql,
            count=1,
        )

    # Make product category comparison case-insensitive: append COLLATE NOCASE
    sql = re.sub(
        r"(p\.category\s*=\s*'[^']+')",
        lambda m: f"{m.group(1)} COLLATE NOCASE",
        sql,
        flags=re.IGNORECASE,
    )

    # Minor cleanup: remove duplicate commas if introduced
    sql = re.sub(r",\s*,", ",", sql)
    sql = re.sub(r"SELECT\s+,\s*", "SELECT ", sql, flags=re.IGNORECASE)

    return sql


# --------------------------
# Model invocation with token safety
# --------------------------
def estimate_prompt_tokens(prompt: str) -> int:
    """Rough token estimate for the prompt: use characters/4 heuristic."""
    return max(1, len(prompt) // 4)


def call_model(
    prompt: str,
    model_path: str,
    max_tokens: int = 128,
    n_ctx: int = 1024,
    n_threads: int = 2,
) -> str:
    """Call local Llama model via llama_cpp with automatic token cap.

    Ensures requested generation + prompt tokens do not exceed n_ctx.
    """
    if Llama is None:
        raise RuntimeError("llama-cpp-python is not available in this environment.")
    if not os.path.isfile(model_path):
        raise RuntimeError(f"Model file not found: {model_path}")

    prompt_tokens = estimate_prompt_tokens(prompt)
    SAFETY_MARGIN = 24  # some overhead for special tokens & internal use

    allowed = n_ctx - prompt_tokens - SAFETY_MARGIN
    if allowed < 16:
        # Prompt too long to safely generate with this context window — signal explicitly
        raise RuntimeError(
            f"Prompt too long for n_ctx={n_ctx}. Estimated prompt tokens={prompt_tokens}. "
            "Reduce prompt length or increase --n-ctx."
        )

    final_max_tokens = min(max_tokens, allowed)

    model = Llama(model_path=model_path, n_ctx=n_ctx, n_threads=n_threads)
    response = model(
        prompt=prompt,
        max_tokens=final_max_tokens,
        stop=["### NL:", "\n\n"],
        echo=False,
    )

    if isinstance(response, dict) and response.get("choices"):
        choice = response["choices"][0]
        raw_text = choice.get("text") or choice.get("message") or str(choice)
    else:
        raw_text = str(response)
    return raw_text.strip()


# --------------------------
# Logging
# --------------------------
def append_log(entry: dict, path: str = LOG_PATH) -> None:
    """Append a JSON object as a line to logs/pipeline.log (JSONL)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Truncate prompt snapshot for log size safety
    if "prompt_snapshot" in entry and isinstance(entry["prompt_snapshot"], str):
        entry["prompt_snapshot"] = entry["prompt_snapshot"][:PROMPT_SNAPSHOT_MAX]
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")


# --------------------------
# Retry / pipeline
# --------------------------
def ask_with_retries(
    nl_question: str,
    schema: str,
    conn: sqlite3.Connection,
    shots: List[dict],
    model_path: Optional[str] = None,
    mock: bool = False,
    n_ctx: int = 1024,
    n_threads: int = 2,
    max_tokens: int = 128,
) -> str:
    """Main loop: ask model, validate, and retry with repair prompts if necessary."""
    prompt = build_prompt(nl_question, schema, shots)
    last_error = None
    mock_outputs = []
    if mock:
        mock_outputs = [
            # simulate invalid first response, then corrected SQL
            "SELECT u.name, u.email FROMM users u JOIN orders o ON u.user_id = o.user_id "
            "JOIN products p ON p.product_id = o.product_id WHERE p.category = 'Electronics'",
            "SELECT u.name, u.email FROM users u JOIN orders o ON u.user_id = o.user_id "
            "JOIN products p ON p.product_id = o.product_id WHERE p.category = 'Electronics';",
        ]
    mock_idx = 0

    for attempt in range(1, RETRY_LIMIT + 1):
        start_ts = time.time()
        try:
            if mock:
                raw_sql = mock_outputs[min(mock_idx, len(mock_outputs) - 1)]
                mock_idx += 1
            else:
                raw_sql = call_model(
                    prompt,
                    model_path=model_path or DEFAULT_MODEL_PATH,
                    max_tokens=max_tokens,
                    n_ctx=n_ctx,
                    n_threads=n_threads,
                )
        except Exception as exc:  # capture invocation error (missing model, token overflow, etc.)
            duration = time.time() - start_ts
            entry = {
                "timestamp": time.time(),
                "attempt": attempt,
                "duration_s": duration,
                "raw_model_output": None,
                "cleaned_sql": None,
                "validation": {"is_valid": False, "error": str(exc)},
                "prompt_snapshot": prompt[:PROMPT_SNAPSHOT_MAX],
            }
            append_log(entry)
            print(f"Attempt {attempt}: model invocation error: {exc}", file=sys.stderr)
            last_error = str(exc)
            # small backoff
            time.sleep(0.2)
            continue

        duration = time.time() - start_ts
        cleaned_sql = clean_model_output(raw_sql)
        vr = validate_sql_with_sqlite(cleaned_sql, conn)
        log_line = f"Attempt {attempt}: {len(cleaned_sql)} chars, {duration:.2f}s, valid={vr.is_valid}"
        print(log_line, file=sys.stderr)

        entry = {
            "timestamp": time.time(),
            "attempt": attempt,
            "duration_s": duration,
            "raw_model_output": raw_sql,
            "cleaned_sql": cleaned_sql,
            "validation": {"is_valid": vr.is_valid, "error": vr.error},
            "prompt_snapshot": prompt[:PROMPT_SNAPSHOT_MAX],
        }
        append_log(entry)

        if not vr.is_valid:
            print(f"Validation error: {vr.error}", file=sys.stderr)
            last_error = vr.error
            repair_prompt = (
                prompt
                + "\n-- The following SQL produced an error when validated:\n"
                + f"{cleaned_sql}\n-- Error: {vr.error}\n"
                + "Please provide a corrected SQL query that answers the NL question and is valid for the schema. Only output the SQL.\nSQL:"
            )
            prompt = repair_prompt
            continue

        # Success
        return cleaned_sql

    raise RuntimeError(
        f"Failed to generate valid SQL after {RETRY_LIMIT} attempts. Last error: {last_error}"
    )


# --------------------------
# CLI / Entrypoint
# --------------------------
def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Tiny SQL Expert: NL -> SQL with validation loop"
    )
    parser.add_argument("question", type=str, help="Natural-language SQL question")
    parser.add_argument(
        "--raw", action="store_true", help="Print only the resulting SQL to stdout"
    )
    parser.add_argument(
        "--model", type=str, default=None, help="Path to local GGUF/GGML model"
    )
    parser.add_argument(
        "--mock", action="store_true", help="Use mock mode (no real model; for testing)"
    )
    parser.add_argument(
        "--n-threads", type=int, default=2, help="Threads to use for model"
    )
    parser.add_argument(
        "--n-ctx",
        type=int,
        default=1024,
        help="Context window for model (lower -> less memory)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=128,
        help="Maximum tokens to request from model",
    )
    parser.add_argument(
        "--save-sql", type=str, default=None, help="Save final SQL to this file"
    )
    args = parser.parse_args(argv)

    schema_path = "schema.sql"
    if not os.path.isfile(schema_path):
        print("ERROR: schema.sql not found in current directory.", file=sys.stderr)
        return 2

    with open(schema_path, "r", encoding="utf-8") as fh:
        schema_sql = fh.read()

    conn = create_inmemory_db(schema_sql)

    shots = [
        {
            "question": "List all users (name and email) who ordered products in category 'Electronics'.",
            "sql": "SELECT u.name, u.email FROM users u JOIN orders o ON u.user_id = o.user_id "
            "JOIN products p ON p.product_id = o.product_id WHERE p.category = 'Electronics';",
        },
        {
            "question": "Count number of orders per product.",
            "sql": "SELECT p.name, COUNT(*) AS orders_count FROM products p JOIN orders o ON p.product_id = o.product_id GROUP BY p.product_id;",
        },
        {
            "question": "Total revenue per user for completed orders.",
            "sql": "SELECT u.user_id, u.name, SUM(p.price * o.quantity) AS revenue FROM users u JOIN orders o ON u.user_id = o.user_id JOIN products p ON p.product_id = o.product_id WHERE o.status = 'completed' GROUP BY u.user_id;",
        },
        {
            "question": "List top 5 products by revenue.",
            "sql": "SELECT p.name, SUM(p.price * o.quantity) AS revenue FROM products p JOIN orders o ON p.product_id = o.product_id WHERE o.status = 'completed' GROUP BY p.product_id ORDER BY revenue DESC LIMIT 5;",
        },
    ]

    try:
        final_sql = ask_with_retries(
            args.question,
            schema_sql,
            conn,
            shots,
            model_path=args.model,
            mock=args.mock,
            n_ctx=args.n_ctx,
            n_threads=args.n_threads,
            max_tokens=args.max_tokens,
        )
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    final_sql = postprocess_sql(final_sql, args.question)

    if args.save_sql:
        with open(args.save_sql, "w", encoding="utf-8") as fh:
            fh.write(final_sql + "\n")

    if args.raw:
        sys.stdout.write(final_sql + "\n")
    else:
        print("=== FINAL SQL ===")
        print(final_sql)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# --------------------------
# Backwards-compatible wrapper names (for the Flask app)
# --------------------------


def validate_sql(sql: str, conn: sqlite3.Connection) -> tuple[bool, Optional[str]]:
    """
    Backwards-compatible wrapper for validation.
    Returns (is_valid, error_message_or_None).
    Delegates to validate_sql_with_sqlite defined earlier.
    """
    # validate_sql_with_sqlite returns ValidationResult dataclass
    try:
        vr = validate_sql_with_sqlite(sql, conn)
        return (vr.is_valid, vr.error)
    except Exception as exc:
        # If something unexpected happens, report as validation failure with message
        return (False, str(exc))


def clean_sql_output(text: str) -> str:
    """
    Backwards-compatible wrapper that returns cleaned SQL string.
    Delegates to clean_model_output defined earlier.
    """
    return clean_model_output(text)


def create_inmemory_db_alias(schema_sql: str) -> sqlite3.Connection:
    """
    Alias for create_inmemory_db to match possible alternate import names.
    """
    return create_inmemory_db(schema_sql)
