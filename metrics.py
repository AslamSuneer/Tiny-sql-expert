"""
metrics.py

Run a set of natural-language prompts through the tiny_sql_expert pipeline
and report pass/fail rate. Write per-prompt records to logs/metrics.jsonl.

Usage (mock):
    venv\Scripts\activate
    python metrics.py --mock

Usage (real model):
    python metrics.py --model ./models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf --n-ctx 1024 --n-threads 2
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import List

from tiny_sql_expert import (
    create_inmemory_db,
    ask_with_retries,
)

DEFAULT_PROMPTS: List[str] = [
    "List users who bought electronics",
    "List distinct users (name and email) who bought electronics",
    "Count orders per product",
    "Total revenue per user for completed orders",
    "Top 3 products by revenue",
    "List orders placed in March 2025",
    "How many products are in category 'Home'?",
    "List users who bought products priced above 100",
    "List user emails who have pending orders",
    "List products with zero sales"
]


def load_schema() -> str:
    with open("schema.sql", "r", encoding="utf-8") as fh:
        return fh.read()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--mock", action="store_true", help="Use mock mode")
    parser.add_argument("--n-ctx", type=int, default=1024)
    parser.add_argument("--n-threads", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=128)
    args = parser.parse_args()

    schema = load_schema()
    conn = create_inmemory_db(schema)

    # Ensure logs dir
    os.makedirs("logs", exist_ok=True)
    out_path = "logs/metrics.jsonl"

    results = []
    success_count = 0
    start_all = time.time()

    for prompt in DEFAULT_PROMPTS:
        t0 = time.time()
        try:
            final_sql = ask_with_retries(
                prompt,
                schema,
                conn,
                shots=[],  # use built-in shots inside the pipeline or pass similar small shots
                model_path=args.model,
                mock=args.mock,
                n_ctx=args.n_ctx,
                n_threads=args.n_threads,
                max_tokens=args.max_tokens,
            )
            ok = True
            err = None
        except Exception as exc:
            final_sql = None
            ok = False
            err = str(exc)

        duration = time.time() - t0
        entry = {
            "prompt": prompt,
            "success": ok,
            "error": err,
            "final_sql": final_sql,
            "duration_s": duration,
            "timestamp": time.time(),
        }
        results.append(entry)
        # append to JSONL
        with open(out_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")

        if ok:
            success_count += 1

        # small pause to avoid hammering local resources
        time.sleep(0.05)

    total_time = time.time() - start_all
    print(f"Ran {len(DEFAULT_PROMPTS)} prompts in {total_time:.2f}s")
    print(f"Success rate: {success_count}/{len(DEFAULT_PROMPTS)} = {success_count/len(DEFAULT_PROMPTS):.2%}")
    print(f"Per-prompt results written to {out_path}")


if __name__ == "__main__":
    main()
