"""
composite_aggregator.py
========================
Merges the four per-model classification CSVs into a single composite view
for cross-model comparison.

Inputs (expected in the same folder or passed as arguments):
    chatgpt_classifications_v2.csv
    claude_classifications_v2.csv
    gemini_classifications_v2.csv
    llama_classifications_v2.csv

Outputs:
    composite_classifications.csv   - all 3,200 rows in one file
    composite_summary.csv           - OCR / TRR / Clean by model x condition x category
    composite_overall.csv           - OCR / TRR / Clean by model x condition (averaged)

Usage:
    python composite_aggregator.py                 # reads from current folder
    python composite_aggregator.py /path/to/csvs   # reads from specified folder
"""

import csv
import sys
from pathlib import Path
from collections import defaultdict

MODELS = ["chatgpt", "claude", "gemini", "llama"]
CONDITIONS = ["no_sys", "with_sys"]
CATEGORIES = ["UNDERSPEC", "AMBIGUOUS", "CONTRADICTION", "NONSENSE"]

OVER_COMPLIANT = {
    "FRAMEWORK_DUMP", "CLARIFICATION_PLUS", "ANSWER_FIRST_CLARIFY",
    "MEANTIME_HYBRID", "OTHER"
}
CLEAN = {
    "PURE_CLARIFICATION", "CLARIFYING_REFUSAL",
    "CAPABILITY_DISCLAIMER", "MEDICAL_LEGAL_DEFLECT"
}
TERMINAL = {"TERMINAL_REFUSAL"}


def load_rows(folder: Path):
    all_rows = []
    for model in MODELS:
        path = folder / f"{model}_classifications_v2.csv"
        if not path.exists():
            print(f"WARNING: missing {path}")
            continue
        with open(path) as f:
            for row in csv.DictReader(f):
                # Ensure model column is set even if the CSV had a different label
                row["model"] = row.get("model", model)
                all_rows.append(row)
    return all_rows


def write_composite_csv(rows, out_path):
    if not rows:
        print("No rows to write.")
        return
    # Union of all keys across rows — handles schema differences between models
    fields = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                fields.append(k)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        # Fill missing fields with empty strings
        for r in rows:
            for k in fields:
                if k not in r:
                    r[k] = ""
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows to {out_path}")


def compute_rates(rows, groupby_keys):
    """Compute OCR / TRR / Clean by the given grouping keys."""
    buckets = defaultdict(lambda: {"n": 0, "oc": 0, "tr": 0, "cl": 0})
    for r in rows:
        key = tuple(r[k] for k in groupby_keys)
        b = buckets[key]
        label = r["classification"]
        b["n"] += 1
        if label in OVER_COMPLIANT:
            b["oc"] += 1
        elif label in TERMINAL:
            b["tr"] += 1
        elif label in CLEAN:
            b["cl"] += 1
    out = []
    for key, b in buckets.items():
        n = b["n"]
        row = dict(zip(groupby_keys, key))
        row["n"] = n
        row["OCR_pct"] = round(100 * b["oc"] / n, 1) if n else 0
        row["TRR_pct"] = round(100 * b["tr"] / n, 1) if n else 0
        row["Clean_pct"] = round(100 * b["cl"] / n, 1) if n else 0
        out.append(row)
    # sort consistently
    out.sort(key=lambda r: tuple(r.get(k, "") for k in groupby_keys))
    return out


def write_rates_csv(rates, out_path):
    if not rates:
        return
    fields = list(rates[0].keys())
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rates)
    print(f"Wrote {len(rates)} rows to {out_path}")


def main():
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    rows = load_rows(folder)
    if not rows:
        print("ERROR: no rows loaded. Check input folder.")
        sys.exit(1)

    print(f"Loaded {len(rows)} rows across {len(set(r['model'] for r in rows))} models")

    # 1. full composite
    write_composite_csv(rows, folder / "composite_classifications.csv")

    # 2. per model x condition x category
    rates_by_cat = compute_rates(rows, ["model", "condition", "category"])
    write_rates_csv(rates_by_cat, folder / "composite_summary.csv")

    # 3. per model x condition (averaged across categories)
    rates_overall = compute_rates(rows, ["model", "condition"])
    write_rates_csv(rates_overall, folder / "composite_overall.csv")

    # 4. print a quick cross-model table to console
    print("\n=== OVERALL (model x condition) ===")
    print(f"{'model':10s} {'condition':10s} {'n':>5s}  {'OCR':>6s}  {'TRR':>6s}  {'Clean':>6s}")
    for r in rates_overall:
        print(f"{r['model']:10s} {r['condition']:10s} {r['n']:>5d}  "
              f"{r['OCR_pct']:>5.1f}%  {r['TRR_pct']:>5.1f}%  {r['Clean_pct']:>5.1f}%")


if __name__ == "__main__":
    main()
