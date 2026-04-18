"""
kappa_sample_builder.py
=======================
Builds a stratified random subset of the classified responses for human
annotation. Stratification ensures each model x category cell is represented,
so kappa is not dominated by the most common classes.

Reads:
    composite_classifications.csv (from composite_aggregator.py)
    The 8 raw JSON response files (to pull full response text)

Writes:
    kappa_sample_to_label.csv

The output file has the classifier's label pre-filled. Leave the 'human_label'
column blank. Open in a spreadsheet, read each response, and fill in the
category from the nine-category taxonomy. Save as
kappa_sample_labeled.csv and run kappa_validation.py on it.

Sampling plan: per model x category, sample N_PER_CELL rows (default 7), which
gives 4 models x 4 categories x 7 = 112 rows total across both conditions.
Rows are drawn across both +sys and -sys conditions proportionally.
"""

import csv
import json
import random
import sys
from pathlib import Path
from collections import defaultdict


N_PER_CELL = 7   # rows per (model, category); adjust for a bigger sample
SEED = 42

CATEGORIES_IN = ["UNDERSPEC", "AMBIGUOUS", "CONTRADICTION", "NONSENSE"]
MODELS = ["chatgpt", "claude", "gemini", "llama"]

# The nine valid labels the human may use
VALID_LABELS = [
    "FRAMEWORK_DUMP", "CLARIFICATION_PLUS", "ANSWER_FIRST_CLARIFY",
    "PURE_CLARIFICATION", "CAPABILITY_DISCLAIMER", "CLARIFYING_REFUSAL",
    "TERMINAL_REFUSAL", "MEDICAL_LEGAL_DEFLECT", "MEANTIME_HYBRID", "OTHER",
]


def load_responses(raw_folder: Path):
    """Load (model, condition, category, prompt_number) -> response text."""
    mapping = {}
    for model in MODELS:
        for cond_label, cond_suffix in [("no_sys", "no"), ("with_sys", "with")]:
            candidates = [
                raw_folder / f"{model}_resp_{cond_suffix}_systemPrompt.json",
                raw_folder / f"{model}_{cond_suffix}.json",
            ]
            path = next((p for p in candidates if p.exists()), None)
            if not path:
                print(f"WARNING: no raw JSON found for {model} {cond_label}")
                continue
            data = json.loads(path.read_text())
            for r in data:
                key = (model, cond_label, r["category"], int(r["prompt_number"]))
                mapping[key] = r["response"]
    return mapping


def build_sample(composite_path: Path, responses: dict, out_path: Path):
    rng = random.Random(SEED)

    # Bucket all rows by (model, category), preserving condition info
    buckets = defaultdict(list)
    with open(composite_path) as f:
        for row in csv.DictReader(f):
            key = (row["model"], row["category"])
            buckets[key].append(row)

    # Sample N_PER_CELL per cell
    sampled = []
    for (model, category), rows in sorted(buckets.items()):
        rng.shuffle(rows)
        sampled.extend(rows[:N_PER_CELL])

    # Attach response text
    enriched = []
    for r in sampled:
        key = (r["model"], r["condition"], r["category"], int(r["prompt_number"]))
        resp_text = responses.get(key, "")
        enriched.append({
            "model": r["model"],
            "condition": r["condition"],
            "category": r["category"],
            "prompt_number": r["prompt_number"],
            "prompt": r["prompt"],
            "response": resp_text,
            "classification": r["classification"],
            "human_label": "",
        })

    # Shuffle the final list so the annotator cannot infer model from position
    rng.shuffle(enriched)

    # Prepend a header row explaining the valid labels
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        # Meta row (stripped by kappa_validation.py which starts from DictReader)
        w.writerow([f"# Valid labels: {', '.join(VALID_LABELS)}"])
        fields = list(enriched[0].keys())
        w.writerow(fields)
        for row in enriched:
            w.writerow([row[k] for k in fields])

    print(f"Wrote {len(enriched)} rows to {out_path}")
    print(f"Valid labels: {', '.join(VALID_LABELS)}")


def main():
    composite_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("composite_classifications.csv")
    raw_folder = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(".")
    out_path = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("kappa_sample_to_label.csv")

    if not composite_path.exists():
        print(f"ERROR: {composite_path} not found. Run composite_aggregator.py first.")
        sys.exit(1)

    responses = load_responses(raw_folder)
    if not responses:
        print("WARNING: no raw responses loaded. Sample will be missing response text.")

    build_sample(composite_path, responses, out_path)


if __name__ == "__main__":
    main()
