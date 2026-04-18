"""
kappa_validation.py
===================
Computes Cohen's kappa between the rule-based classifier's labels and human
labels on a stratified subset.

This script is the SECOND step. First, you need:
  1. The sampling file (kappa_sample_to_label.csv), produced separately
  2. That file filled in by a human annotator in the 'human_label' column

Then run this script to get the kappa score and confusion matrix.

Usage:
    python kappa_validation.py kappa_sample_labeled.csv

Outputs:
    - Prints Cohen's kappa (overall and per-model)
    - Prints agreement counts
    - Writes kappa_disagreements.csv for rows where classifier != human
    - Writes kappa_confusion.csv with the confusion matrix

Interpretation of kappa:
    < 0.20  poor
    0.21 - 0.40  fair
    0.41 - 0.60  moderate
    0.61 - 0.80  substantial
    0.81 - 1.00  almost perfect
Target for a publishable classifier: >= 0.60
"""

import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path


CATEGORIES = [
    "FRAMEWORK_DUMP", "CLARIFICATION_PLUS", "ANSWER_FIRST_CLARIFY",
    "PURE_CLARIFICATION", "CAPABILITY_DISCLAIMER", "CLARIFYING_REFUSAL",
    "TERMINAL_REFUSAL", "MEDICAL_LEGAL_DEFLECT", "MEANTIME_HYBRID", "OTHER",
]


def cohens_kappa(pairs):
    """
    pairs: list of (rater1_label, rater2_label) tuples
    Returns Cohen's kappa using the standard formula:
        kappa = (po - pe) / (1 - pe)
    """
    n = len(pairs)
    if n == 0:
        return 0.0

    # Observed agreement
    agree = sum(1 for a, b in pairs if a == b)
    po = agree / n

    # Expected agreement by chance
    r1_counts = Counter(a for a, b in pairs)
    r2_counts = Counter(b for a, b in pairs)
    labels = set(r1_counts) | set(r2_counts)
    pe = sum((r1_counts[l] / n) * (r2_counts[l] / n) for l in labels)

    if pe == 1.0:
        return 1.0 if po == 1.0 else 0.0
    return (po - pe) / (1 - pe)


def load_labeled_sample(path):
    rows = []
    missing = 0
    with open(path) as f:
        for row in csv.DictReader(f):
            if not row.get("human_label", "").strip():
                missing += 1
                continue
            rows.append(row)
    if missing:
        print(f"WARNING: {missing} rows had no human_label; skipping them.")
    return rows


def write_disagreements(rows, out_path):
    disagreements = [r for r in rows if r["classification"] != r["human_label"]]
    if not disagreements:
        print("No disagreements — perfect agreement.")
        return
    fields = list(disagreements[0].keys())
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(disagreements)
    print(f"Wrote {len(disagreements)} disagreement rows to {out_path}")


def write_confusion(rows, out_path):
    # Rows = classifier label, columns = human label
    confusion = defaultdict(lambda: defaultdict(int))
    for r in rows:
        confusion[r["classification"]][r["human_label"]] += 1

    labels = sorted(set(CATEGORIES) | set(
        r["classification"] for r in rows
    ) | set(r["human_label"] for r in rows))

    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["classifier \\ human"] + labels)
        for row_label in labels:
            row = [row_label] + [confusion[row_label][col_label] for col_label in labels]
            w.writerow(row)
    print(f"Wrote confusion matrix to {out_path}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python kappa_validation.py <labeled_sample.csv>")
        sys.exit(1)

    path = Path(sys.argv[1])
    if not path.exists():
        print(f"ERROR: {path} not found.")
        sys.exit(1)

    rows = load_labeled_sample(path)
    if not rows:
        print("ERROR: no labeled rows found.")
        sys.exit(1)

    # Overall kappa
    pairs = [(r["classification"], r["human_label"]) for r in rows]
    k = cohens_kappa(pairs)
    agree = sum(1 for a, b in pairs if a == b)
    print(f"\n=== OVERALL (n={len(pairs)}) ===")
    print(f"  Cohen's kappa:       {k:.3f}")
    print(f"  Raw agreement:       {agree}/{len(pairs)} = {100*agree/len(pairs):.1f}%")

    # Per-model kappa
    print("\n=== BY MODEL ===")
    by_model = defaultdict(list)
    for r in rows:
        by_model[r["model"]].append((r["classification"], r["human_label"]))
    for model, m_pairs in sorted(by_model.items()):
        mk = cohens_kappa(m_pairs)
        ma = sum(1 for a, b in m_pairs if a == b)
        print(f"  {model:10s} n={len(m_pairs):3d}  kappa={mk:.3f}  agreement={100*ma/len(m_pairs):.1f}%")

    # Per-category kappa (treating each category as a binary problem)
    print("\n=== PER-CATEGORY (one-vs-rest) ===")
    for cat in CATEGORIES:
        bin_pairs = [
            ("yes" if a == cat else "no", "yes" if b == cat else "no")
            for a, b in pairs
        ]
        if any(p[0] == "yes" or p[1] == "yes" for p in bin_pairs):
            ck = cohens_kappa(bin_pairs)
            print(f"  {cat:25s} kappa={ck:.3f}")

    # Artifacts
    out_dir = path.parent
    write_disagreements(rows, out_dir / "kappa_disagreements.csv")
    write_confusion(rows, out_dir / "kappa_confusion.csv")

    print("\nInterpretation guide:")
    print("  < 0.20: poor  |  0.21-0.40: fair  |  0.41-0.60: moderate")
    print("  0.61-0.80: substantial  |  0.81-1.00: almost perfect")


if __name__ == "__main__":
    main()
