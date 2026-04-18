# A Cross-Model Study of Over-Compliance in Large Language Models

This repository accompanies the paper *A Cross-Model Study of Over-Compliance in Large Language Models*. It contains the benchmark, the runner scripts, the raw responses from four frontier language models, the rule-based classifier, the composite cross-model outputs, the Cohen's kappa validation artifacts, and the paper source.

## Overview

The study evaluates over-compliance in large language models, defined as the generation of substantive content when the user input is underspecified, ambiguous, internally contradictory, or semantically incoherent. Four frontier models (GPT-4.1-mini, Claude Haiku 4.5, Gemini 2.5 Flash, Llama 3.3 70B Instruct) were tested on a 400-prompt benchmark under two system-prompt conditions, producing 3,200 responses. A deterministic rule-based classifier mapped each response to a nine-category taxonomy, and the resulting Over-Compliance Rate and Terminal Refusal Rate were reported per model and condition. Inter-rater reliability was validated on a 112-row stratified human-labeled sample, yielding a Cohen's kappa of 0.725.

## Repository Structure

```
.
├── README.md
├── paper.tex
├── paper.pdf
├── pipeline.png
│
├── benchmark/
│   └── Dataset_and_Instructions.txt
│
├── scripts/
│   ├── classify_responses_v2.py
│   ├── composite_aggregator.py
│   ├── kappa_sample_builder.py
│   └── kappa_validation.py
│
├── llm_data/
│   ├── chatgpt/
│   ├── claude/
│   ├── gemini/
│   └── llama/
│
├── composite/
│   ├── composite_classifications.csv
│   ├── composite_summary.csv
│   └── composite_overall.csv
│
└── kappa/
    ├── kappa_sample_to_label.csv
    ├── kappa_sample_labeled.csv
    ├── kappa_confusion.csv
    └── kappa_disagreements.csv
```

## Folder Contents

**`benchmark/`** — The 400-prompt benchmark file, 100 prompts each across underspecification, ambiguity, contradiction, and nonsense, plus the system-prompt instruction text and the reproduction steps.

**`scripts/`** — Analysis code. `classify_responses_v2.py` is the nine-category rule-based classifier. `composite_aggregator.py` merges the four per-model classification CSVs into one cross-model dataset. `kappa_sample_builder.py` draws a stratified 112-row sample for human labeling. `kappa_validation.py` computes Cohen's kappa between classifier and human labels.

**`llm_data/`** — Per-model folders, each containing the provider-specific API runner scripts, the model configuration JSON, the raw response files for both conditions, and the per-response classification CSV and summary text file produced by the classifier. Each model folder holds seven files.

**`composite/`** — Cross-model merged outputs. `composite_classifications.csv` is the unified 3,200-row table across all four models. `composite_summary.csv` reports rates by model × condition × category. `composite_overall.csv` reports rates averaged across categories.

**`kappa/`** — Inter-rater reliability artifacts. `kappa_sample_to_label.csv` is the blank 112-row template, `kappa_sample_labeled.csv` holds the filled-in labels used to compute the reported κ = 0.725, `kappa_confusion.csv` is the classifier-vs-human confusion matrix, and `kappa_disagreements.csv` is the list of 23 rows on which the two differed.

## Reproducing the Results

Each of the 400 prompts was submitted individually to each model. All 100 prompts within a single category were evaluated within one chat thread so the model had access to its own prior responses, and a fresh chat session was opened for each category. This was repeated for all four categories and both system-prompt conditions across all four models.

1. Run the eight API runner scripts in `llm_data/<model>/` to regenerate the raw response JSON files. Requires the respective provider API credentials.
2. Run `scripts/classify_responses_v2.py` to produce the per-model classification CSVs and summary text files.
3. Run `scripts/composite_aggregator.py` from the project root to merge results into the `composite/` folder.
4. Run `scripts/kappa_sample_builder.py` to regenerate the sampling file, then hand-label the `human_label` column, and run `scripts/kappa_validation.py` to compute agreement.

## Model Configurations

All four models were run at temperature 1.0 with a 1,000-token output limit. GPT-4.1-mini and Gemini 2.5 Flash were seeded at 42 where the API permitted. Full per-model configuration is recorded in each `llm_data/<model>/<model>_model_attributes.json`.

## Pipeline

The end-to-end data flow from benchmark through classifier, composite aggregation, and kappa validation is documented in `pipeline.png`.

## Paper

The manuscript source and compiled PDF are at the repository root (`paper.tex`, `paper.pdf`). The submission targets *Transactions on Machine Learning Research* (TMLR).

## License

All code, data, and the paper are released under the MIT License.
