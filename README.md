# SR-QA

Code for **SR-QA**, a test collection for evidence finding in full-text studies, built
from the data-extraction tables of CC BY systematic reviews. Each question comes with the
CC BY full text of the study it was extracted from and the review authors' own answer,
copied verbatim from the table; 48,219 of the 166,308 questions also carry passage-level
relevance labels. No gold answer and no relevance label is written by a model.

* **Data** (CC BY 4.0): Zenodo, https://doi.org/10.5281/zenodo.23432598
* **Paper**: *SR-QA: A Test Collection for Evidence Finding in Full-Text Studies from
  Systematic Reviews*, ECIR 2027 resource track (under review)
* **Code** (MIT): this repository

This repository holds the code needed to rebuild the released items from the table maps,
to check them, to produce and score the retrieval labels and baselines, and to run and
judge the grounded-answering evaluation (with its prompts).

## Get the data

Download the data archives from the Zenodo deposit and unzip them in this folder. Their
paths mirror this repository, so every script finds its inputs:

```
release/v2/          items, table maps, review context, metadata, datasheet, MANIFEST.json
SOURCE_XML/          the CC BY full texts (JATS XML)
data/v2/...          retrieval labels, baseline runs, evaluation sample, parsed review tables
```

## Quick start

```bash
python -m venv .venv && .venv/Scripts/activate      # Windows; source .venv/bin/activate elsewhere
pip install -r requirements.txt
python scripts/score_trec.py data/v2/retrieval_labels/test-2026/qrels.trec \
                             data/v2/eval/retrieval_full/runs/bm25.trec
python scripts/rebuild_check.py      # rebuilds every item from the table maps
python -m pytest tests/
```

```python
import gzip, json
items = [json.loads(l) for l in gzip.open('release/v2/data/test-2026.jsonl.gz', 'rt', encoding='utf-8')]
print(items[0]['question'], '->', items[0]['gold'])
```

## What is here

| Path | Purpose |
|---|---|
| `scripts/10_build_items.py`, `11_filter_items.py`, `13_make_splits.py` | build items from the table maps (answers copied verbatim from cells), filter them, cut the splits by each review's online date |
| `scripts/17_check_dataset.py`, `15_bias_table.py`, `18_answerability_check.py` | the automatic checks, the bias table and the answerability check reported in the paper |
| `scripts/20_freeze_release.py` | freezes a release with SHA-256 checksums (byte-reproducible) |
| `scripts/rebuild_check.py` | rebuilds every released item from the released maps and tables and checks it is identical |
| `scripts/eval/retrieval.py` | passages, relevance labels and the retrieval baselines (BM25, MedCPT, BGE-M3, fusion, reranker) |
| `scripts/score_trec.py` | scores a TREC run against the relevance labels (standard library only) |
| `scripts/21_eval_sample.py`, `scripts/eval/run_answers.py`, `judge_answers.py`, `crossjudge.py`, `providers.py` | the grounded-answering evaluation: sample, answering prompt, the two blind judges and the cross-judge |
| `scripts/analysis/` | confidence intervals (cluster bootstrap over reviews) and paired tests; how many passages the dense retrievers truncate (`passage_lengths.py`) |
| `scripts/02_list_tables.py`, `07_run_tables.py`, `table_map_*.py`, `eval/jats_source.py`, `_env.py` | shared code the scripts above import (table parsing, map schema and validation, the paper reader, API keys from the environment) |
| `docs/DATASHEET.md` | the datasheet |
| `docs/SUPPLEMENT.md` | supplementary notes to the ECIR 2027 paper (detail cut for space) |
| `tests/` | unit tests (`python -m pytest tests/`) |

Steps that call a model (answering, judging) read their API keys from the environment
(see `.env.example`). The table maps themselves were produced by Claude Sonnet 5 and are
released, so rebuilding the items needs no model.

## Errata

Please report errors through GitHub issues. Corrections are released as new versions under
the same Zenodo concept DOI, and earlier versions stay available unchanged.

## Licence

Code: MIT (`LICENSE`). Data: CC BY 4.0, distributed separately on Zenodo.
