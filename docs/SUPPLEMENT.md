# SR-QA: supplementary notes to the ECIR 2027 paper

Detail left out of the paper for space. Every number here is generated from the same
result files as the paper's numbers; section numbers refer to the paper.

## 0. Construction funnel (Sect. 4.3)

Papers counts papers found in PMC. Every step, with its counts, is also in the release's
`quality/funnel.csv`.

| Step | Items | Papers | Reviews | Why |
|---|---:|---:|---:|---|
| Items from accepted maps | 807,431 | - | 3,305 | one per study row and extraction column |
| Citation in the study's row | 799,305 | - | 3,298 | the only link used; no title matching |
| Paper in PMC | 330,733 | 29,188 | 3,277 | full text obtainable |
| Paper is CC BY | 181,287 | 16,007 | 3,046 | full text redistributable |
| File holds the full text | 181,218 | 16,000 | 3,046 | some deposits are metadata only |
| Origin reported, score at least 3 | 174,569 | 15,947 | 3,037 | answer should be in the paper |
| Gold not blank | 168,235 | 15,944 | 3,037 | a blank or footnote mark cannot be graded |
| No duplicate in a review | 166,308 | 15,944 | 3,037 | one study in two tables |

## 1. Choosing and checking the map model (Sect. 5)

- Two candidate map models, Claude Haiku 4.5 and Claude Sonnet 5, mapped 200
  pilot tables, chosen to over-represent messy tables. We corrected the maps of
  60 of these tables by hand.
- Each correction started from one model's map, picked at random and not shown to the
  corrector. A corrected map leans towards its starting map, so each model is scored only on
  the tables whose correction started from the other model's map: 35 tables for
  Sonnet and 25 for Haiku.
- Item recall against the hand-corrected maps, same (first) prompt: Sonnet
  95.7%, Haiku 42.2%. Haiku mostly skipped rows of long tables.
- Sonnet with the final prompt, on the same 35 tables: recall 93.8%, precision
  96.7%.
- Stability across the prompt revision: on the 193 pilot tables Sonnet mapped usably
  with both the first and the final prompt, it made the same accept decision for
  97.4% and gave the same origin (reported, calculated, judgement, ...) to
  97.6% of 1,005 extraction columns.

## 2. Checks on the released items (Sect. 5)

A hand audit of 100 test and development items found the question clear in
100, the gold exactly the value the table gives for that study in
99, and the right study linked in 100. The audit checks the
copy against the review, not against the paper. The one error exposed a construction defect
(study names merged across columns), which was fixed before release.

All 10 automatic checks (`scripts/17_check_dataset.py`) pass on every item. Each
recomputes from the source files:

1. every item ID is unique;
2. the gold re-parses as whole cell texts of its column;
3. rebuilding from the released maps and tables gives the same question and gold;
4. the item's citation comes from the study column (or an identifier column) of its own
   study's rows, never from a content column;
5. the linked paper is present, CC BY, and has a body;
6. every filter holds: origin reported and score at least 3, gold not blank, no duplicate
   within a review, no group label giving the answer;
7. splits are consistent: one split per review, 2026 splits hold only reviews online in
   2026, and no source paper is shared between an evaluation split and any other split;
8. the funnel's final count equals the number of items in the splits;
9. no item draws on a row that is a single merged heading cell;
10. no gold comes from the row's own study-label cell.

The answerability check finds the gold in the source paper for 78.5% of items,
as an exact phrase, through all of its numbers, or through most of its content words.

## 3. Hand check of the relevance labels (Sect. 5 and 7)

- 195 question-passage pairs from test-2026 were judged blind to the label. They
  combine passages the labels call relevant and, for other questions, the highest-ranked
  passage of the best retrieval baseline that the labels call not relevant. 15
  pairs could not be judged either way.
- Phrase matches (grade 2) were right in 47/47 (100.0%,
  95% CI [92.4, 100.0]). Number matches (grade 1) were right in 40/47
  (85.1%, [72.3, 92.6]). Weighted by the share of each grade, label
  precision is 96.1%.
- Incompleteness: the top-ranked unlabelled passage stated the answer in 19/86 cases
  (22.1%, [14.6, 31.9]).
- Agreement: two further annotators judged an overlapping subset. Their Cohen's kappa with
  our judgements (states the answer or not) was 0.43 for the second annotator
  and 0.76 for the third. Fleiss' kappa over all three was
  0.53, on 87 pairs. On top-ranked unlabelled passages, the second
  annotator judged 18/44 to state the answer, against 11/47
  for us and 10/49 for the third.
- Corrected Success@10 of the best pipeline (BGE-M3 + re-ranker): on the queries it fails
  by our labels, its top-ranked unlabelled passage states the answer in 4/27 judged
  cases (14.8%). Counting that share of its failures as successes raises the measured
  0.685 to 0.732 [0.704, 0.788].

## 4. Checking the LLM judge (Sect. 7)

- Second judge: Claude Sonnet 5.5 re-judged the answers of four models (DeepSeek
  V3.2, Gemma 4, GLM-5, Qwen3.5-27B) on 100 test-2026 papers.
- Correctness: the two judges agree on 93.9-95.4% of verdicts.
  There is no same-lab (Google) advantage for Gemma: +0.3 points, [−1.9, +3.0].
- Faithfulness: the Claude judge is far more lenient (95.1-96.5%
  faithful, against 77.7-84.7% for Gemini). Taken at face
  value, the Gemini judge favours Gemma by +5.2 points [+1.8, +8.6]
  relative to the other models. The gap lies in answers whose quote is not found in the
  paper, which the rubric counts as grounding failures; Gemini applies that rule and Claude
  mostly does not. Where the quote is found, the same-lab difference is −0.9
  points [−2.7, +0.7].
- Human check: one author judged 100 decisions blind to both judges. The author
  agreed with Gemini on 25/25 faithfulness decisions with a found quote, and on
  25/25 correctness decisions. For answers whose quote is not found, the author
  judged the section text to support 43/46 (93.5%, [82.5, 97.8]), against
  10.9% for Gemini and 91.3% for Claude on the same items. Where the judges
  split, the author sided with Claude in 36/37 cases.

## 5. Answering setup (Sect. 6)

- Every model is served only by providers with fp8 or better weights, a context of at least
  128k tokens and schema-constrained output. The serving provider is recorded per request.
- Decoding uses temperature 0. One reply answers all of a study's questions. A reply that
  answers fewer than half of them is re-sent once at temperature 0.5 (2.1% of
  requests). Reasoning modes are off for every model.
- Dense retrievers truncate passages at 256 tokens. This shortens at most
  0.3% of test-2026 passages (2.5% of relevant ones), mostly
  long sentences (`scripts/analysis/passage_lengths.py`).

## 6. Who published the source papers (Sect. 5)

Restricting to CC BY full texts changes the publisher mix (total variation distance
0.54). The released source papers come from Springer Nature (32%),
MDPI (20%) and Frontiers (13%); Elsevier falls from
20% of candidate papers to 3%. It also changes the years
(TVD 0.30): 75% of source papers appeared from 2020 on.

## 7. Release integrity and maintenance (Sect. 8)

- Each version is frozen with a manifest of SHA-256 checksums over every file, every
  pipeline script and every source document. Freezing twice yields byte-identical files.
- Versions share one Zenodo concept DOI, and earlier versions stay available unchanged.
- Report errors as GitHub issues. They are fixed in the pipeline and released as a new
  version with a change note.

## 8. The retrieval set (Sect. 4.5)

Why items are left out, over all 166,308 items:

| Outcome | Items | % |
|---|---:|---:|
| Labelled | 48,219 | 29.0 |
| No passage matches | 84,520 | 50.8 |
| More than ten passages match | 18,914 | 11.4 |
| Gold "not reported" | 11,013 | 6.6 |
| Yes/no gold | 2,314 | 1.4 |
| No letter or digit | 1,328 | 0.8 |

On test-2026, a query has a median of 2 relevant passages (mean
3.1; a table value counts as both its row and its cell), and
30.6% of queries have exactly one.

Labelled items against the other items with a gold value. Where the best evidence (the
highest-graded relevant passage) sits is known only for labelled items.

| Property | Labelled | Other |
|---|---:|---:|
| Mean gold length (characters) | 20 | 38 |
| Gold has a digit (%) | 65 | 47 |
| Gold verbatim in the paper (%) | 69 | 25 |
| Result question (%) | 17 | 18 |
| Best evidence in a table (%) | 29 | - |

For the items no passage matches, the answerability check, a looser search of the whole
article, finds:

- the gold verbatim for 7.3%, though never as a whole word or phrase within one
  passage (e.g. only inside a longer number);
- all of its numbers for 26.9%;
- most of its words for 27.7%, which suggests a reworded value;
- nothing for 38.0%, as expected when a value appears only in a figure or
  supplementary file, or was derived by the reviewers.

These causes were not checked by hand.

## 9. Notes on the comparison table (Table 2)

- BioASQ's size counts the Task 13b training questions of 2025 (Nentidis et al., 2025).
- LEADSInstruct's size counts its four extraction subsets, not its screening and search
  instructions. It releases full text for part of three extraction subsets, and titles
  with abstracts otherwise.

## 10. Pairwise significance tests (Sect. 6 and 7)

Each row compares two systems on test-2026, higher-scoring first. The difference is
paired: in each of 2,000 bootstrap replicates both systems are scored on the same
resampled reviews. The 95% interval is the percentile interval of the replicate
differences; p is two-sided and Holm-adjusted within each table, and "significant"
means an adjusted p below 0.05. With 2,000 replicates, p cannot be resolved below
0.001. The other retrieval metrics are in `data/v2/analysis/retrieval_results.json` in
the release.

**Retrieval, Success@10** (15 pairs of runs; difference in Success@10)

| Higher | Lower | Difference (Success@10) | 95% interval | Holm p | Significant |
|---|---|---|---|---|---|
| BGE-M3 → rerank | BM25 | 0.270 | [0.251, 0.290] | <0.001 | yes |
| BGE-M3 → rerank | MedCPT | 0.218 | [0.197, 0.237] | <0.001 | yes |
| BGE-M3 → rerank | BM25 + MedCPT | 0.199 | [0.175, 0.219] | <0.001 | yes |
| BGE-M3 | BM25 | 0.169 | [0.150, 0.190] | <0.001 | yes |
| BGE-M3 → rerank | BM25 + BGE-M3 | 0.147 | [0.133, 0.162] | <0.001 | yes |
| BM25 + BGE-M3 | BM25 | 0.122 | [0.110, 0.137] | <0.001 | yes |
| BGE-M3 | MedCPT | 0.118 | [0.098, 0.136] | <0.001 | yes |
| BGE-M3 → rerank | BGE-M3 | 0.100 | [0.088, 0.111] | <0.001 | yes |
| BGE-M3 | BM25 + MedCPT | 0.099 | [0.076, 0.118] | <0.001 | yes |
| BM25 + MedCPT | BM25 | 0.071 | [0.055, 0.090] | <0.001 | yes |
| BM25 + BGE-M3 | MedCPT | 0.070 | [0.052, 0.089] | <0.001 | yes |
| MedCPT | BM25 | 0.052 | [0.031, 0.073] | <0.001 | yes |
| BM25 + BGE-M3 | BM25 + MedCPT | 0.051 | [0.033, 0.066] | <0.001 | yes |
| BGE-M3 | BM25 + BGE-M3 | 0.047 | [0.035, 0.059] | <0.001 | yes |
| BM25 + MedCPT | MedCPT | 0.019 | [0.005, 0.031] | 0.007 | yes |

**Grounded answering, faithfulness** (45 pairs of models; difference in percentage points)

| Higher | Lower | Difference (points) | 95% interval | Holm p | Significant |
|---|---|---|---|---|---|
| Gemma 4 31B | Qwen3.5-9B | 19.6 | [17.5, 21.9] | <0.001 | yes |
| Gemma 4 31B | Llama 4 Scout | 16.9 | [14.0, 19.8] | <0.001 | yes |
| DeepSeek V3.2 | Qwen3.5-9B | 15.5 | [13.0, 17.9] | <0.001 | yes |
| GLM-5 | Qwen3.5-9B | 14.7 | [12.2, 17.3] | <0.001 | yes |
| DeepSeek V3.2 | Llama 4 Scout | 12.9 | [10.1, 15.8] | <0.001 | yes |
| Gemma 4 31B | Qwen3.5-35B-A3B | 12.8 | [10.9, 14.8] | <0.001 | yes |
| Gemma 4 31B | Llama 4 Maverick | 12.7 | [10.3, 15.1] | <0.001 | yes |
| Gemma 4 31B | Qwen3.5-122B-A10B | 11.8 | [9.6, 14.2] | <0.001 | yes |
| GLM-5 | Llama 4 Scout | 11.6 | [7.8, 15.2] | <0.001 | yes |
| Qwen3.5-397B-A17B | Qwen3.5-9B | 10.5 | [7.8, 13.1] | <0.001 | yes |
| Qwen3.5-27B | Qwen3.5-9B | 10.3 | [8.1, 12.5] | <0.001 | yes |
| Gemma 4 31B | Qwen3.5-27B | 9.3 | [7.4, 11.2] | <0.001 | yes |
| DeepSeek V3.2 | Llama 4 Maverick | 8.7 | [6.5, 11.0] | <0.001 | yes |
| Gemma 4 31B | Qwen3.5-397B-A17B | 8.6 | [6.6, 10.6] | <0.001 | yes |
| DeepSeek V3.2 | Qwen3.5-35B-A3B | 8.6 | [6.4, 10.8] | <0.001 | yes |
| GLM-5 | Qwen3.5-35B-A3B | 8.1 | [6.0, 10.3] | <0.001 | yes |
| Qwen3.5-397B-A17B | Llama 4 Scout | 7.9 | [4.5, 11.3] | <0.001 | yes |
| GLM-5 | Llama 4 Maverick | 7.9 | [5.4, 10.4] | <0.001 | yes |
| Qwen3.5-122B-A10B | Qwen3.5-9B | 7.8 | [5.1, 10.4] | <0.001 | yes |
| DeepSeek V3.2 | Qwen3.5-122B-A10B | 7.4 | [4.7, 10.0] | <0.001 | yes |
| Llama 4 Maverick | Qwen3.5-9B | 7.4 | [4.6, 10.1] | <0.001 | yes |
| Qwen3.5-27B | Llama 4 Scout | 7.3 | [3.9, 10.8] | <0.001 | yes |
| GLM-5 | Qwen3.5-122B-A10B | 7.0 | [4.6, 9.2] | <0.001 | yes |
| Qwen3.5-35B-A3B | Qwen3.5-9B | 6.6 | [4.3, 9.1] | <0.001 | yes |
| DeepSeek V3.2 | Qwen3.5-27B | 5.0 | [3.0, 6.9] | <0.001 | yes |
| Gemma 4 31B | GLM-5 | 4.9 | [3.0, 6.8] | <0.001 | yes |
| GLM-5 | Qwen3.5-397B-A17B | 4.5 | [2.4, 6.7] | <0.001 | yes |
| GLM-5 | Qwen3.5-27B | 4.4 | [2.4, 6.5] | <0.001 | yes |
| Gemma 4 31B | DeepSeek V3.2 | 4.0 | [2.2, 5.6] | <0.001 | yes |
| Qwen3.5-397B-A17B | Qwen3.5-35B-A3B | 3.7 | [1.4, 6.0] | <0.001 | yes |
| DeepSeek V3.2 | Qwen3.5-397B-A17B | 4.3 | [1.9, 6.7] | 0.015 | yes |
| Qwen3.5-27B | Qwen3.5-35B-A3B | 3.2 | [1.4, 5.0] | 0.028 | yes |
| Qwen3.5-27B | Llama 4 Maverick | 3.4 | [0.9, 5.9] | 0.104 | no |
| Qwen3.5-397B-A17B | Llama 4 Maverick | 3.8 | [1.0, 6.4] | 0.108 | no |
| Qwen3.5-122B-A10B | Llama 4 Scout | 4.8 | [0.8, 8.6] | 0.198 | no |
| Llama 4 Scout | Qwen3.5-9B | 4.0 | [0.7, 7.3] | 0.198 | no |
| Qwen3.5-397B-A17B | Qwen3.5-122B-A10B | 2.7 | [0.5, 5.1] | 0.198 | no |
| Llama 4 Maverick | Llama 4 Scout | 3.5 | [0.1, 6.7] | 0.360 | no |
| Qwen3.5-27B | Qwen3.5-122B-A10B | 2.1 | [−0.1, 4.4] | 0.385 | no |
| Qwen3.5-35B-A3B | Llama 4 Scout | 3.4 | [−0.3, 6.9] | 0.408 | no |
| Qwen3.5-122B-A10B | Qwen3.5-35B-A3B | 0.9 | [−1.4, 3.2] | 1.000 | no |
| Qwen3.5-122B-A10B | Llama 4 Maverick | 0.8 | [−2.1, 3.6] | 1.000 | no |
| DeepSeek V3.2 | GLM-5 | 0.7 | [−1.5, 2.9] | 1.000 | no |
| Qwen3.5-397B-A17B | Qwen3.5-27B | 0.2 | [−2.0, 2.4] | 1.000 | no |
| Qwen3.5-35B-A3B | Llama 4 Maverick | 0.0 | [−2.5, 2.7] | 1.000 | no |

**Grounded answering, correctness** (45 pairs of models; difference in percentage points)

| Higher | Lower | Difference (points) | 95% interval | Holm p | Significant |
|---|---|---|---|---|---|
| Qwen3.5-397B-A17B | Llama 4 Scout | 12.5 | [10.8, 14.4] | <0.001 | yes |
| Qwen3.5-35B-A3B | Llama 4 Scout | 11.8 | [10.0, 13.5] | <0.001 | yes |
| Qwen3.5-27B | Llama 4 Scout | 11.8 | [9.8, 13.8] | <0.001 | yes |
| DeepSeek V3.2 | Llama 4 Scout | 11.3 | [9.5, 13.1] | <0.001 | yes |
| Qwen3.5-122B-A10B | Llama 4 Scout | 10.9 | [8.9, 12.8] | <0.001 | yes |
| GLM-5 | Llama 4 Scout | 10.7 | [8.9, 12.5] | <0.001 | yes |
| Gemma 4 31B | Llama 4 Scout | 9.1 | [7.2, 11.1] | <0.001 | yes |
| Qwen3.5-9B | Llama 4 Scout | 8.3 | [6.5, 10.0] | <0.001 | yes |
| Llama 4 Maverick | Llama 4 Scout | 7.1 | [5.3, 9.0] | <0.001 | yes |
| Qwen3.5-397B-A17B | Llama 4 Maverick | 5.9 | [4.4, 7.4] | <0.001 | yes |
| Qwen3.5-35B-A3B | Llama 4 Maverick | 4.9 | [3.7, 6.2] | <0.001 | yes |
| Qwen3.5-27B | Llama 4 Maverick | 4.7 | [3.3, 6.0] | <0.001 | yes |
| Qwen3.5-397B-A17B | Qwen3.5-9B | 4.2 | [2.7, 5.8] | <0.001 | yes |
| DeepSeek V3.2 | Llama 4 Maverick | 4.1 | [2.2, 5.7] | <0.001 | yes |
| Qwen3.5-397B-A17B | Gemma 4 31B | 4.0 | [2.9, 5.4] | <0.001 | yes |
| Qwen3.5-122B-A10B | Llama 4 Maverick | 3.9 | [2.4, 5.3] | <0.001 | yes |
| GLM-5 | Llama 4 Maverick | 3.3 | [2.0, 4.6] | <0.001 | yes |
| Qwen3.5-35B-A3B | Qwen3.5-9B | 3.2 | [1.9, 4.6] | <0.001 | yes |
| Qwen3.5-27B | Qwen3.5-9B | 3.1 | [1.7, 4.5] | <0.001 | yes |
| Qwen3.5-35B-A3B | Gemma 4 31B | 2.9 | [1.7, 4.2] | <0.001 | yes |
| Qwen3.5-397B-A17B | GLM-5 | 2.8 | [1.6, 4.2] | <0.001 | yes |
| Qwen3.5-27B | Gemma 4 31B | 2.7 | [1.6, 3.9] | <0.001 | yes |
| Qwen3.5-122B-A10B | Qwen3.5-9B | 2.5 | [1.1, 4.1] | <0.001 | yes |
| DeepSeek V3.2 | Qwen3.5-9B | 2.5 | [1.1, 4.0] | 0.044 | yes |
| Qwen3.5-35B-A3B | GLM-5 | 1.7 | [0.5, 2.9] | 0.084 | no |
| Qwen3.5-397B-A17B | Qwen3.5-122B-A10B | 2.0 | [0.7, 3.3] | 0.100 | no |
| Qwen3.5-397B-A17B | Qwen3.5-27B | 1.5 | [0.4, 2.5] | 0.100 | no |
| Qwen3.5-122B-A10B | Gemma 4 31B | 1.9 | [0.5, 3.4] | 0.126 | no |
| Qwen3.5-9B | Llama 4 Maverick | 1.9 | [0.3, 3.4] | 0.187 | no |
| Gemma 4 31B | Llama 4 Maverick | 1.7 | [0.4, 3.0] | 0.192 | no |
| GLM-5 | Qwen3.5-9B | 1.8 | [0.2, 3.4] | 0.375 | no |
| Qwen3.5-27B | GLM-5 | 1.3 | [0.1, 2.4] | 0.375 | no |
| GLM-5 | Gemma 4 31B | 1.4 | [0.2, 2.6] | 0.390 | no |
| DeepSeek V3.2 | Gemma 4 31B | 2.1 | [0.2, 3.8] | 0.408 | no |
| Qwen3.5-397B-A17B | DeepSeek V3.2 | 1.2 | [0.1, 2.4] | 0.408 | no |
| Qwen3.5-397B-A17B | Qwen3.5-35B-A3B | 0.9 | [−0.1, 2.0] | 0.970 | no |
| DeepSeek V3.2 | GLM-5 | 0.9 | [−0.9, 2.5] | 1.000 | no |
| Qwen3.5-35B-A3B | Qwen3.5-122B-A10B | 0.9 | [−0.3, 2.1] | 1.000 | no |
| Qwen3.5-35B-A3B | DeepSeek V3.2 | 0.8 | [−0.6, 2.4] | 1.000 | no |
| Qwen3.5-122B-A10B | GLM-5 | 0.6 | [−0.7, 2.0] | 1.000 | no |
| Qwen3.5-27B | Qwen3.5-122B-A10B | 0.6 | [−0.6, 1.7] | 1.000 | no |
| Gemma 4 31B | Qwen3.5-9B | 0.5 | [−1.0, 2.1] | 1.000 | no |
| Qwen3.5-35B-A3B | Qwen3.5-27B | 0.4 | [−0.6, 1.4] | 1.000 | no |
| Qwen3.5-27B | DeepSeek V3.2 | 0.4 | [−1.0, 2.1] | 1.000 | no |
| DeepSeek V3.2 | Qwen3.5-122B-A10B | 0.1 | [−1.8, 1.7] | 1.000 | no |
