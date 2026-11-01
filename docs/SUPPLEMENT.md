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

- Second judge: Claude Sonnet 5.5 re-judged the answers of four models on
  100 test-2026 papers.
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
