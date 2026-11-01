# Datasheet for SR-QA v2.0.1

Following *Datasheets for Datasets* (Gebru et al., 2021). Figures are from the frozen release
(`MANIFEST.json`, `quality/`).

## Motivation

**Purpose.** SR-QA tests whether a language model can answer the data-extraction questions of a
systematic review from the full text of an included study, and whether it stays grounded in that
text: answering from what the paper says, and abstaining when it says nothing. Data extraction is
one of the most labour-intensive stages of a systematic review and, unlike title and abstract
screening, has had little benchmark support at the full-text level.

**Source of the supervision.** Published systematic reviews already contain the gold labels: each
review's per-study extraction table records, for every included study, the values its authors
extracted. SR-QA turns those tables into question-answer items and pairs each item with the full
text of the study it was extracted from.

**Creators and funding.** Opeoluwa Akinseloyin and Vasile Palade (Centre for Computational
Science and Mathematical Modelling, Coventry University, UK) and Xiaorui Jiang (School of
Information, Journalism and Communications, The University of Sheffield, UK). This material is
based upon work supported by the Google Cloud Research Credits program with the award
165bd6d1-26b9-4f86-8cae-771673a172e7 (an in-kind resource, not a grant).

## Composition

**Instances.** Each item is one question about one included study, with:

| field | meaning |
|---|---|
| `question` | the question, written from the table column; when one study has several rows that differ by group or outcome, the question names it, e.g. "(group: athletes)" or "(outcome: ICU length of stay)" |
| `gold` | the review's own extraction, copied verbatim from the table cell(s); several cells are joined with " \| " |
| `source_pmcid` | the included study's full text (CC BY, supplied with the release) |
| `review_pmcid` | the review the item comes from; its title, objectives and eligibility criteria are in `review_context.jsonl` |
| `split` | train-2025, train-2026, dev-2026, test-2026, test-2025 (see Splits) |
| `content` | `characteristic` (describes the study: design, setting, sample, intervention, measures) or `result` (what the study found) |
| `answerability_category`, `answerability_score` | where the information would come from, judged per column without seeing any paper; all released items are `reported` with score 3 to 5 |
| `gold_is_absence` | the review records the value as not reported ("NR", "not reported", a dash, "/", ...): the expected behaviour is to abstain |
| `arm`, `outcome` | group and outcome labels, where needed to tell a study's rows apart |
| `table_no`, `rows`, `col_index`, `study_key`, `citation_rid` | where in the review's table the item comes from, and the citation that links it to the study |

**Size.**

| | count |
|---|---|
| items | 166,308 |
| reviews | 3,037 |
| source papers (full texts) | 15,944 |
| review tables | 4,921 |
| items whose gold records "not reported" | 11,013 (6.6%) |
| result items | 31,832 (19.1%) |
| items with a group or outcome label | 30,915 (18.6%) |

Tables are characteristics tables (69,733 items), results tables (3,421) or mixed (93,154).

**Splits.** Splits are cut on each review's **online** publication date, the first day a web crawl
could have seen it:

| split | items | role |
|---|---|---|
| train-2026 | 71,255 | training |
| dev-2026 | 10,333 | model selection |
| **test-2026** | **25,561** | **headline test set: 539 reviews online 2 January to 28 June 2026, after the training cutoff of models released before 2026** |
| train-2025 | 47,719 | training |
| test-2025 | 11,440 | secondary contrast; 97.7% of 2025 reviews went online in October to December 2025, so this is a bounded contamination check, not a pre-cutoff arm |

Reviews that share a source paper are kept in the same split, and groups spanning both years go to
training, so **no source paper appears in an evaluation split and any other split**. (213 papers
appear in both training splits, train-2025 and train-2026, by design.)

**Retrieval dataset.** For evidence retrieval, each source paper is split, in document order, into
passages (sentences, table data rows, table cells holding a number, captions and footnotes). A
passage is relevant to a question if it contains a fragment of the gold: grade 2 for a whole-phrase
match, grade 1 when every distinctive number of the fragment occurs. Questions are left out when the
gold is "not reported", yes/no, or has no letter or digit, when no passage matches, or when more than
ten passages match. This leaves 48,219 questions with labels (test-2026: 7,500), released for every
split with the grade and kind of match of each label, in JSONL and TREC format, with the reason each
other question was left out. A blind hand check of 195 test-2026 pairs found phrase matches always
correct and number matches correct in 40 of 47; the best baseline's top-ranked unlabelled passage
stated the answer in 19 of 86 cases, so the labels are precise but incomplete, and scores computed
on them are lower bounds.

**Is it a sample?** The reviews are the CC BY primary systematic reviews in SCIE Q1 journals
indexed in PubMed Central, with issue dates January 2025 to June 2026 (5,434 reviews). Items
exist only where a review has a per-study table, its rows can be linked to a paper, and that paper
is CC BY with full text in PubMed Central (see Collection). The release publishes the list of
reviews (PMCIDs), not journal quartile values, which are proprietary (JCR).

**Known errors and noise.** The gold is what the review's authors extracted, and reviews contain
errors of their own (v1 found recorded values that contradict the source). A hand audit of 100
evaluation items found the question clear in 100, the gold correct in 99 and the right study
linked in 100; the one error revealed a construction defect that was fixed before freezing
(`quality/audit_audit.json`). An automatic check finds the gold in the source text for 78.5% of
items (exact, every number, or 80% of content words); this is a lower bound, since derived answers
count as not found (`quality/answerability_check.csv`).

**Confidential or personal data.** None beyond what the published, openly licensed articles
contain. The source texts are published research articles.

## Collection process

1. **Reviews.** CC BY systematic reviews with JATS XML from PubMed Central; umbrella reviews,
   corrections and abstract-only records removed (5,434 primary reviews).
2. **Tables.** Every table in every review (15,596) parsed into a grid, merged cells kept as one
   cell. A table is sent for mapping if any cell below its header cites a reference that could be a
   CC BY paper (7,359 tables).
3. **Table maps.** Claude Sonnet 5 (Anthropic) read each whole table and returned a structure map:
   whether it is a per-study table, its header rows, its study column, per row the study, group,
   outcome and citation, and per column a field name, a question, characteristic or result, and
   answerability. It never writes answers. Maps were checked automatically against their tables;
   7,299 were usable (99.2%). On 60 hand-checked tables, items from the maps reached 93.8% recall and
   96.7% precision against the hand-corrected maps; re-running the model agreed on 97.5% of
   acceptance decisions and 97.6% of answerability categories.
4. **Items.** Built from the maps by code: answers copied verbatim from cells; a study's
   name cell never counts as an answer; an item is dropped when its label would reveal its answer.
5. **Linking.** Each item links to its study only through the citation in the study's own study or
   identifier column, resolved to a PMCID (NCBI ID Converter), and the full text downloaded from
   PubMed Central.
6. **Filters**, in order (`quality/funnel.csv`):

| step | items | papers | reviews |
|---|---|---|---|
| candidate items | 807,431 | 29,188 | 3,305 |
| the paper is in PubMed Central | 330,733 | 29,188 | 3,277 |
| the paper is CC BY | 181,287 | 16,007 | 3,046 |
| the file holds the full text | 181,218 | 16,000 | 3,046 |
| answerability "reported", score 3 or more | 174,569 | 15,947 | 3,037 |
| gold not blank | 168,400 | 15,946 | 3,037 |
| gold is not only a footnote marker | 168,235 | 15,944 | 3,037 |
| no exact duplicate within a review | **166,308** | **15,944** | **3,037** |

(The papers column counts papers with a PMCID.)

**Who did the collection.** Automated pipeline (`scripts/`); hand checks by the authors.
**Timeframe.** Reviews issued January 2025 to June 2026; data collected September and October 2026.
**Ethics review.** No institutional review was sought: the data are published, openly licensed
articles and no human-subjects data were collected.

## Preprocessing, cleaning and labelling

Licences are read from each article's own `<license>` element; when the wording is stricter than
the licence link, the stricter reading wins. Study design for the bias analysis comes from PubMed
publication types and design MeSH headings, then design words in the title, then Gemini 3.7 Flash
labels from the abstract (agreement with PubMed's labels on a 1,000-paper check sample: 88.2% on
coarse groups, 70.0% on fine categories). Publishers are from OpenAlex, mapped to parent companies.
The table maps (`table_maps.jsonl.gz`) are released, so every item can be rebuilt exactly.

## Uses

**Intended.** Benchmarking grounded question answering and abstention over long scientific
documents; evaluating LLM support for systematic-review data extraction; evidence-passage
retrieval; training and evaluating models for extraction (train and dev splits).

**Not intended.** As a substitute for human data extraction in a systematic review; as a
representative sample of closed-access literature.

**Composition effects to keep in mind** (`quality/bias_table.csv`, total variation distance from all
candidate items to the final dataset): restricting to CC BY full texts barely changes review field
(0.06), study design (0.14, mostly the unlabelled candidates) or question type (0.01), but strongly
changes the publisher mix (0.54: Springer Nature 32%, MDPI 20%, Frontiers 13%; Elsevier falls from
20% to 3%) and the publication years (0.30: 75% of source papers from 2020 on). Report results by
publisher and year where they matter.

**Contamination.** Measure it with models used zero-shot. The training splits hold 81% of 2025
items but 66% of 2026 items, which would confound a contrast run with a model fine-tuned on them.

## Distribution

**Licence.** CC BY 4.0. Every review and every source paper is CC BY; attribution metadata for
each is included.
**Where.** Data and code on Zenodo (https://doi.org/10.5281/zenodo.23432598); code also on GitHub
(https://github.com/Opeoluwa-Akinseloyin/SR-QA, MIT licence).
**Fees or restrictions.** None.

## Maintenance

**Maintainer and contact.** Xiaorui Jiang (xiaorui.jiang@sheffield.ac.uk), The University of
Sheffield.
**Versioning.** Each version is fixed and checksummed (`MANIFEST.json`). Later versions are released
under the same Zenodo concept DOI, and earlier versions stay available unchanged.

**Changes in v2.0.1** (14 October 2026, before first public release). A check of golds with no
letter or digit found three kinds of defect, now fixed in `11_filter_items.py`: 59 golds of "--" and
310 of "/" or "\\" are placeholders for "not reported" and are now flagged as such, like a dash;
46 golds that were only an invisible character (a zero-width space) count as blank and are dropped;
125 golds that were only footnote markers ("*", "∗", "#") are dropped. 171 items were removed and
369 re-flagged; no other item changed, and every review keeps its v2.0 split. Golds in review
notation (ticks, arrows, "+", "−") are real answers and are kept.

**Errata.** Report errors through GitHub issues (https://github.com/Opeoluwa-Akinseloyin/SR-QA/issues).
Corrections are fixed in the pipeline and released as a new version with a change note.
