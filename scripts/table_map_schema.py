"""
table_map_schema.py
-------------------
Stage 2.3 of the v2 pipeline: the TABLE MAP - exactly what the LLM returns for
each table. Shared by the runner (2.4), the validator, the pilot checks (2.6)
and the item builder (3.1).

The map is the LLM's ONLY contribution. It says WHERE to read, never WHAT the
answer is: every field is an index, a label from a fixed list, a short name, or
a question. Gold answers are copied verbatim from cells by code afterwards.

Direct ancestor: the v1 structure map (old/scripts/llm_restructure_tables.py,
RESPONSE_SCHEMA). Carried over: n_header_rows, study_col_index, and per column
role / field / question. New in v2:

  table level   accept, reject_reason (v1 decided acceptance in a separate,
                header-only stage)
  row level     one entry per row below the header - kind, study_key, arm,
                outcome, ref, ref_matches_label (v1 had only a list of
                non-study rows)
  column level  content (characteristic / result) and answerability

The table type (characteristics / results / mixed) is NOT asked of the LLM; it
is derived from the column `content` tags by `table_type()`, so the two can
never disagree.

FIELD_DEFINITIONS below is the single written definition of every field. The
prompt in 2.4 quotes it verbatim, so the rules and the prompt cannot drift.
"""

ROW_KINDS = ['study', 'banner', 'subtotal', 'note']
COLUMN_ROLES = ['extraction', 'identifier', 'quality', 'other']
CONTENT_TYPES = ['characteristic', 'result', 'none']
ANSWERABILITY_CATEGORIES = ['reported', 'reviewer_judgement', 'review_calculated',
                            'bibliographic', 'not_applicable']
REF_MATCH = ['yes', 'no', 'unclear']

FIELD_DEFINITIONS = """\
TABLE LEVEL

accept - true only if the table has one row per included primary study (or per \
study arm or subgroup), and its columns hold data the review authors extracted \
from each study. Both study CHARACTERISTICS tables and per-study RESULTS tables \
qualify. false for: risk-of-bias / quality-appraisal / certainty tables \
(Newcastle-Ottawa, RoB, GRADE, QUADAS, AMSTAR, PEDro, Jadad), search strategies, \
lists of excluded studies, PRISMA counts, eligibility criteria, and summary \
tables whose rows are categories or totals rather than individual studies \
(e.g. "RCT | 12 | 40%", or cells that each list many citations), tables \
whose studies run across the COLUMNS (one column per study) instead of down \
the rows, and tables whose rows are themselves REVIEWS (systematic reviews or \
meta-analyses, e.g. with a "number of primary studies" column) rather than \
primary studies.

reject_reason - when accept is false, a few words saying why ("risk of bias \
table", "summary counts by category"). Empty string when accept is true.

n_header_rows - how many rows from the top are header rows. Judge from the \
content, not from markup: if row 0 holds spanning group names ("Intervention \
group") and row 1 holds the names under them ("Baseline", "Week 12"), it is 2. \
Getting this wrong turns a header row into a study. A full-width section \
heading directly below the header (e.g. "Observational studies") is NOT a \
header row: count it as a "banner" row.

study_col_index - the column that identifies each study (author, year, \
citation). You are given the program's proposal; keep it unless the table \
clearly shows another column identifies the studies.

When accept is false, return empty lists for rows and columns.

ROW LEVEL - one entry for EVERY row from n_header_rows to the last row, in order.

kind - "study": the row holds data about an included study - the whole study, \
or one arm, subgroup, outcome or time point of it. "banner": a full-width \
section heading, not a study. "subtotal": a total, pooled or summary row. \
"note": a footnote or other non-study row.

study_key - an integer shared by all rows of the same study, starting at 0 and \
counting up in table order. All rows of one study share one key - its arm rows, \
its outcome rows, and continuation rows (the same study's data carried on to \
the next row). Use -1 for banner, subtotal and note rows.

arm - when a study has several rows for different arms, groups or subgroups of \
participants, a short name for this row's group taken from the table \
("athletes", "control", "high dose"). Empty string otherwise.

outcome - when a study has several rows for different outcomes, measures or \
time points, a short name for this row's outcome or time point taken from the \
table ("ICU length of stay", "28-day mortality", "week 12"). Empty string \
otherwise. A row can have both an arm and an outcome.

Rows of one study with the SAME arm and outcome are treated as one row: their \
cells are joined. So give continuation rows the same labels as the row they \
continue, and give rows that differ by group or outcome labels that tell them \
apart - two rows of one study must never hold different data yet share both \
labels.

ref - the citation id (e.g. "R3") of the study's OWN paper, chosen from the \
citation ids shown in that study's rows in the study column or in another \
"identifier" column (e.g. a separate "Ref." or "Citation" column). Give the \
same ref on every row of the study, even when only its first row shows the \
citation. If several papers are cited for the study, pick its primary results \
paper; if you cannot tell which, or none is cited, use an empty string. Never \
pick a citation from a content column (results, comments, comparator, and so \
on). Empty string for non-study rows.

ref_matches_label - does the bibliography entry for ref match the row's study \
label (first author and year)? "yes", "no" (e.g. the label says Smith 2019 but \
the entry is Jones 2021), or "unclear" (the label is only a number such as \
"[18]", or ref is empty).

COLUMN LEVEL - one entry for EVERY column index, in ascending order.

role - decide from what the column CONTAINS. "extraction": the cells hold data \
taken from each study - the default and the large majority of columns (country, \
design, sample size, population, age, sex, intervention, comparator, dose, \
outcomes, instruments, results, effect sizes, follow-up, attrition, funding). \
"identifier": study, author, year, reference, citation, number, id. "quality": \
risk-of-bias, quality or certainty ratings. "other": RARE - blank spacer \
columns and free-text comment columns only. A column is not "other" because its \
content is long or messy.

field - a SHORT name that distinguishes this column from every other column. \
When columns share a header, name the group, arm, time point or measure they \
belong to: "Number of subjects (control)" and "Number of subjects (T2DM)", \
never two identical names.

question - for "extraction" columns only: one clear question about a single \
study, answerable only from THIS column ("How many subjects were in the \
control group?", not "How many subjects?"). Never name the study, its authors \
or its year - the paper is given alongside the question. A column holding two \
facts gets one question asking for both. Empty string for every other role.

content - for "extraction" columns: "characteristic" if the cells describe the \
study itself (country, design, setting, sample size, participants, age, sex, \
intervention, comparator, dose, instruments, WHICH outcomes were measured, \
follow-up length); "result" if they report what the study found (effect \
estimates, confidence intervals, p-values, means or counts at follow-up, event \
rates, main findings, conclusions). "none" for every other role.

answerability_category - WITHOUT seeing any paper, where would this column's \
information come from? "reported": normally stated in the primary study's own \
full text. "reviewer_judgement": a judgement the review authors made (ratings, \
classifications they assigned). "review_calculated": a value the review authors \
computed or converted (effect sizes derived from raw counts, pooled or \
standardised values, unit conversions). "bibliographic": publication metadata \
(journal, publication year, funding source listed in the paper's front matter \
counts as "reported"). "not_applicable" for non-extraction columns.

answerability_score - 1 to 5: how likely it is that the answer to this \
column's question can be found in the primary study's full text. 5 = almost \
always stated explicitly; 3 = often stated, sometimes only derivable; \
1 = almost never in the paper. The score must agree with the category: \
"reported" columns normally score 3 to 5; "reviewer_judgement" and \
"review_calculated" columns normally score 1 or 2, because the value itself is \
the review's, even when the paper holds what it was based on. 0 for \
non-extraction columns.
"""

RESPONSE_SCHEMA = {
    'type': 'object',
    'properties': {
        'accept': {'type': 'boolean'},
        'reject_reason': {'type': 'string'},
        'n_header_rows': {'type': 'integer'},
        'study_col_index': {'type': 'integer'},
        'rows': {
            'type': 'array',
            'items': {
                'type': 'object',
                'properties': {
                    'row_index': {'type': 'integer'},
                    'kind': {'type': 'string', 'enum': ROW_KINDS},
                    'study_key': {'type': 'integer'},
                    'arm': {'type': 'string'},
                    'outcome': {'type': 'string'},
                    'ref': {'type': 'string'},
                    'ref_matches_label': {'type': 'string', 'enum': REF_MATCH},
                },
                'required': ['row_index', 'kind', 'study_key', 'arm', 'outcome',
                             'ref', 'ref_matches_label'],
                'additionalProperties': False,
            },
        },
        'columns': {
            'type': 'array',
            'items': {
                'type': 'object',
                'properties': {
                    'col_index': {'type': 'integer'},
                    'role': {'type': 'string', 'enum': COLUMN_ROLES},
                    'field': {'type': 'string'},
                    'question': {'type': 'string'},
                    'content': {'type': 'string', 'enum': CONTENT_TYPES},
                    'answerability_category': {'type': 'string',
                                               'enum': ANSWERABILITY_CATEGORIES},
                    'answerability_score': {'type': 'integer'},
                },
                'required': ['col_index', 'role', 'field', 'question', 'content',
                             'answerability_category', 'answerability_score'],
                'additionalProperties': False,
            },
        },
    },
    'required': ['accept', 'reject_reason', 'n_header_rows', 'study_col_index',
                 'rows', 'columns'],
    'additionalProperties': False,
}


def table_type(table_map):
    """Derive the table type from the column content tags.

    'characteristics', 'results' or 'mixed' for accepted tables, 'rejected'
    otherwise - computed, never asked of the LLM, so it cannot contradict the
    per-column tags.
    """
    if not table_map['accept']:
        return 'rejected'
    kinds = {c['content'] for c in table_map['columns'] if c['role'] == 'extraction'}
    if kinds == {'characteristic'}:
        return 'characteristics'
    if kinds == {'result'}:
        return 'results'
    return 'mixed'


def qualified_question(question, arm, outcome):
    """The question as asked for one item: the column's question plus the row's labels.

    "What effect estimate was reported?" + outcome "ICU length of stay" ->
    "What effect estimate was reported? (outcome: ICU length of stay)".
    Code adds the labels, so each column's question is written once.
    """
    labels = []
    if arm:
        labels.append(f'group: {arm}')
    if outcome:
        labels.append(f'outcome: {outcome}')
    return f"{question} ({'; '.join(labels)})" if labels else question
