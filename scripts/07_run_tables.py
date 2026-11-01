"""
07_run_tables.py
----------------
Stage 2.7 (preparation) of the v2 pipeline: decide which tables the full
mapping run sends to the LLM, and record WHY every other table is left out.

Every table gets exactly one decision. A table is left out only when it
CANNOT produce a single item under v2's linking rules, whatever the LLM
decides - so these are stated filters, not guesses:

  no_citation_links        no cell below the header cites a bibliography entry,
                           so no row can be linked to a paper
  no_possible_ccby_source  every cited paper is either (a) on disk but not
                           CC-BY, (b) confirmed by NCBI as not in PubMed Central,
                           or (c) listed with no PMCID, PMID or DOI - so no row
                           can be linked to a CC-BY paper we hold. (c) means
                           "cannot be linked by our method", not "proven
                           unavailable"; the paper should say so.
  mapped                   everything else - sent to the LLM. Includes tables
                           citing papers not yet downloaded or not yet looked
                           up, which Phase 3 resolves.

Inputs:
  data/v2/tables.jsonl.gz, data/v2/study_columns.csv  (stages 2.1, 2.2)
  data/v2/source_licences.csv                         (stage 1)
  data/input/resolved_ids_v1.csv   v1's NCBI ID-converter results (PMID/DOI ->
                                   PMCID or "not_in_pmc"), frozen as an input
Outputs:
  data/v2/run_tables.csv     review_pmcid, table_no, decision, n_refs and the
                             count of cited refs per status
  data/v2/mapped_tables.csv  the `mapped` tables only - the --tables list for
                             the full run of 04_map_tables.py

Usage:
  python scripts/07_run_tables.py
"""
import collections
import csv
import gzip
import json
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TABLES_PATH = os.path.join(BASE, 'data', 'v2', 'tables.jsonl.gz')
STUDY_COLUMNS_CSV = os.path.join(BASE, 'data', 'v2', 'study_columns.csv')
LICENCES_CSV = os.path.join(BASE, 'data', 'v2', 'source_licences.csv')
RESOLVED_CSV = os.path.join(BASE, 'data', 'input', 'resolved_ids_v1.csv')
RESOLVED_V2_CSV = os.path.join(BASE, 'data', 'v2', 'resolved_ids_v2.csv')
OUT_CSV = os.path.join(BASE, 'data', 'v2', 'run_tables.csv')
MAPPED_CSV = os.path.join(BASE, 'data', 'v2', 'mapped_tables.csv')

STATUSES = ['ccby_on_disk', 'not_downloaded', 'unresolved', 'not_ccby', 'not_in_pmc', 'no_id']
DEAD = {'not_ccby', 'not_in_pmc', 'no_id'}


def norm_pmcid(value):
    value = (value or '').strip().upper()
    if not value:
        return ''
    return value if value.startswith('PMC') else 'PMC' + value


def load_inputs():
    with open(LICENCES_CSV, encoding='utf-8') as f:
        licences = {row['pmcid']: row['licence'] for row in csv.DictReader(f)}
    # v1's NCBI lookups, then v2's own (09_fetch_sources.py), which take
    # precedence. A v2 'lookup_failed' is no answer at all, so it is skipped.
    resolved = {}
    for path in (RESOLVED_CSV, RESOLVED_V2_CSV):
        if not os.path.exists(path):
            continue
        with open(path, encoding='utf-8') as f:
            for row in csv.DictReader(f):
                if row['status'] in ('ok', 'not_in_pmc'):
                    resolved[(row['key_type'], row['query_id'].strip().lower())] = row['resolved_pmcid']
    with open(STUDY_COLUMNS_CSV, encoding='utf-8') as f:
        methods = {(row['review_pmcid'], int(row['table_no'])): row['method']
                   for row in csv.DictReader(f)}
    return licences, resolved, methods


def ref_status(ref, licences, resolved):
    """Where one cited reference stands: can it become a CC-BY paper we hold?"""
    pmcid = norm_pmcid(ref['pmcid'])
    if not pmcid:
        for key_type in ('pmid', 'doi'):
            value = (ref[key_type] or '').strip().lower()
            if value and (key_type, value) in resolved:
                pmcid = norm_pmcid(resolved[(key_type, value)])
                if not pmcid:
                    return 'not_in_pmc'
                break
        else:
            return 'unresolved' if (ref['pmid'] or ref['doi']) else 'no_id'
    if pmcid not in licences:
        return 'not_downloaded'
    return 'ccby_on_disk' if licences[pmcid] == 'CC-BY' else 'not_ccby'


def decide(table, method, licences, resolved):
    """(decision, per-status counts of the table's cited refs)."""
    counts = collections.Counter(ref_status(ref, licences, resolved)
                                 for ref in table['refs'].values())
    if method != 'links':
        return 'no_citation_links', counts
    if counts and set(counts) <= DEAD:
        return 'no_possible_ccby_source', counts
    return 'mapped', counts


def main():
    licences, resolved, methods = load_inputs()
    rows = []
    with gzip.open(TABLES_PATH, 'rt', encoding='utf-8') as f:
        for line in f:
            table = json.loads(line)
            key = (table['review_pmcid'], table['table_no'])
            decision, counts = decide(table, methods[key], licences, resolved)
            rows.append({'review_pmcid': key[0], 'table_no': key[1], 'decision': decision,
                         'n_refs': sum(counts.values()),
                         **{status: counts.get(status, 0) for status in STATUSES}})

    with open(OUT_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    with open(MAPPED_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['review_pmcid', 'table_no'])
        writer.writerows([row['review_pmcid'], row['table_no']]
                         for row in rows if row['decision'] == 'mapped')

    decisions = collections.Counter(row['decision'] for row in rows)
    print(f'tables: {len(rows):,}')
    for decision in ('mapped', 'no_citation_links', 'no_possible_ccby_source'):
        print(f'  {decision:<25} {decisions[decision]:>6,}')
    mapped = [row for row in rows if row['decision'] == 'mapped']
    print('cited refs in mapped tables, by status:')
    for status in STATUSES:
        print(f'  {status:<16} {sum(row[status] for row in mapped):>7,}')
    print(f'-> {OUT_CSV}')


if __name__ == '__main__':
    main()
