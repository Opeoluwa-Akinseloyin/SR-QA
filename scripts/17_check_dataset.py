"""
17_check_dataset.py
-------------------
Stage 3.6 of the v2 pipeline: re-verify the final dataset independently,
before it is frozen. Every check recomputes from the source files rather than
trusting a flag an earlier stage set.

HARD checks - any failure exits non-zero, so the dataset cannot be frozen on a
broken check:
  ids_unique             every item_id appears once
  gold_verbatim          gold re-parses as whole cell texts of its column
  rebuild_identical      rebuilding from the saved maps and tables gives the
                         same question and gold (nothing drifted)
  link_in_study_columns  the item's citation is in the study column or an
                         identifier column of ANY row of its study (the
                         validator's rule) - never in a content column
  source_valid           the linked paper is on disk, CC-BY, with a body
  filters_hold           answerability reported and >= 3, gold not blank, no
                         duplicate within a review, no label giving the answer
  splits_consistent      one arm per review; 2026 arms hold only reviews online
                         in 2026 or later, 2025 arms only reviews online before
                         2026 (13_make_splits.py's rule); no paper shared between
                         an evaluation arm and any other arm
  counts_agree           funnel's final row = filtered items = split items
  no_heading_rows        no item draws on a row that is one merged cell spanning
                         the table (a study's name heading; 4.1 audit finding)
  no_label_cell_answers  no gold fragment comes from the row's own study-label
                         cell merged into another column (4.1 audit finding)

SOFT checks - reported with examples, not fatal:
  gold_is_study_label    the gold equals the study label in its row (legitimate
                         only for a column that really asks for the study name)
  study_name_in_question a surname written like a citation in the study cell
                         ("Smith et al.", "Smith and Jones", "Smith (2019)")
                         appears in the question

Output: data/v2/checks_report.json, and a printed summary.

Usage:
  python scripts/17_check_dataset.py
"""
import collections
import csv
import gzip
import importlib.util
import json
import os
import re
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V2 = os.path.join(BASE, 'data', 'v2')
SOURCE_DIR = os.path.join(BASE, 'SOURCE_XML')
REPORT_PATH = os.path.join(V2, 'checks_report.json')
EVAL_ARMS = {'dev-2026', 'test-2026', 'test-2025'}
MIN_ANSWERABILITY = 3
# A surname written the way citations write it: "Smith et al.", "Smith and Jones",
# "Smith & Lee", "Smith (2019)", "Smith 2019" - not any capitalised word.
CITED_SURNAME = re.compile(r"\b([A-Z][A-Za-zÀ-ſ'-]{2,})\s*(?:et\s+al|and\s+[A-Z]|&|,?\s*\(?(?:19|20)\d{2})")


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


build_items = load('build_items', os.path.join(BASE, 'scripts', '10_build_items.py'))
filter_items = load('filter_items', os.path.join(BASE, 'scripts', '11_filter_items.py'))


def read_jsonl(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(line) for line in f]


def main():
    items = read_jsonl(os.path.join(V2, 'items_split.jsonl'))
    filtered_count = sum(1 for _ in open(os.path.join(V2, 'items_filtered.jsonl'), encoding='utf-8'))
    with open(os.path.join(V2, 'funnel.csv'), encoding='utf-8') as f:
        funnel_final = int(list(csv.DictReader(f))[-1]['items'])
    maps = {(m['review_pmcid'], m['table_no']): m for m in read_jsonl(os.path.join(V2, 'table_maps.jsonl'))}
    with open(os.path.join(V2, 'source_licences.csv'), encoding='utf-8') as f:
        licences = {r['pmcid']: r for r in csv.DictReader(f)}
    with open(os.path.join(V2, 'review_splits.csv'), encoding='utf-8') as f:
        splits = {r['review_pmcid']: r for r in csv.DictReader(f)}

    needed = {(i['review_pmcid'], i['table_no']) for i in items}
    tables = {}
    with gzip.open(os.path.join(V2, 'tables.jsonl.gz'), 'rt', encoding='utf-8') as f:
        for line in f:
            table = json.loads(line)
            key = (table['review_pmcid'], table['table_no'])
            if key in needed:
                tables[key] = table

    failures = collections.defaultdict(list)
    soft = collections.defaultdict(list)

    # ids_unique
    for item_id, n in collections.Counter(i['item_id'] for i in items).items():
        if n > 1:
            failures['ids_unique'].append(item_id)

    # rebuild once per table, for rebuild_identical
    class Linker:                                   # the stored link is checked separately
        def link(self, ref):
            return '', ''
    rebuilt = {}
    for key in needed:
        # the same map preparation as 10_build_items.main (literal \uXXXX decoded)
        record = {**maps[key], 'map': build_items.unescape_map(maps[key]['map'])}
        built, _ = build_items.build_table_items(tables[key], record, Linker())
        rebuilt.update({b['item_id']: b for b in built})

    seen_dup = set()
    has_body_cache = {}
    headings = {key: build_items.heading_rows(tables[key]) for key in needed}
    for item in items:
        key = (item['review_pmcid'], item['table_no'])
        table, table_map = tables[key], maps[key]['map']
        c = item['col_index']

        if not build_items.verify_gold(item['gold'], table, c):
            failures['gold_verbatim'].append(item['item_id'])
        if set(item['rows']) & headings[key]:
            failures['no_heading_rows'].append(item['item_id'])
        study_col = table_map['study_col_index']
        if c != study_col and any(
                table['grid'][r][c] is not None
                and table['grid'][r][c] == table['grid'][r][study_col]
                and table['cells'][table['grid'][r][c]]['text'] in item['gold'].split(' | ')
                for r in item['rows']):
            failures['no_label_cell_answers'].append(item['item_id'])
        label_cell = table['grid'][item['rows'][0]][table_map['study_col_index']]
        if (label_cell is not None and c != table_map['study_col_index']
                and item['gold'] == table['cells'][label_cell]['text']):
            soft['gold_is_study_label'].append((item['item_id'], item['field'], item['gold']))

        again = rebuilt.get(item['item_id'])
        if again is None or again['gold'] != item['gold'] or again['question'] != item['question']:
            failures['rebuild_identical'].append(item['item_id'])

        # link_in_study_columns: citation from the study column or an identifier column
        allowed_cols = {table_map['study_col_index']} | {
            col['col_index'] for col in table_map['columns'] if col['role'] == 'identifier'}
        # Any row of the item's study, as the validator allows: a study's citation
        # usually sits only in its first row, while its other groups have their own rows.
        study_rows = [row['row_index'] for row in table_map['rows']
                      if row['kind'] == 'study' and row['study_key'] == item['study_key']]
        cited = {rid for r in study_rows for col in allowed_cols
                 if table['grid'][r][col] is not None
                 for rid in table['cells'][table['grid'][r][col]]['rids']}
        if item['ref'] not in cited:
            failures['link_in_study_columns'].append(item['item_id'])

        # source_valid
        pmcid = item['source_pmcid']
        licence = licences.get(pmcid)
        if pmcid not in has_body_cache:
            path = os.path.join(SOURCE_DIR, pmcid + '.xml')
            has_body_cache[pmcid] = os.path.exists(path) and bool(
                re.search(r'<body[\s>]', open(path, encoding='utf-8', errors='ignore').read()))
        if not (licence and licence['licence'] == 'CC-BY' and has_body_cache[pmcid]):
            failures['source_valid'].append(item['item_id'])

        # filters_hold
        dup_key = (item['review_pmcid'], pmcid, item['question'].lower(), item['gold'].lower())
        if (item['answerability_category'] != 'reported'
                or item['answerability_score'] < MIN_ANSWERABILITY
                or filter_items.is_blank(item['gold'])
                or filter_items.is_footnote_only(item['gold'])
                or item['gold_is_absence'] != filter_items.is_absence(item['gold'])
                or dup_key in seen_dup
                or build_items.leaks(item['gold'], item['arm'],
                                     item['outcome'])):
            failures['filters_hold'].append(item['item_id'])
        seen_dup.add(dup_key)

        # soft: study name in question
        study_cell = table['grid'][item['rows'][0]][table_map['study_col_index']]
        if study_cell is not None:
            names = CITED_SURNAME.findall(table['cells'][study_cell]['text'])
            if names and re.search(rf'\b{re.escape(names[0])}\b', item['question']):
                soft['study_name_in_question'].append((item['item_id'], names[0], item['question']))

    # splits_consistent
    arms_of_review = collections.defaultdict(set)
    papers_by_arm = collections.defaultdict(set)
    for item in items:
        arms_of_review[item['review_pmcid']].add(item['split'])
        papers_by_arm[item['split']].add(item['source_pmcid'])
    for review, arms in arms_of_review.items():
        arm = next(iter(arms))
        online_year = splits[review]['online'][:4]
        arm_year = '2026' if online_year >= '2026' else '2025'      # 13_make_splits.py's rule
        if len(arms) > 1 or splits[review]['split'] != arm or not arm.endswith(arm_year):
            failures['splits_consistent'].append(f'review {review}: {sorted(arms)} online {online_year}')
    arms = sorted(papers_by_arm)
    for a in arms:
        for b in arms:
            if a < b and (a in EVAL_ARMS or b in EVAL_ARMS):
                shared = papers_by_arm[a] & papers_by_arm[b]
                if shared:
                    failures['splits_consistent'].append(f'{a} & {b} share {len(shared)} papers')

    # counts_agree
    if not funnel_final == filtered_count == len(items):
        failures['counts_agree'].append(f'funnel {funnel_final}, filtered {filtered_count}, '
                                        f'split {len(items)}')

    hard = ['ids_unique', 'gold_verbatim', 'rebuild_identical', 'link_in_study_columns',
            'source_valid', 'filters_hold', 'splits_consistent', 'counts_agree',
            'no_heading_rows', 'no_label_cell_answers']
    print(f'items checked: {len(items):,}\n')
    for name in hard:
        n = len(failures[name])
        print(f'  {"PASS" if n == 0 else "FAIL"}  {name:<24} {n:>7,} failures'
              + (f'   e.g. {failures[name][:2]}' if n else ''))
    n_label = len(soft['gold_is_study_label'])
    print(f'\n  soft  gold_is_study_label     {n_label:>7,} items; examples:')
    for item_id, field, gold in soft['gold_is_study_label'][:5]:
        print(f'          [{field[:30]}] {gold[:70]}')
    n_soft = len(soft['study_name_in_question'])
    print(f'\n  soft  study_name_in_question  {n_soft:>7,} items ({n_soft / len(items):.2%}); examples:')
    for item_id, name, question in soft['study_name_in_question'][:8]:
        print(f'          [{name}] {question[:100]}')

    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        json.dump({'items': len(items),
                   'hard': {name: {'failures': len(failures[name]), 'examples': failures[name][:20]}
                            for name in hard},
                   'soft': {'gold_is_study_label': {
                       'items': n_label, 'examples': soft['gold_is_study_label'][:50]},
                       'study_name_in_question': {
                       'items': n_soft, 'examples': soft['study_name_in_question'][:50]}}},
                  f, indent=1, ensure_ascii=False)
    print(f'\n-> {REPORT_PATH}')
    if any(failures[name] for name in hard):
        sys.exit(1)


if __name__ == '__main__':
    main()
