"""
15_bias_table.py
----------------
Stage 3.5 of the v2 pipeline: the bias table the ECIR resource call asks for -
how each filter step changes the make-up of the dataset.

For each key step of the funnel (the same steps as 11_filter_items.py, via its
filter_steps()), the item-weighted share of:

  design      of the source paper: PubMed labels, then title words, then
              Gemini's label from the abstract (16_design_labels.py) where
              those leave it unknown; still-unknown is its own category
  design source  which of those gave the label (so readers can weigh it)
  design group   design in coarse groups - RCT / other trial / observational /
              case report / laboratory / other / unknown. Gemini agrees with
              PubMed on 88.2% at this level (70.0% on the fine categories, where
              cohort / case-control / cross-sectional are often confused), so
              this is the version to quote
  publisher   parent company of the source paper's publisher
  field       OpenAlex field of the REVIEW
  year        publication year of the source paper (bands)
  content     characteristic vs result question

and the total variation distance (TVD, 0 = same mix, 1 = disjoint) of each step
against the candidate items - so "restricting to CC-BY shifts the publisher
mix by X" is a number, not an impression.

Inputs:  data/v2/items_all.jsonl, data/v2/item_papers.csv,
         data/v2/source_metadata.csv, data/v2/review_topics.csv
Outputs: data/v2/bias_table.csv    step, dimension, category, items, share
         data/v2/bias_tvd.csv      step, dimension, tvd_vs_candidates

Usage:
  python scripts/15_bias_table.py
"""
import collections
import csv
import importlib.util
import json
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ITEMS_PATH = os.path.join(BASE, 'data', 'v2', 'items_all.jsonl')
ITEM_PAPERS_CSV = os.path.join(BASE, 'data', 'v2', 'item_papers.csv')
METADATA_CSV = os.path.join(BASE, 'data', 'v2', 'source_metadata.csv')
TOPICS_CSV = os.path.join(BASE, 'data', 'v2', 'review_topics.csv')
DESIGN_LABELS_CSV = os.path.join(BASE, 'data', 'v2', 'design_labels.csv')
TABLE_CSV = os.path.join(BASE, 'data', 'v2', 'bias_table.csv')
TVD_CSV = os.path.join(BASE, 'data', 'v2', 'bias_tvd.csv')

TOP = 8                       # categories shown per dimension; the rest go in REST
REST = '(rest)'               # not "other": that is a real design category
DESIGN_GROUP = {'randomised controlled trial': 'RCT', 'other clinical trial': 'other trial',
                'case-control': 'observational', 'cohort': 'observational',
                'cross-sectional': 'observational', 'observational (unspecified)': 'observational',
                'case report or series': 'case report', 'animal or laboratory': 'laboratory',
                'other': 'other design', 'unknown': 'unknown', 'no identifier': 'no identifier'}
# Funnel steps reported (labels as in 11_filter_items.filter_steps), with short names.
KEY_STEPS = [
    ('candidate items', 'all candidates'),
    ('the paper is in PubMed Central', 'in PMC'),
    ('the paper is CC-BY', 'CC-BY'),
    ('the file holds the full text', 'CC-BY full text'),
    ('no exact duplicate within a review', 'final dataset'),
]

_spec = importlib.util.spec_from_file_location(
    'filter_items', os.path.join(BASE, 'scripts', '11_filter_items.py'))
filter_items = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(filter_items)


def year_band(year):
    if not year.isdigit():
        return 'unknown'
    y = int(year)
    return '<= 2015' if y <= 2015 else '2016-2019' if y <= 2019 else '2020-2022' if y <= 2022 else '2023+'


def tvd(a, b):
    ta, tb = sum(a.values()), sum(b.values())
    return 0.5 * sum(abs(a[k] / ta - b[k] / tb) for k in set(a) | set(b)) if ta and tb else float('nan')


def main():
    with open(ITEM_PAPERS_CSV, encoding='utf-8') as f:
        paper_of = {(r['review_pmcid'], int(r['table_no']), r['ref']): r['paper']
                    for r in csv.DictReader(f)}
    with open(METADATA_CSV, encoding='utf-8') as f:
        meta = {r['paper']: r for r in csv.DictReader(f)}
    with open(TOPICS_CSV, encoding='utf-8') as f:
        field_of = {r['review_pmcid']: r['field'] or 'Unknown' for r in csv.DictReader(f)}
    with open(ITEMS_PATH, encoding='utf-8') as f:
        items = [json.loads(line) for line in f]
    gemini = {}
    if os.path.exists(DESIGN_LABELS_CSV):
        with open(DESIGN_LABELS_CSV, encoding='utf-8') as f:
            gemini = {r['pmid']: r['design'] for r in csv.DictReader(f)
                      if r['design'] != 'cannot tell'}

    def design(m):
        if not m:
            return 'no identifier', 'no identifier'
        if m['design'] != 'unknown':
            return m['design'], m['design_source']
        if m['pmid'] in gemini:
            return gemini[m['pmid']], 'gemini'
        return 'unknown', 'none'

    def describe(item):
        m = meta.get(paper_of.get((item['review_pmcid'], item['table_no'], item['ref'])), {})
        label, source = design(m)
        return {'design': label, 'design group': DESIGN_GROUP.get(label, label),
                'design source': source,
                'publisher': m.get('parent_publisher', 'no identifier'),
                'field': field_of.get(item['review_pmcid'], 'Unknown'),
                'year': year_band(m.get('year', '')) if m else 'no identifier',
                'content': item['content']}

    steps = dict(filter_items.filter_steps(items))
    dims = ['design group', 'design', 'design source', 'publisher', 'field', 'year', 'content']
    dist = {}
    for label, short in KEY_STEPS:
        counts = {d: collections.Counter() for d in dims}
        for item in steps[label]:
            for d, value in describe(item).items():
                counts[d][value] += 1
        dist[short] = counts

    shown = {d: [k for k, _ in dist['final dataset'][d].most_common(TOP)] for d in dims}
    table_rows, tvd_rows = [], []
    for _, short in KEY_STEPS:
        for d in dims:
            counts = dist[short][d]
            total = sum(counts.values())
            for category in shown[d]:
                table_rows.append([short, d, category, counts[category], counts[category] / total])
            rest = total - sum(counts[c] for c in shown[d])
            table_rows.append([short, d, REST, rest, rest / total if total else 0])
            tvd_rows.append([short, d, tvd(counts, dist['all candidates'][d])])

    with open(TABLE_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['step', 'dimension', 'category', 'items', 'share'])
        writer.writerows(table_rows)
    with open(TVD_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['step', 'dimension', 'tvd_vs_candidates'])
        writer.writerows(tvd_rows)

    steps_short = [s for _, s in KEY_STEPS]
    for d in dims:
        print(f'\n{d.upper()} (share of items)')
        print(f'  {"":<28}' + ''.join(f'{s[:15]:>16}' for s in steps_short))
        for category in shown[d] + [REST]:
            cells = []
            for s in steps_short:
                c = dist[s][d]
                n = c[category] if category != REST else sum(c.values()) - sum(c[k] for k in shown[d])
                cells.append(f'{n / sum(c.values()):>15.1%} ')
            print(f'  {category[:28]:<28}' + ''.join(cells))
        print(f'  {"TVD vs all candidates":<28}' +
              ''.join(f'{tvd(dist[s][d], dist["all candidates"][d]):>15.3f} ' for s in steps_short))
    print(f'-> {TABLE_CSV}\n-> {TVD_CSV}')


if __name__ == '__main__':
    main()
