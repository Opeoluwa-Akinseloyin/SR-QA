"""
18_answerability_check.py
-------------------------
Stage 4.2 of the v2 pipeline: does the LLM's answerability score - judged per
column WITHOUT seeing any paper - predict whether the gold answer can actually
be found in the source paper? And are the items the filter dropped found less
often? If so, the filter is justified by data rather than by assumption.

"Found" is graded, because reviewers rarely copy text exactly:
  exact    the normalised gold appears in the normalised paper text
  numbers  the gold contains numbers and every one appears in the paper
           ("OR 2.54 (1.30-4.96)" -> 2.54, 1.30, 4.96; "1,068" == "1068")
  words    >= WORD_SHARE of the gold's content words appear in the paper
  none     none of the above
Multi-fragment golds (" | ") are found at a level only if every fragment is.

The paper text is the whole article EXCEPT its reference list (author names and
years there would create false matches); tables and captions are included,
also those in <floats-group>. Gold "not reported"-type answers are skipped:
their answer should not be in the paper.

This is a LOWER bound on answerability: an answer that must be derived (e.g.
computed from two reported numbers) counts as "none" though the paper
supports it.

v1's check (old/scripts/analysis/facts_dataset.py) read <body> only, sampled
2,500 papers and kept answers <= 40 characters with one fragment; that subset
is reported too ("v1-comparable").

Inputs:  data/v2/items_split.jsonl (kept), data/v2/items_all.jsonl (dropped
         by the answerability filter, otherwise eligible), SOURCE_XML/
Outputs: data/v2/answerability_check.csv  item_id, group, category, score,
                                          content, level

Usage:
  python scripts/18_answerability_check.py
"""
import collections
import csv
import json
import os
import re
import unicodedata
import xml.etree.ElementTree as ET
from multiprocessing import Pool

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V2 = os.path.join(BASE, 'data', 'v2')
SOURCE_DIR = os.path.join(BASE, 'SOURCE_XML')
OUT_CSV = os.path.join(V2, 'answerability_check.csv')

WORD_SHARE = 0.8
SEPARATOR = ' | '
NUMBER = re.compile(r'\d+(?:[.,]\d+)*')
WORD = re.compile(r'[a-z]{3,}')
STOPWORDS = {'the', 'and', 'for', 'with', 'was', 'were', 'not', 'from', 'that', 'this', 'are',
             'had', 'has', 'have', 'all', 'each', 'per', 'than', 'but', 'its', 'their', 'who'}
# "Not reported" golds (answers that should NOT be in the paper): the one definition in
# 11_filter_items.py, imported (v2.0.1; this script used to keep its own copy).
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location('filter_items', os.path.join(BASE, 'scripts', '11_filter_items.py'))
_filter_items = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_filter_items)
is_absence = _filter_items.is_absence


def normalise(text):
    """Lowercase, NFKC, unify dashes/quotes/minus, single spaces."""
    text = unicodedata.normalize('NFKC', text).lower()
    for dash in ('–', '—', '−', '‐', '‑'):
        text = text.replace(dash, '-')
    for quote in ('‘', '’', '“', '”'):
        text = text.replace(quote, "'")
    return ' '.join(text.split())


def numbers_of(text):
    """Numbers as canonical strings: thousands separators dropped, '1,5' kept as is."""
    out = set()
    for n in NUMBER.findall(text):
        if re.fullmatch(r'\d{1,3}(,\d{3})+', n):
            n = n.replace(',', '')
        out.add(n.rstrip('.,'))
    return out


def paper_text(pmcid):
    """Normalised text of the whole article except its reference list."""
    root = ET.parse(os.path.join(SOURCE_DIR, pmcid + '.xml')).getroot()
    for parent in root.iter():
        for child in list(parent):
            if child.tag.rsplit('}', 1)[-1] in ('ref-list', 'back'):
                parent.remove(child)
    return normalise(' '.join(root.itertext()))


def level(gold, text, text_numbers, text_words):
    """The best level at which every fragment of the gold is found."""
    fragments = [normalise(f) for f in gold.split(SEPARATOR) if f.strip()]
    best_per_fragment = []
    for fragment in fragments:
        if fragment in text:
            best_per_fragment.append(3)
            continue
        numbers = numbers_of(fragment)
        if numbers and numbers <= text_numbers:
            best_per_fragment.append(2)
            continue
        words = {w for w in WORD.findall(fragment) if w not in STOPWORDS}
        if words and len(words & text_words) / len(words) >= WORD_SHARE:
            best_per_fragment.append(1)
            continue
        best_per_fragment.append(0)
    return ['none', 'words', 'numbers', 'exact'][min(best_per_fragment)] if fragments else 'none'


def check_paper(job):
    pmcid, items = job
    try:
        text = paper_text(pmcid)
    except (ET.ParseError, OSError):
        return [(i['item_id'], 'unreadable') for i in items]
    text_numbers = numbers_of(text)
    text_words = set(WORD.findall(text))
    return [(i['item_id'], level(i['gold'], text, text_numbers, text_words)) for i in items]


def load_items():
    """Kept items, plus items the answerability filter dropped (otherwise eligible)."""
    items = []
    with open(os.path.join(V2, 'items_split.jsonl'), encoding='utf-8') as f:
        for item in map(json.loads, f):
            items.append(dict(item, group='kept'))
    with open(os.path.join(V2, 'items_all.jsonl'), encoding='utf-8') as f:
        for item in map(json.loads, f):
            if item['source_status'] != 'ccby_full_text' or not item['gold'].strip():
                continue
            kept = (item['answerability_category'] == 'reported'
                    and item['answerability_score'] >= 3)
            if not kept:
                items.append(dict(item, group='dropped'))
    return [i for i in items if not is_absence(i['gold'])]


def main():
    items = load_items()
    by_paper = collections.defaultdict(list)
    for item in items:
        by_paper[item['source_pmcid']].append(item)
    with Pool() as pool:
        results = dict(r for chunk in pool.imap_unordered(check_paper, by_paper.items(), chunksize=20)
                       for r in chunk)

    rows = []
    for item in items:
        rows.append({'item_id': item['item_id'], 'group': item['group'],
                     'category': item['answerability_category'],
                     'score': item['answerability_score'], 'content': item['content'],
                     'short_single': int(len(item['gold']) <= 40 and SEPARATOR not in item['gold']),
                     'level': results[item['item_id']]})
    with open(OUT_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    def table(title, key):
        print(f'\n{title}')
        print(f'  {"":<34}{"items":>9}{"exact":>8}{"+numbers":>10}{"+words":>9}{"none":>8}')
        groups = collections.defaultdict(list)
        for row in rows:
            groups[key(row)].append(row['level'])
        for name in sorted(groups, key=str):
            levels = collections.Counter(groups[name])
            n = len(groups[name])
            exact = levels['exact'] / n
            numbers = exact + levels['numbers'] / n
            words = numbers + levels['words'] / n
            print(f'  {str(name):<34}{n:>9,}{exact:>8.1%}{numbers:>10.1%}{words:>9.1%}'
                  f'{levels["none"] / n:>8.1%}')

    print('found = cumulative share (exact; exact or all numbers; ... or 80% of words)')
    table('BY FILTER DECISION', lambda r: r['group'])
    table('KEPT ITEMS BY ANSWERABILITY SCORE', lambda r: f"score {r['score']}" if r['group'] == 'kept' else None)
    table('DROPPED ITEMS BY REASON', lambda r: (r['category'] if r['category'] != 'reported'
                                                else f"reported, score {r['score']}")
          if r['group'] == 'dropped' else None)
    table('KEPT ITEMS BY CONTENT', lambda r: r['content'] if r['group'] == 'kept' else None)
    table('v1-COMPARABLE (kept, gold <= 40 chars, one fragment)',
          lambda r: 'kept, short single' if r['group'] == 'kept' and r['short_single'] else None)
    print(f'\n-> {OUT_CSV}')


if __name__ == '__main__':
    main()
