"""
rebuild_check.py
----------------
Rebuilds every released item from the released table maps and the parsed review tables,
and checks that the question and gold come out identical. This is the evidence for the
claim that every item can be rebuilt from the table maps without calling a model.

Inputs (all in the Zenodo deposit):
  release/v2/table_maps.jsonl.gz   the structure maps (core archive)
  data/v2/tables.jsonl.gz          the parsed review tables behind the released items
                                   (tables archive; every review is CC BY)
  release/v2/data/*.jsonl.gz       the released items

The items are rebuilt by 10_build_items.build_table_items, the same function the pipeline
uses; links to source papers are checked separately by the pipeline and not rebuilt here.

Usage:  python scripts/rebuild_check.py
"""
import gzip
import importlib.util
import json
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLITS = ['train-2025', 'train-2026', 'dev-2026', 'test-2026', 'test-2025']

spec = importlib.util.spec_from_file_location('build_items', os.path.join(BASE, 'scripts', '10_build_items.py'))
build_items = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build_items)


class NoLinker:
    """Links are verified by the pipeline's own checks; here only questions and golds."""
    def link(self, ref):
        return '', ''


def main():
    with gzip.open(os.path.join(BASE, 'data', 'v2', 'tables.jsonl.gz'), 'rt', encoding='utf-8') as f:
        tables = {(t['review_pmcid'], t['table_no']): t for t in map(json.loads, f)}
    rebuilt = {}
    with gzip.open(os.path.join(BASE, 'release', 'v2', 'table_maps.jsonl.gz'), 'rt', encoding='utf-8') as f:
        for rec in map(json.loads, f):
            if rec.get('status') != 'mapped':
                continue
            record = {**rec, 'map': build_items.unescape_map(rec['map'])}
            built, _ = build_items.build_table_items(tables[(rec['review_pmcid'], rec['table_no'])], record, NoLinker())
            rebuilt.update({b['item_id']: b for b in built})
    n = same = 0
    missing, differ = [], []
    for split in SPLITS:
        with gzip.open(os.path.join(BASE, 'release', 'v2', 'data', split + '.jsonl.gz'), 'rt', encoding='utf-8') as f:
            for it in map(json.loads, f):
                n += 1
                b = rebuilt.get(it['item_id'])
                if b is None:
                    missing.append(it['item_id'])
                elif b['question'] == it['question'] and b['gold'] == it['gold']:
                    same += 1
                else:
                    differ.append(it['item_id'])
    print(f'released items: {n:,} | rebuilt identically: {same:,} | missing: {len(missing)} | differ: {len(differ)}')
    if missing or differ:
        print('examples:', (missing + differ)[:5])
        raise SystemExit(1)


if __name__ == '__main__':
    main()
