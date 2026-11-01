"""
map_retest.py
-------------
Annotator reliability for the paper (spec task 2.6, "test-retest"): the same model
(Claude Sonnet 5) mapped the 200 pilot tables twice, with the original and the final
prompt (runs pilot-sonnet-5 and pilot-sonnet-5-v2). Free: reads the stored maps.

  acceptance   share of tables (usable in both runs) whose accept decision differs
  category     on extraction columns present in both maps, share with the same
               answerability category; and share whose score differs by more than 1

Output: data/v2/analysis/map_retest.json

Usage:  python scripts/analysis/map_retest.py
"""
import json
import os

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MAPS = os.path.join(BASE, 'data', 'v2', 'maps')
RUNS = ('pilot-sonnet-5', 'pilot-sonnet-5-v2')
OUT = os.path.join(BASE, 'data', 'v2', 'analysis', 'map_retest.json')


def load(run):
    with open(os.path.join(MAPS, run, 'maps.jsonl'), encoding='utf-8') as f:
        return {r['custom_id']: r for r in map(json.loads, f) if r.get('usable')}


def main():
    a, b = (load(r) for r in RUNS)
    both = sorted(a.keys() & b.keys())
    flips = [t for t in both if bool(a[t]['map']['accept']) != bool(b[t]['map']['accept'])]
    same_cat = n_cols = score_gap = 0
    for t in both:
        ca = {c['col_index']: c for c in a[t]['map']['columns'] if c['role'] == 'extraction'}
        cb = {c['col_index']: c for c in b[t]['map']['columns'] if c['role'] == 'extraction'}
        for i in ca.keys() & cb.keys():
            n_cols += 1
            same_cat += ca[i]['answerability_category'] == cb[i]['answerability_category']
            score_gap += abs(ca[i]['answerability_score'] - cb[i]['answerability_score']) > 1
    out = {'runs': RUNS, 'tables_usable_in_both': len(both),
           'acceptance_flips': len(flips), 'acceptance_flip_rate': round(len(flips) / len(both), 4),
           'acceptance_agreement': round(1 - len(flips) / len(both), 4),
           'extraction_columns_in_both': n_cols,
           'category_agreement': round(same_cat / n_cols, 4),
           'score_gap_over_1_rate': round(score_gap / n_cols, 4),
           'flipped_tables': flips}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != 'flipped_tables'}, indent=1))


if __name__ == '__main__':
    main()
