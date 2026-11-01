"""
13_make_splits.py
-----------------
Stage 3.4 of the v2 pipeline: split the items into train / dev / test arms by
the REVIEW's online publication date, keeping every source paper in one arm.

Design (decided 2026-10-05, option c):

  2026 (online in 2026)   train 65%  |  dev 10%  |  test 25%   <- ECIR headline:
                                                                   a post-cutoff test set
  2025 (online in 2025)   train 80%  |  test-pre 20%           <- secondary, BOUNDED
                                                                   contamination check

  * The year is the review's ONLINE date (epub / date-type="pub"), the first
    day a crawler could have seen it. v1 cut on the issue year (OpenAlex), which
    put 3.7% of its evaluation reviews on the wrong side.
  * Graph-disjoint: reviews joined by a shared source paper form one connected
    component and always land in the same arm, so no paper is in two arms.
    Components with reviews from BOTH years are forced into train.
  * Balanced dealing (v1's default): components are dealt in random order,
    each to the arm whose unmet per-field quota it best fills, so every arm
    approaches its year's field mix. Random order, not largest-first, keeps
    review counts in the small eval arms high (review-level N drives the
    clustered standard errors).
  * The no-leakage check runs here and the script stops if any path leaks.

Ported from old/scripts/analysis/make_splits.py (UnionFind, assign_year,
tvd - unchanged) and old/scripts/analysis/review_dates.py (online date).

Inputs:  data/v2/items_filtered.jsonl, data/v2/review_topics.csv, PMC_XML_/
Outputs: data/v2/review_splits.csv   review_pmcid, online, issue, date_source,
                                     year, field, component, split
         data/v2/items_split.jsonl   the items with `split` and `review_online`
                                     (`arm` stays the item's group label)

Usage:
  python scripts/13_make_splits.py            # seed 42, as v1
"""
import argparse
import collections
import csv
import json
import os
import random
import sys
import xml.etree.ElementTree as ET

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ITEMS_PATH = os.path.join(BASE, 'data', 'v2', 'items_filtered.jsonl')
TOPICS_CSV = os.path.join(BASE, 'data', 'v2', 'review_topics.csv')
REVIEW_XML_DIR = os.path.join(BASE, 'PMC_XML_')
SPLITS_CSV = os.path.join(BASE, 'data', 'v2', 'review_splits.csv')
ITEMS_OUT = os.path.join(BASE, 'data', 'v2', 'items_split.jsonl')

TARGETS = {
    '2026': [('train', 0.65), ('dev', 0.10), ('test', 0.25)],
    '2025': [('train', 0.80), ('test-pre', 0.20)],
}
ARM = {('train', '2026'): 'train-2026', ('dev', '2026'): 'dev-2026',
       ('test', '2026'): 'test-2026', ('train', '2025'): 'train-2025',
       ('test-pre', '2025'): 'test-2025'}
ARM_ORDER = ['train-2026', 'dev-2026', 'test-2026', 'train-2025', 'test-2025']
EVAL_ARMS = {'dev-2026', 'test-2026', 'test-2025'}
SIZE_TOLERANCE = 1.02

ONLINE = ('epub', 'pub', 'epub-ppub')          # first public appearance
ISSUE = ('ppub', 'collection')                 # volume/issue date


# ---------------------------------------------------------------------------
# 1. Online dates (from v1's review_dates.py)
# ---------------------------------------------------------------------------
def review_date(pmcid):
    """((y, m, d) online, (y, m) issue or None, date label), or None."""
    path = os.path.join(REVIEW_XML_DIR, pmcid + '.xml')
    if not os.path.exists(path):
        return None
    candidates = []
    for pub_date in ET.parse(path).getroot().iterfind('.//front/article-meta/pub-date'):
        year = (pub_date.findtext('year') or '').strip()
        if not year.isdigit():
            continue
        month = (pub_date.findtext('month') or '').strip()
        day = (pub_date.findtext('day') or '').strip()
        candidates.append((pub_date.get('pub-type') or pub_date.get('date-type'), int(year),
                           int(month) if month.isdigit() else 1, int(day) if day.isdigit() else 1))
    if not candidates:
        return None
    issue = [c for c in candidates if c[0] in ISSUE]
    issue_ym = min(issue, key=lambda c: c[1:4])[1:3] if issue else None
    online = [c for c in candidates if c[0] in ONLINE]
    best = min(online or candidates, key=lambda c: c[1:4])
    return best[1:4], issue_ym, best[0]


# ---------------------------------------------------------------------------
# 2. Components and dealing (from v1's make_splits.py, unchanged)
# ---------------------------------------------------------------------------
class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:      # path compression
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def tvd(a, b):
    """Total variation distance between two count distributions."""
    ta, tb = sum(a.values()), sum(b.values())
    if not ta or not tb:
        return float('nan')
    return 0.5 * sum(abs(a[k] / ta - b[k] / tb) for k in set(a) | set(b))


def assign_year(pure, year, targets, committed_qa, global_field, rnd):
    """Deal one year's single-year components to roles: {cid: role}.

    committed_qa: items already in train for this year via the forced
    cross-year components - it counts against train's size target.
    """
    total = sum(c['qa'][year] for c in pure) + committed_qa
    caps = {role: total * share for role, share in targets}
    out = {}
    cur_qa = collections.Counter()
    cur_qa['train'] = committed_qa
    cur_field = {role: collections.Counter() for role, _ in targets}
    global_total = sum(global_field.values()) or 1
    quota = {role: {f: caps[role] * n / global_total for f, n in global_field.items()}
             for role, _ in targets}

    ordered = pure[:]
    rnd.shuffle(ordered)
    for c in ordered:
        size = c['qa'][year]
        fields = c['field'][year]
        candidates = [role for role, _ in targets
                      if cur_qa[role] + size <= caps[role] * SIZE_TOLERANCE]
        if not candidates:
            candidates = [max((r for r, _ in targets), key=lambda r: caps[r] - cur_qa[r])]

        def score(role):
            filled = 0.0
            for f, n in fields.items():
                deficit = quota[role][f] - cur_field[role][f]
                if deficit > 0:
                    filled += min(n, deficit)
            headroom = (caps[role] - cur_qa[role]) / caps[role] if caps[role] else 0
            return (filled / size if size else 0) + 0.05 * headroom

        best = max(candidates, key=score)
        out[c['cid']] = best
        cur_qa[best] += size
        cur_field[best].update(fields)
    return out


# ---------------------------------------------------------------------------
# 3. Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--reuse', metavar='CSV',
                        help='keep each review in the arm a previous review_splits.csv gave it '
                             '(used after item-level fixes, so audited items stay in their arms); '
                             'the leakage check still runs')
    args = parser.parse_args()

    with open(ITEMS_PATH, encoding='utf-8') as f:
        items = [json.loads(line) for line in f]
    with open(TOPICS_CSV, encoding='utf-8') as f:
        field_of = {r['review_pmcid']: r['field'] or 'Unknown' for r in csv.DictReader(f)}

    dates, no_date = {}, []
    for review in sorted({i['review_pmcid'] for i in items}):
        got = review_date(review)
        if got is None:
            no_date.append(review)
        else:
            dates[review] = got
    if no_date:
        sys.exit(f'{len(no_date)} reviews have no publication date, e.g. {no_date[:3]}')
    year_of = {r: '2026' if d[0][0] >= 2026 else '2025' for r, d in dates.items()}

    uf = UnionFind()
    rev_qa = collections.Counter()
    rev_papers = collections.defaultdict(set)
    for item in items:
        review, paper = item['review_pmcid'], item['source_pmcid']
        rev_qa[review] += 1
        rev_papers[review].add(paper)
        uf.union(('R', review), ('P', paper))

    comp_reviews = collections.defaultdict(list)
    for review in rev_qa:
        comp_reviews[uf.find(('R', review))].append(review)
    comps = []
    for cid, reviews in comp_reviews.items():
        qa = collections.Counter()
        field = collections.defaultdict(collections.Counter)
        for review in reviews:
            qa[year_of[review]] += rev_qa[review]
            field[year_of[review]][field_of[review]] += rev_qa[review]
        years = {year_of[r] for r in reviews}
        comps.append({'cid': cid, 'reviews': reviews, 'qa': qa, 'field': field,
                      'mixed': len(years) > 1, 'years': years})

    year_totals = collections.Counter()
    global_field = collections.defaultdict(collections.Counter)
    for review in rev_qa:
        year_totals[year_of[review]] += rev_qa[review]
        global_field[year_of[review]][field_of[review]] += rev_qa[review]
    mixed = [c for c in comps if c['mixed']]
    print(f'items {len(items):,} | reviews {len(rev_qa):,} | components {len(comps):,} | '
          f'cross-year components forced to train {len(mixed):,} '
          f'({sum(sum(c["qa"].values()) for c in mixed):,} items)')
    for year in sorted(year_totals):
        n_rev = sum(1 for r in rev_qa if year_of[r] == year)
        print(f'  online {year}: {n_rev:,} reviews, {year_totals[year]:,} items')

    assign = {c['cid']: 'train' for c in mixed}
    rnd = random.Random(args.seed)
    for year, targets in TARGETS.items():
        pure = [c for c in comps if not c['mixed'] and c['years'] == {year}]
        committed = sum(c['qa'][year] for c in mixed)
        assign.update(assign_year(pure, year, targets, committed, global_field[year], rnd))

    arm_of, comp_of = {}, {}
    for c in comps:
        for review in c['reviews']:
            arm_of[review] = ARM[(assign[c['cid']], year_of[review])]
            comp_of[review] = c['cid'][1]
    if args.reuse:
        with open(args.reuse, encoding='utf-8') as f:
            # 'split' since 2026-10-07; older files called the column 'arm'
            previous = {r['review_pmcid']: r.get('split') or r['arm'] for r in csv.DictReader(f)}
        unknown = sorted(set(arm_of) - set(previous))
        if unknown:
            sys.exit(f'--reuse: {len(unknown)} reviews have no previous arm, e.g. {unknown[:3]}')
        moved = sum(arm_of[r] != previous[r] for r in arm_of)
        arm_of = {r: previous[r] for r in arm_of}
        print(f'  --reuse: kept every review in its previous arm '
              f'({moved} would have moved under a fresh deal)')

    # --- no paper may be in two arms, unless both are train arms ---
    papers_by_arm = collections.defaultdict(set)
    for review, arm in arm_of.items():
        papers_by_arm[arm] |= rev_papers[review]
    leaks = []
    for a in ARM_ORDER:
        for b in ARM_ORDER:
            if a < b and (a in EVAL_ARMS or b in EVAL_ARMS):
                shared = papers_by_arm[a] & papers_by_arm[b]
                if shared:
                    leaks.append((a, b, len(shared)))
    if leaks:
        sys.exit(f'LEAKAGE between arms: {leaks}')
    benign = len(papers_by_arm['train-2025'] & papers_by_arm['train-2026'])

    arm_qa, arm_reviews = collections.Counter(), collections.Counter()
    arm_field = collections.defaultdict(collections.Counter)
    for review, arm in arm_of.items():
        arm_qa[arm] += rev_qa[review]
        arm_reviews[arm] += 1
        arm_field[arm][field_of[review]] += rev_qa[review]
    print(f'\n  {"arm":<15}{"reviews":>8}{"papers":>8}{"items":>9}{"% of year":>11}')
    for arm in ARM_ORDER:
        year = arm.rsplit('-', 1)[1]
        print(f'  {arm:<15}{arm_reviews[arm]:>8,}{len(papers_by_arm[arm]):>8,}'
              f'{arm_qa[arm]:>9,}{arm_qa[arm] / year_totals[year]:>10.1%}')
    print('  no paper shared between an evaluation arm and any other arm: verified')
    print(f'  (benign: {benign} papers in both train arms)')
    print('\n  field TVD (0 = same mix):')
    for a, b in (('dev-2026', 'test-2026'), ('train-2026', 'test-2026'),
                 ('train-2025', 'test-2025'), ('test-2026', 'test-2025')):
        print(f'    {a:<15} vs {b:<15} {tvd(arm_field[a], arm_field[b]):.3f}')

    with open(SPLITS_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['review_pmcid', 'online', 'issue', 'date_source', 'year', 'field',
                         'component', 'split'])
        for review in sorted(arm_of):
            (y, m, d), issue, label = dates[review]
            writer.writerow([review, f'{y:04d}-{m:02d}-{d:02d}',
                             f'{issue[0]:04d}-{issue[1]:02d}' if issue else '', label,
                             year_of[review], field_of[review], comp_of[review], arm_of[review]])
    with open(ITEMS_OUT, 'w', encoding='utf-8') as f:
        for item in items:
            (y, m, d), _, _ = dates[item['review_pmcid']]
            # 'split', not 'arm': 'arm' is the item's group label (e.g. "athletes")
            item['split'] = arm_of[item['review_pmcid']]
            item['review_online'] = f'{y:04d}-{m:02d}-{d:02d}'
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f'\n-> {SPLITS_CSV}\n-> {ITEMS_OUT}')


if __name__ == '__main__':
    main()
