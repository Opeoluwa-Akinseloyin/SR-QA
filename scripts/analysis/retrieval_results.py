"""
retrieval_results.py
--------------------
Phase 6 retrieval intervals for the ECIR paper (M3 + the agreed significance testing).
Free: reads the run files, no model calls.

Inputs:  data/v2/retrieval_labels/test-2026/{queries,qrels}.jsonl and passages.jsonl.gz
         (ALL test-2026 retrieval queries; agreed 2026-10-12), and the runs in
         data/v2/eval/retrieval_full/runs/<run>.jsonl
Metrics: per query, exactly as retrieval.py `score` (imported, not re-implemented):
         recall@5 / recall@10 (= at least one relevant passage in the top k), MRR,
         nDCG@10 (graded); groups: all, content type, evidence type (table vs text)

Intervals (agreed 2026-10-10, decision C2; same scheme as qa_results.py):
  cluster bootstrap over REVIEWS (a review's questions are correlated), 2,000 replicates,
  percentile 95% CIs. Every run is scored on the same queries, so one set of resampled
  reviews serves all runs and every paired difference.
  Paired differences between runs: the mean per-query difference, its CI, and a two-sided
  bootstrap p-value; Holm correction over the run pairs, separately for each metric.

Output: data/v2/analysis/retrieval_results.json (and a readable print-out)

Usage:  python scripts/analysis/retrieval_results.py
"""
import collections
import gzip
import importlib.util
import itertools
import json
import os
import sys

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LABELS = os.path.join(BASE, 'data', 'v2', 'retrieval_labels', 'test-2026')
RUNS = os.path.join(BASE, 'data', 'v2', 'eval', 'retrieval_full', 'runs')
OUT = os.path.join(BASE, 'data', 'v2', 'analysis', 'retrieval_results.json')

N_BOOT = 2000
SEED = 20261012
RUN_ORDER = ['bm25', 'medcpt', 'hybrid_bm25+medcpt', 'bge-m3', 'hybrid_bm25+bge-m3', 'bge-m3+rerank']
METRICS = ['recall@5', 'recall@10', 'mrr', 'ndcg@10']

try:
    sys.stdout.reconfigure(encoding='utf-8')
except (AttributeError, ValueError):
    pass

spec = importlib.util.spec_from_file_location('retrieval', os.path.join(BASE, 'scripts', 'eval', 'retrieval.py'))
retrieval = importlib.util.module_from_spec(spec)
spec.loader.exec_module(retrieval)


def read_jsonl(path, opener=open):
    with opener(path, 'rt', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def load():
    """Per-query metric matrix: runs x queries x metrics, plus each query's review and groups."""
    queries = read_jsonl(os.path.join(LABELS, 'queries.jsonl'))
    qrels = {r['item_id']: {p: g['grade'] for p, g in r['qrel'].items()}
             for r in read_jsonl(os.path.join(LABELS, 'qrels.jsonl'))}
    ptype = {p['pid']: p['type'] for p in read_jsonl(os.path.join(LABELS, 'passages.jsonl.gz'), gzip.open)}
    with gzip.open(os.path.join(BASE, 'release', 'v2', 'data', 'test-2026.jsonl.gz'), 'rt', encoding='utf-8') as f:
        review_of = {it['item_id']: it['review_pmcid'] for it in map(json.loads, f)}
    qids = [q['item_id'] for q in queries]
    info = []
    for q in queries:
        qrel = qrels[q['item_id']]
        best = max(qrel.values())
        where = 'table' if any(ptype[p] in ('cell', 'row') for p, g in qrel.items() if g == best) else 'text'
        info.append({'review': review_of[q['item_id']], 'content': q['content'], 'evidence': where})
    runs = [r for r in RUN_ORDER if os.path.exists(os.path.join(RUNS, r + '.jsonl'))]
    scores = np.zeros((len(runs), len(qids), len(METRICS)))
    for ri, run in enumerate(runs):
        rows = read_jsonl(os.path.join(RUNS, run + '.jsonl'))
        ranking = {r['item_id']: [pid for pid, _ in r['ranking']] for r in rows[1:]}
        # every current query must be ranked; a run may hold extra queries that a later label
        # rule excluded (v2.0.1 dropped symbol-only golds) - they are simply not scored
        missing = set(qids) - set(ranking)
        assert not missing, f'{run} lacks {len(missing)} test-2026 queries'
        for qi, qid in enumerate(qids):
            m = retrieval.metrics_for(ranking[qid], qrels[qid])
            scores[ri, qi] = [m[k] for k in METRICS]
    return runs, info, scores


def review_sums(mask, info, scores):
    """Per review: summed metrics (runs x reviews x metrics) and query counts, over masked queries."""
    reviews = sorted({info[i]['review'] for i in np.flatnonzero(mask)})
    col = {rv: k for k, rv in enumerate(reviews)}
    sums = np.zeros((scores.shape[0], len(reviews), scores.shape[2]))
    counts = np.zeros(len(reviews))
    for i in np.flatnonzero(mask):
        k = col[info[i]['review']]
        sums[:, k, :] += scores[:, i, :]
        counts[k] += 1
    return sums, counts


def holm(pvalues):
    """Holm-adjusted p-values (step-down, monotone), same order as the input."""
    order = np.argsort(pvalues)
    m = len(pvalues)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvalues[i]))
        adj[i] = running
    return adj


def main():
    rng = np.random.default_rng(SEED)
    runs, info, scores = load()
    groups = {'all': np.ones(len(info), bool)}
    for key in ('content', 'evidence'):
        for val in sorted({x[key] for x in info}):
            groups[f'{key}={val}'] = np.array([x[key] == val for x in info])

    report = {'note': ('cluster bootstrap over reviews, %d replicates, percentile 95%% CIs; paired '
                       'differences share the resampled reviews; Holm over run pairs per metric' % N_BOOT),
              'queries': len(info), 'reviews': len({x['review'] for x in info}),
              'runs': runs, 'per_run': {}, 'paired_differences': {}}
    for g, mask in groups.items():
        sums, counts = review_sums(mask, info, scores)
        idx = rng.integers(0, len(counts), size=(N_BOOT, len(counts)))
        point = sums.sum(1) / counts.sum()                                   # runs x metrics
        boot = sums[:, idx, :].sum(2) / counts[idx].sum(1)[None, :, None]    # runs x boot x metrics
        lo, hi = np.percentile(boot, [2.5, 97.5], axis=1)
        report['per_run'][g] = {
            run: {'n': int(counts.sum()), 'reviews': len(counts),
                  **{m: {'value': round(float(point[ri, mi]), 4),
                         'ci95': [round(float(lo[ri, mi]), 4), round(float(hi[ri, mi]), 4)]}
                     for mi, m in enumerate(METRICS)}}
            for ri, run in enumerate(runs)}
        if g != 'all':
            continue
        pairs = list(itertools.combinations(range(len(runs)), 2))
        diffs = {}
        for mi, m in enumerate(METRICS):
            rows, pvals = [], []
            for a, b in pairs:
                d_point = point[b, mi] - point[a, mi]
                d_boot = boot[b, :, mi] - boot[a, :, mi]
                p = min(1.0, 2 * min((d_boot <= 0).mean(), (d_boot >= 0).mean()))
                dlo, dhi = np.percentile(d_boot, [2.5, 97.5])
                rows.append({'pair': f'{runs[b]} - {runs[a]}', 'diff': round(float(d_point), 4),
                             'ci95': [round(float(dlo), 4), round(float(dhi), 4)], 'p_boot': round(float(p), 4)})
                pvals.append(p)
            for row, p_adj in zip(rows, holm(np.array(pvals))):
                row['p_holm'] = round(float(p_adj), 4)
                row['significant_0.05'] = bool(p_adj < 0.05)
            diffs[m] = rows
        report['paired_differences'] = diffs

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(report, f, indent=1)

    print(f"test-2026: {report['queries']:,} queries from {report['reviews']} reviews; "
          f"95% CI, review cluster bootstrap")
    for run in runs:
        r = report['per_run']['all'][run]
        print(f'  {run:<20}' + '  '.join(f"{m} {r[m]['value']:.3f} [{r[m]['ci95'][0]:.3f}, {r[m]['ci95'][1]:.3f}]"
                                         for m in ('recall@10', 'mrr')))
    print('paired differences, recall@10 (Holm over %d pairs):' % len(report['paired_differences']['recall@10']))
    for row in report['paired_differences']['recall@10']:
        print(f"  {row['pair']:<42} {row['diff']:+.3f} [{row['ci95'][0]:+.3f}, {row['ci95'][1]:+.3f}]"
              f"  p_holm {row['p_holm']:.4f}{'  *' if row['significant_0.05'] else ''}")
    print(f'-> {os.path.relpath(OUT, BASE)}')


if __name__ == '__main__':
    main()
