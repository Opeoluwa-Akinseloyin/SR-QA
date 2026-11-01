"""
qa_results.py
-------------
Phase 6 (M5, M6's Kendall tau, and the should-haves) for the ECIR paper: the QA results of
the 10 API models, with 95% intervals. Free: reads the run outputs, no API calls.

Inputs (per model, data/v2/eval/answers/eval_sample_linear/<model>/):
  answers.json         locatability (quote found verbatim) per answered item
  judge/judged.json    per-item faithful / correct verdicts, abstention class, weight
and data/v2/eval/eval_sample.jsonl (split, review online date), and
data/v2/eval/crossjudge/crossjudge_result.json (Gemini vs Sonnet, for one tau).

Definitions (as agreed in the spec, Phase 5):
  faithful      judged fully or partially supported (items with a faithfulness verdict)
  correct       match_level >= minor_deviation (items with a correctness verdict)
  pooled rates  weighted by the sample's weights (characteristic ~2.78); per-content unweighted
  locatable     answered with a value AND its quote found verbatim, over answered-with-value
  abstention    on gold "not reported" items: correct abstention rate = abstained / all;
                on reported items: missed-value rate = "not reported" answers / all
  2x2           items with both verdicts; P(correct | unfaithful)

Intervals: cluster bootstrap over REVIEWS (items of one review are correlated; agreed
2026-10-11), 2,000 replicates, percentile 95%. Differences between models are paired on the
items both have verdicts for, resampling reviews, with a two-sided bootstrap p-value and Holm
correction over the 45 model pairs per axis (added 2026-10-13).

Analyses:
  1. per model x split (test-2026 headline, test-2025): all rates with CIs
  2. paired differences between every pair of models (faithful, correct) on test-2026
  3. Kendall tau: faithfulness ranking vs correctness ranking over the 10 models (with a
     bootstrap interval); Gemini vs Sonnet faithfulness ranking over the 4 cross-judged
     models (n = 4: descriptive only)
  4. month check (rule (c) models assumed to have a 2025 cutoff): (Jan - Apr..Jun) rate for
     rule (c) models minus the same for the other models, test-2026; leakage would show as a
     positive difference in correctness
  5. test-2025 minus test-2026, per model and pooled
  6. zone map: the section type each located quote comes from (abstract, methods, results,
     tables, ...), per model and by content type, and faithful / correct rates by zone

Output: data/v2/analysis/qa_results.json (and a readable print-out)

Usage:
  python scripts/analysis/qa_results.py
"""
import collections
import csv
import gzip
import itertools
import json
import os
import sys

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ANSWERS = os.path.join(BASE, 'data', 'v2', 'eval', 'answers', 'eval_sample_linear')
SAMPLE = os.path.join(BASE, 'data', 'v2', 'eval', 'eval_sample.jsonl')
CROSSJUDGE = os.path.join(BASE, 'data', 'v2', 'eval', 'crossjudge', 'crossjudge_result.json')
OUT = os.path.join(BASE, 'data', 'v2', 'analysis', 'qa_results.json')

N_BOOT = 2000
SEED = 20261011

# folder name -> (display name, eligibility rule from spec task 5.0)
MODELS = {
    'qwen3.5-397b-a17b': ('Qwen3.5-397B-A17B', 'c'),
    'qwen3.5-122b-a10b': ('Qwen3.5-122B-A10B', 'c'),
    'qwen3.5-35b-a3b': ('Qwen3.5-35B-A3B', 'c'),
    'qwen3.5-27b': ('Qwen3.5-27B', 'c'),
    'qwen3.5-9b': ('Qwen3.5-9B', 'c'),
    'glm-5': ('GLM-5', 'c'),
    'gemma-4-31b-it': ('Gemma 4 31B', 'b'),
    'deepseek-v3.2': ('DeepSeek V3.2', 'a'),
    'llama-4-maverick': ('Llama 4 Maverick', 'a'),
    'llama-4-scout': ('Llama 4 Scout', 'a'),
}
SPLITS = ['test-2026', 'test-2025']

try:
    sys.stdout.reconfigure(encoding='utf-8')
except (AttributeError, ValueError):
    pass


# --------------------------------------------------------------------------- #
# loading: one row per (model, item)
# --------------------------------------------------------------------------- #
def load():
    with open(SAMPLE, encoding='utf-8') as f:
        sample = {r['item_id']: r for r in map(json.loads, f)}
    release = release_items()
    rows = {}
    for model in MODELS:
        with open(os.path.join(ANSWERS, model, 'answers.json'), encoding='utf-8') as f:
            answers = json.load(f)
        located = {}
        for inst in answers['instances']:
            # the zone (abstract, methods, results, tables, ...) of each section, from the reader
            zone_of = {s['id']: s.get('zone') or 'other' for s in inst['source_sections']}
            for q in inst['questions']:
                p = q['prediction']
                value = bool(p['answered'] and not p['not_reported'])
                quote = bool(value and p['provenance'] and p['provenance'].get('quote_found'))
                zone = zone_of.get(p['provenance']['located_section_id']) if quote else None
                for item_id in q['item_ids']:
                    located[item_id] = (value, quote, zone)
        with open(os.path.join(ANSWERS, model, 'judge', 'judged.json'), encoding='utf-8') as f:
            judged = json.load(f)['items']
        out = []
        for r in judged:
            if r['item_id'] not in release:          # dropped in v2.0.1 (blank / footnote gold)
                continue
            s = sample[r['item_id']]
            value, quote, zone = located.get(r['item_id'], (False, False, None))
            absence, abstention, correct = r['gold_is_absence'], r['abstention'], r['correct']
            if release[r['item_id']] != absence:      # flipped to "not reported" in v2.0.1
                absence = release[r['item_id']]
                if abstention != 'missing_from_response':
                    abstention = 'over_answer' if value else 'correct_abstention'
                correct = None                         # correctness is not graded against "not reported"
            out.append({
                'item': r['item_id'], 'review': r['instance_id'].split('__')[0],
                'split': s['split'], 'month': s['review_online_date'][:7],
                'content': r['content'], 'weight': r['weight'] if r['judge'] else 0.0,
                'absence': absence, 'abstention': abstention,
                'faithful': r['faithful'], 'correct': correct,
                'value': value, 'quote': quote, 'zone': zone,
            })
        rows[model] = out
    return rows


def release_items():
    """item_id -> gold_is_absence for the evaluation splits of the CURRENT release (v2.0.1,
    2026-10-14). The answers and verdicts were made on v2.0: items v2.0.1 dropped are left
    out, and items it re-flagged as "not reported" are reclassified, without re-judging."""
    out = {}
    for split in SPLITS:
        with gzip.open(os.path.join(BASE, 'release', 'v2', 'data', split + '.jsonl.gz'), 'rt', encoding='utf-8') as f:
            for it in map(json.loads, f):
                out[it['item_id']] = it['gold_is_absence']
    return out


# --------------------------------------------------------------------------- #
# ratio estimates with a cluster bootstrap over reviews
# --------------------------------------------------------------------------- #
def ratio_ci(rows, num, den, rng):
    """sum(num)/sum(den) with a review-cluster bootstrap. num/den: functions of a row."""
    by_review = collections.defaultdict(lambda: [0.0, 0.0])
    for r in rows:
        d = den(r)
        if d:
            by_review[r['review']][0] += num(r)
            by_review[r['review']][1] += d
    if not by_review:
        return None
    sums = np.array(list(by_review.values()))
    point = sums[:, 0].sum() / sums[:, 1].sum()
    idx = rng.integers(0, len(sums), size=(N_BOOT, len(sums)))
    boot = sums[idx, 0].sum(axis=1) / sums[idx, 1].sum(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {'rate': round(float(point), 4), 'ci95': [round(float(lo), 4), round(float(hi), 4)],
            'n': int(sum(1 for r in rows if den(r))), 'reviews': len(sums)}


def metric_defs():
    """name -> (row filter, numerator, denominator)."""
    def has_f(r):
        return r['faithful'] is not None

    def has_c(r):
        return r['correct'] is not None
    return {
        'faithful_pooled': (lambda r: True, lambda r: r['weight'] * bool(r['faithful']),
                            lambda r: r['weight'] if has_f(r) else 0),
        'correct_pooled': (lambda r: True, lambda r: r['weight'] * bool(r['correct']),
                           lambda r: r['weight'] if has_c(r) else 0),
        'faithful_characteristic': (lambda r: r['content'] == 'characteristic',
                                    lambda r: bool(r['faithful']), lambda r: has_f(r)),
        'correct_characteristic': (lambda r: r['content'] == 'characteristic',
                                   lambda r: bool(r['correct']), lambda r: has_c(r)),
        'faithful_result': (lambda r: r['content'] == 'result', lambda r: bool(r['faithful']),
                            lambda r: has_f(r)),
        'correct_result': (lambda r: r['content'] == 'result', lambda r: bool(r['correct']),
                           lambda r: has_c(r)),
        'locatable': (lambda r: True, lambda r: r['quote'], lambda r: r['value']),
        'correct_abstention_rate': (lambda r: r['absence'],
                                    lambda r: r['abstention'] == 'correct_abstention',
                                    lambda r: r['abstention'] != 'missing_from_response'),
        'missed_value_rate': (lambda r: not r['absence'],
                              lambda r: r['abstention'] == 'missed_value',
                              lambda r: r['abstention'] != 'missing_from_response'),
        'correct_given_unfaithful': (lambda r: has_c(r) and r['faithful'] is False,
                                     lambda r: bool(r['correct']), lambda r: 1),
        'correct_given_faithful': (lambda r: has_c(r) and r['faithful'] is True,
                                   lambda r: bool(r['correct']), lambda r: 1),
    }


def per_model(rows, rng):
    out = {}
    for model, rs in rows.items():
        out[model] = {}
        for split in SPLITS:
            sub = [r for r in rs if r['split'] == split]
            res = {}
            for name, (keep, num, den) in metric_defs().items():
                res[name] = ratio_ci([r for r in sub if keep(r)], num, den, rng)
            both = [r for r in sub if r['faithful'] is not None and r['correct'] is not None]
            res['two_by_two'] = dict(collections.Counter(
                ('faithful' if r['faithful'] else 'unfaithful') + '_' +
                ('correct' if r['correct'] else 'incorrect') for r in both))
            out[model][split] = res
    return out


# --------------------------------------------------------------------------- #
# paired differences between models, and Kendall tau
# --------------------------------------------------------------------------- #
def paired_diffs(rows, rng, split='test-2026'):
    out = {}
    for axis, w_key in (('faithful', 'faithful'), ('correct', 'correct')):
        table = {m: {r['item']: r for r in rs if r['split'] == split and r[w_key] is not None}
                 for m, rs in rows.items()}
        out[axis] = {}
        for a, b in itertools.combinations(MODELS, 2):
            common = table[a].keys() & table[b].keys()
            by_review = collections.defaultdict(lambda: [0.0, 0.0, 0.0])
            for i in sorted(common):     # sorted: set order depends on the hash seed (fixed 2026-10-19)
                ra, rb = table[a][i], table[b][i]
                cell = by_review[ra['review']]
                cell[0] += ra['weight'] * bool(ra[w_key])
                cell[1] += rb['weight'] * bool(rb[w_key])
                cell[2] += ra['weight']
            sums = np.array(list(by_review.values()))
            if not len(sums) or not sums[:, 2].sum():
                continue
            point = (sums[:, 0].sum() - sums[:, 1].sum()) / sums[:, 2].sum()
            idx = rng.integers(0, len(sums), size=(N_BOOT, len(sums)))
            boot = (sums[idx, 0].sum(1) - sums[idx, 1].sum(1)) / sums[idx, 2].sum(1)
            lo, hi = np.percentile(boot, [2.5, 97.5])
            # two-sided bootstrap p-value, from the same replicates (no extra draws)
            p = min(1.0, 2 * min((boot <= 0).mean(), (boot >= 0).mean()))
            out[axis][f'{a} - {b}'] = {'diff': round(float(point), 4),
                                       'ci95': [round(float(lo), 4), round(float(hi), 4)],
                                       'n': len(common), 'p_boot': round(float(p), 4)}
        # Holm over the model pairs, per axis (agreed 2026-10-10, decision C2; added 2026-10-13)
        keys = list(out[axis])
        for k, p_adj in zip(keys, holm(np.array([out[axis][k]['p_boot'] for k in keys]))):
            out[axis][k]['p_holm'] = round(float(p_adj), 4)
            out[axis][k]['significant_0.05'] = bool(p_adj < 0.05)
    return out


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


def kendall_tau(x, y):
    """Kendall tau-b for two equal-length score lists."""
    n, conc, disc, tx, ty = len(x), 0, 0, 0, 0
    for i, j in itertools.combinations(range(n), 2):
        dx, dy = np.sign(x[i] - x[j]), np.sign(y[i] - y[j])
        if dx == 0 and dy == 0:
            continue
        if dx == 0:
            tx += 1
        elif dy == 0:
            ty += 1
        elif dx == dy:
            conc += 1
        else:
            disc += 1
    denom = np.sqrt((conc + disc + tx) * (conc + disc + ty))
    return (conc - disc) / denom if denom else 0.0


def tau_axes(rows, rng, split='test-2026'):
    """Faithfulness vs correctness rankings over the models, with a review bootstrap."""
    models = list(MODELS)
    sub = {m: [r for r in rows[m] if r['split'] == split] for m in models}
    reviews = sorted({r['review'] for m in models for r in sub[m]})
    ridx = {rv: i for i, rv in enumerate(reviews)}
    # per model, per review: weighted faithful num/den, correct num/den
    arr = np.zeros((len(models), len(reviews), 4))
    for mi, m in enumerate(models):
        for r in sub[m]:
            k = ridx[r['review']]
            if r['faithful'] is not None:
                arr[mi, k, 0] += r['weight'] * bool(r['faithful'])
                arr[mi, k, 1] += r['weight']
            if r['correct'] is not None:
                arr[mi, k, 2] += r['weight'] * bool(r['correct'])
                arr[mi, k, 3] += r['weight']

    def taus(a):
        f = a[:, :, 0].sum(1) / a[:, :, 1].sum(1)
        c = a[:, :, 2].sum(1) / a[:, :, 3].sum(1)
        return kendall_tau(list(f), list(c))
    point = taus(arr)
    boots = []
    for _ in range(N_BOOT):
        pick = rng.integers(0, len(reviews), size=len(reviews))
        boots.append(taus(arr[:, pick, :]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {'tau_b': round(float(point), 4), 'ci95': [round(float(lo), 4), round(float(hi), 4)],
            'models': len(models)}


def ratio_arrays(sub, models, reviews, metrics):
    """per model x review x metric: [num, den] for ratio metrics given as (num, den) functions."""
    ridx = {rv: i for i, rv in enumerate(reviews)}
    arr = np.zeros((len(models), len(reviews), len(metrics), 2))
    for mi, m in enumerate(models):
        for r in sub[m]:
            k = ridx[r['review']]
            for j, (num, den) in enumerate(metrics):
                d = den(r)
                if d:
                    arr[mi, k, j, 0] += num(r)
                    arr[mi, k, j, 1] += d
    return arr


def locatability_check(rows, rng, split='test-2026'):
    """Is the faithfulness-vs-correctness divergence a locatability effect? (2026-10-19)
    faithfulness is also computed over answers whose quote IS found verbatim only (where the two
    judges agree), and each pair of model rankings is compared with Kendall tau-b and a review
    bootstrap. Its own random stream, so the other results are unchanged."""
    models = list(MODELS)
    sub = {m: [r for r in rows[m] if r['split'] == split] for m in models}
    reviews = sorted({r['review'] for m in models for r in sub[m]})
    metrics = {
        'faithful': (lambda r: r['weight'] * bool(r['faithful']),
                     lambda r: r['weight'] if r['faithful'] is not None else 0),
        'faithful_given_located': (lambda r: r['weight'] * bool(r['faithful']),
                                   lambda r: r['weight'] if r['faithful'] is not None and r['quote'] else 0),
        'correct': (lambda r: r['weight'] * bool(r['correct']),
                    lambda r: r['weight'] if r['correct'] is not None else 0),
        'locatable': (lambda r: r['quote'], lambda r: r['value']),
    }
    names = list(metrics)
    arr = ratio_arrays(sub, models, reviews, list(metrics.values()))

    def rates(a):
        return a[:, :, :, 0].sum(1) / a[:, :, :, 1].sum(1)     # models x metrics
    pairs = [('faithful', 'correct'), ('faithful_given_located', 'correct'),
             ('faithful', 'locatable'), ('locatable', 'correct'), ('faithful_given_located', 'locatable')]

    def taus(a):
        r = rates(a)
        return [kendall_tau(list(r[:, names.index(x)]), list(r[:, names.index(y)])) for x, y in pairs]
    point_rates, point_taus = rates(arr), taus(arr)
    boot_rates, boot_taus = [], []
    for _ in range(N_BOOT):
        pick = rng.integers(0, len(reviews), size=len(reviews))
        boot_rates.append(rates(arr[:, pick]))
        boot_taus.append(taus(arr[:, pick]))
    boot_rates, boot_taus = np.array(boot_rates), np.array(boot_taus)
    out = {'per_model': {}, 'kendall_tau': {}, 'split': split, 'models': len(models)}
    for mi, m in enumerate(models):
        out['per_model'][m] = {}
        for j, n in enumerate(names):
            lo, hi = np.percentile(boot_rates[:, mi, j], [2.5, 97.5])
            out['per_model'][m][n] = {'rate': round(float(point_rates[mi, j]), 4),
                                      'ci95': [round(float(lo), 4), round(float(hi), 4)]}
    for i, (x, y) in enumerate(pairs):
        lo, hi = np.percentile(boot_taus[:, i], [2.5, 97.5])
        out['kendall_tau'][f'{x} vs {y}'] = {'tau_b': round(float(point_taus[i]), 4),
                                             'ci95': [round(float(lo), 4), round(float(hi), 4)]}
    return out


def answerable_check(rows, rng, split='test-2026'):
    """Correctness on items whose gold the automatic search finds in the paper (exact phrase, all
    numbers or most content words; release/v2/quality/answerability_check.csv) and on the rest,
    so model error can be told apart from gold-not-in-paper (2026-10-19). Own random stream."""
    with open(os.path.join(BASE, 'release', 'v2', 'quality', 'answerability_check.csv'), encoding='utf-8') as f:
        found = {r['item_id']: r['level'] != 'none' for r in csv.DictReader(f)}
    models = list(MODELS)
    sub = {m: [r for r in rows[m] if r['split'] == split and r['correct'] is not None and r['weight']]
           for m in models}
    missing = sum(r['item'] not in found for m in models for r in sub[m])
    reviews = sorted({r['review'] for m in models for r in sub[m]})
    metrics = [(lambda r: r['weight'] * bool(r['correct']), lambda r: r['weight']),
               (lambda r: r['weight'] * bool(r['correct']), lambda r: r['weight'] if found.get(r['item']) else 0),
               (lambda r: r['weight'] * bool(r['correct']), lambda r: r['weight'] if found.get(r['item']) is False else 0),
               (lambda r: r['weight'] * bool(found.get(r['item'])), lambda r: r['weight'])]
    names = ['correct', 'correct_gold_found', 'correct_gold_not_found', 'share_gold_found']
    arr = ratio_arrays(sub, models, reviews, metrics)

    def rates(a):
        return a[:, :, :, 0].sum(1) / a[:, :, :, 1].sum(1)
    point = rates(arr)
    boots = np.array([rates(arr[:, rng.integers(0, len(reviews), size=len(reviews))]) for _ in range(N_BOOT)])
    out = {'split': split, 'items_without_answerability_level': missing, 'per_model': {}}
    for mi, m in enumerate(models):
        out['per_model'][m] = {}
        for j, n in enumerate(names):
            lo, hi = np.percentile(boots[:, mi, j], [2.5, 97.5])
            out['per_model'][m][n] = {'rate': round(float(point[mi, j]), 4),
                                      'ci95': [round(float(lo), 4), round(float(hi), 4)]}
    return out


def tau_judges():
    if not os.path.exists(CROSSJUDGE):
        return None
    with open(CROSSJUDGE, encoding='utf-8') as f:
        by_model = json.load(f)['by_model']
    out = {}
    for axis in ('F', 'C'):
        ms = sorted(by_model)
        out[axis] = {'tau_b': round(float(kendall_tau([by_model[m][axis]['gemini'] for m in ms],
                                                      [by_model[m][axis]['sonnet'] for m in ms])), 4),
                     'models': ms, 'note': 'n = 4 models: descriptive only'}
    return out


# --------------------------------------------------------------------------- #
# month check and test-2025 contrast
# --------------------------------------------------------------------------- #
def month_check(rows, rng):
    """(Jan - Apr..Jun) for rule (c) models minus the same for the others, test-2026."""
    def period(r):
        return 'jan' if r['month'] == '2026-01' else 'aprjun' if r['month'] >= '2026-04' else None
    out = {}
    for axis in ('correct', 'faithful'):
        # per review: weighted [num, den] for each (group, period); a review has one period
        cells = collections.defaultdict(lambda: np.zeros(8))
        for m, rs in rows.items():
            g = 0 if MODELS[m][1] == 'c' else 1
            for r in rs:
                p = period(r) if r['split'] == 'test-2026' else None
                if p is None or r[axis] is None:
                    continue
                k = g * 4 + (0 if p == 'jan' else 2)
                cells[r['review']][k] += r['weight'] * bool(r[axis])
                cells[r['review']][k + 1] += r['weight']
        sums = np.array(list(cells.values()))

        def did(s):
            t = s.sum(0)
            rate = [t[i] / t[i + 1] if t[i + 1] else np.nan for i in (0, 2, 4, 6)]
            return (rate[0] - rate[1]) - (rate[2] - rate[3]), rate
        point, rates = did(sums)
        idx = rng.integers(0, len(sums), size=(N_BOOT, len(sums)))
        boot = [did(sums[i])[0] for i in idx]
        lo, hi = np.nanpercentile(boot, [2.5, 97.5])
        out[axis] = {'rule_c_jan': round(float(rates[0]), 4), 'rule_c_aprjun': round(float(rates[1]), 4),
                     'other_jan': round(float(rates[2]), 4), 'other_aprjun': round(float(rates[3]), 4),
                     'did': round(float(point), 4), 'ci95': [round(float(lo), 4), round(float(hi), 4)]}
    out['note'] = ('positive did on correctness = rule (c) models do relatively better on the '
                   'earliest 2026 reviews: a leakage signal')
    return out


def pre_contrast(rows, rng):
    """test-2025 minus test-2026, per model and pooled over models; the two splits hold
    different reviews, so each is resampled on its own (independent cluster bootstrap)."""
    def sums_by_review(rs, axis, split):
        by = collections.defaultdict(lambda: [0.0, 0.0])
        for r in rs:
            if r['split'] == split and r[axis] is not None:
                by[r['review']][0] += r['weight'] * bool(r[axis])
                by[r['review']][1] += r['weight']
        return np.array(list(by.values()))

    def diff_ci(pre, new):
        point = pre[:, 0].sum() / pre[:, 1].sum() - new[:, 0].sum() / new[:, 1].sum()
        ip = rng.integers(0, len(pre), size=(N_BOOT, len(pre)))
        inew = rng.integers(0, len(new), size=(N_BOOT, len(new)))
        boot = pre[ip, 0].sum(1) / pre[ip, 1].sum(1) - new[inew, 0].sum(1) / new[inew, 1].sum(1)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        return {'diff': round(float(point), 4), 'ci95': [round(float(lo), 4), round(float(hi), 4)]}
    out = {}
    for axis in ('correct', 'faithful'):
        out[axis] = {m: diff_ci(sums_by_review(rows[m], axis, 'test-2025'),
                                sums_by_review(rows[m], axis, 'test-2026')) for m in MODELS}
        everyone = [r for m in MODELS for r in rows[m]]
        out[axis]['all_models'] = diff_ci(sums_by_review(everyone, axis, 'test-2025'),
                                          sums_by_review(everyone, axis, 'test-2026'))
    out['note'] = ('test-2025 reviews went online Oct-Dec 2025 (a bounded contrast); the splits '
                   'also differ in field mix, so a difference is not by itself contamination')
    return out


ZONE_ORDER = ['abstract', 'methods', 'results', 'tables', 'content', 'background', 'discussion', 'other']


def zone_map(rows, rng, split='test-2026'):
    """Where answers come from: the zone of the section each located quote sits in.

    shares: per model and all models, share of located answers by zone, overall and by
    content type (unweighted: every located answer counts once). quality: pooled over models,
    weighted faithful / correct rate of the answers quoted from each zone, with CIs.
    """
    out = {'shares': {}, 'shares_by_content': {}, 'quality_by_zone': {}}
    pools = {m: [r for r in rows[m] if r['split'] == split and r['zone']] for m in MODELS}
    pools['all_models'] = [r for m in MODELS for r in pools[m]]
    for name, rs in pools.items():
        c = collections.Counter(r['zone'] for r in rs)
        out['shares'][name] = {z: round(c[z] / len(rs), 4) for z in ZONE_ORDER if c[z]} | {'n': len(rs)}
    for content in ('characteristic', 'result'):
        rs = [r for r in pools['all_models'] if r['content'] == content]
        c = collections.Counter(r['zone'] for r in rs)
        out['shares_by_content'][content] = {z: round(c[z] / len(rs), 4) for z in ZONE_ORDER if c[z]} | {'n': len(rs)}
    for z in ZONE_ORDER:
        rs = [r for r in pools['all_models'] if r['zone'] == z]
        if not rs:
            continue
        out['quality_by_zone'][z] = {
            'faithful': ratio_ci(rs, lambda r: r['weight'] * bool(r['faithful']),
                                 lambda r: r['weight'] if r['faithful'] is not None else 0, rng),
            'correct': ratio_ci(rs, lambda r: r['weight'] * bool(r['correct']),
                                lambda r: r['weight'] if r['correct'] is not None else 0, rng)}
    out['note'] = ('zones from the reader\'s section titles (jats_source.classify_section); '
                   '"content" = substantive sections without IMRaD titles; layout = document order')
    return out


def main():
    rng = np.random.default_rng(SEED)
    rows = load()
    results = per_model(rows, rng)
    report = {
        'note': 'cluster bootstrap over reviews, %d replicates, percentile 95%% intervals' % N_BOOT,
        'models': {m: {'name': n, 'rule': rule} for m, (n, rule) in MODELS.items()},
        'per_model': results,
        'paired_differences_test2026': paired_diffs(rows, rng),
        'kendall_tau_faithful_vs_correct': tau_axes(rows, rng),
        'kendall_tau_gemini_vs_sonnet': tau_judges(),
        'month_check': month_check(rows, rng),
        'pre_minus_2026': pre_contrast(rows, rng),
        'zone_map_test2026': zone_map(rows, rng),
        'locatability_check_test2026': locatability_check(rows, np.random.default_rng(SEED + 1)),
        'answerable_check_test2026': answerable_check(rows, np.random.default_rng(SEED + 2)),
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(report, f, indent=1)

    def fmt(x):
        return f"{x['rate']:.1%} [{x['ci95'][0]:.1%}, {x['ci95'][1]:.1%}]" if x else '-'
    print('test-2026 (weighted pooled; 95% CI, review cluster bootstrap)')
    for m, (name, rule) in MODELS.items():
        r = results[m]['test-2026']
        print(f"  {name:<18} ({rule}) faithful {fmt(r['faithful_pooled'])}  correct {fmt(r['correct_pooled'])}  "
              f"locatable {fmt(r['locatable'])}")
    print('kendall tau faithful vs correct:', report['kendall_tau_faithful_vs_correct'])
    print('kendall tau gemini vs sonnet:', report['kendall_tau_gemini_vs_sonnet'])
    print('month check:', json.dumps(report['month_check']))
    print(f'-> {os.path.relpath(OUT, BASE)}')


if __name__ == '__main__':
    main()
