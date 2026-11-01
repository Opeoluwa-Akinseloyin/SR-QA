"""
21_eval_sample.py
-----------------
Stage 5.2 of the v2 pipeline: draw the evaluation sample from the frozen release.
Every model answers and is judged on exactly these items.

Design (agreed 2026-10-08, two-phase, weighted):

  1. Answering sample = a random share of each test arm's SOURCE PAPERS
     (test-2026 30%, test-2025 20%), drawn across all reviews. Every question
     about a sampled paper is answered, from every review that extracted it.
     Answering is one call per paper, so whole papers cost little more than
     single questions; sampling papers is cluster sampling, so every question
     keeps the same inclusion probability within an arm.
  2. Judging sample = every `result` item + a simple random 36% of the
     `characteristic` items in the answering sample (exact count, no
     replacement). Judging is per answer and dominates cost; result items are
     the scarce type, and taking all of them gives enough to compare the two.
     Each judged characteristic item carries weight n_char / n_char_judged
     (≈ 2.8) in pooled scores; per-content scores need no weight.

The sample is a property of the benchmark, not of any model's output. Seeds are
fixed; the output records the SHA-256 of the input files and of itself, so the
paper can quote it and any later run can check it is scoring the same sample.

Inputs:  release/v2/data/test-2026.jsonl.gz, release/v2/data/test-2025.jsonl.gz
Outputs: data/v2/eval/eval_sample.jsonl   one row per sampled item: item_id, split,
                                          review_pmcid, source_pmcid, review_online_date,
                                          content, judge (bool), weight (judged items;
                                          0 for answered-only items)
         data/v2/eval/eval_sample_summary.json   design, seeds, counts, checksums

Layout test (agreed 2026-10-09): `--layout-test` instead draws 100 random
dev-2026 papers, every item answered and judged, into
data/v2/eval/layout_test_sample.jsonl (same row format).

Usage:
  python scripts/21_eval_sample.py                # agreed design and seeds
  python scripts/21_eval_sample.py --dry-run      # print counts, write nothing
  python scripts/21_eval_sample.py --layout-test  # the layout test's dev sample
"""
import argparse
import collections
import gzip
import hashlib
import json
import os
import random

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE, 'release', 'v2', 'data')
OUT_DIR = os.path.join(BASE, 'data', 'v2', 'eval')
SAMPLE_OUT = os.path.join(OUT_DIR, 'eval_sample.jsonl')
SUMMARY_OUT = os.path.join(OUT_DIR, 'eval_sample_summary.json')

LAYOUT_TEST_OUT = os.path.join(OUT_DIR, 'layout_test_sample.jsonl')
LAYOUT_TEST_PAPERS = 100
LAYOUT_TEST_SEED = 20261010

PAPER_SHARE = {'test-2026': 0.30, 'test-2025': 0.20}
JUDGE_SHARE_CHARACTERISTIC = 0.36


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def load_arm(split):
    path = os.path.join(DATA_DIR, split + '.jsonl.gz')
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        return [json.loads(line) for line in f], path


def draw(items, paper_share, judge_share, rng_papers, rng_judge):
    """Return the sampled items as output rows, plus the per-arm weight."""
    papers = sorted({it['source_pmcid'] for it in items})
    picked = set(rng_papers.sample(papers, round(paper_share * len(papers))))
    sampled = [it for it in items if it['source_pmcid'] in picked]
    sampled.sort(key=lambda it: it['item_id'])

    char_ids = [it['item_id'] for it in sampled if it['content'] == 'characteristic']
    n_char_judged = round(judge_share * len(char_ids))
    judged_char = set(rng_judge.sample(char_ids, n_char_judged))
    weight_char = len(char_ids) / n_char_judged

    rows = []
    for it in sampled:
        if it['content'] == 'result':
            judge, weight = True, 1.0
        elif it['content'] == 'characteristic':
            judge = it['item_id'] in judged_char
            weight = weight_char if judge else 0.0
        else:
            raise SystemExit(f"unknown content {it['content']!r} on {it['item_id']}")
        rows.append({'item_id': it['item_id'], 'split': it['split'],
                     'review_pmcid': it['review_pmcid'], 'source_pmcid': it['source_pmcid'],
                     'review_online_date': it['review_online_date'],
                     'content': it['content'], 'judge': judge, 'weight': weight})
    return rows, len(papers), len(picked), weight_char


def counts(rows):
    by = collections.Counter()
    for r in rows:
        by[(r['content'], 'answered')] += 1
        if r['judge']:
            by[(r['content'], 'judged')] += 1
    return by


def layout_test(dry_run):
    """The layout test's sample (agreed 2026-10-09): 100 random dev-2026 source papers.

    Dev, not test, because the test chooses a method. Every item is answered and
    judged (judge=True, weight 1), so the two layouts are compared on identical,
    fully judged questions.
    """
    items, path = load_arm('dev-2026')
    papers = sorted({it['source_pmcid'] for it in items})
    picked = set(random.Random(f'{LAYOUT_TEST_SEED}:dev-2026').sample(papers, LAYOUT_TEST_PAPERS))
    rows = [{'item_id': it['item_id'], 'split': it['split'],
             'review_pmcid': it['review_pmcid'], 'source_pmcid': it['source_pmcid'],
             'review_online_date': it['review_online_date'],
             'content': it['content'], 'judge': True, 'weight': 1.0}
            for it in sorted(items, key=lambda it: it['item_id'])
            if it['source_pmcid'] in picked]
    content = collections.Counter(r['content'] for r in rows)
    print(f'layout test: {len(picked)}/{len(papers)} dev-2026 papers, '
          f"{len({r['review_pmcid'] for r in rows})} reviews, {len(rows)} items "
          f"({content['characteristic']} characteristic, {content['result']} result)")
    if dry_run:
        return
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(LAYOUT_TEST_OUT, 'w', encoding='utf-8', newline='\n') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f'wrote {os.path.relpath(LAYOUT_TEST_OUT, BASE)} '
          f'(sha256 {sha256(LAYOUT_TEST_OUT)[:16]}...; input {sha256(path)[:16]}...)')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--seed-papers', type=int, default=20261008)
    ap.add_argument('--seed-judge', type=int, default=20261009)
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--layout-test', action='store_true',
                    help='draw the layout-test sample instead (see layout_test)')
    args = ap.parse_args()
    if args.layout_test:
        layout_test(args.dry_run)
        return

    all_rows, summary_arms, inputs = [], {}, {}
    for split, share in PAPER_SHARE.items():
        items, path = load_arm(split)
        inputs[os.path.relpath(path, BASE).replace(os.sep, '/')] = sha256(path)
        # one generator per arm, seeded from the arm name, so changing one arm's share
        # never changes the other arm's draw
        rng_p = random.Random(f'{args.seed_papers}:{split}')
        rng_j = random.Random(f'{args.seed_judge}:{split}')
        rows, n_papers, n_picked, w = draw(items, share, JUDGE_SHARE_CHARACTERISTIC, rng_p, rng_j)
        c = counts(rows)
        months = collections.Counter(r['review_online_date'][:7] for r in rows)
        summary_arms[split] = {
            'papers_total': n_papers, 'papers_sampled': n_picked, 'paper_share': share,
            'items_total': len(items), 'items_answered': len(rows),
            'reviews_sampled': len({r['review_pmcid'] for r in rows}),
            'characteristic_answered': c[('characteristic', 'answered')],
            'characteristic_judged': c[('characteristic', 'judged')],
            'result_answered': c[('result', 'answered')],
            'result_judged': c[('result', 'judged')],
            'judged_total': c[('characteristic', 'judged')] + c[('result', 'judged')],
            'characteristic_weight': w,
            'items_by_online_month': dict(sorted(months.items())),
        }
        all_rows.extend(rows)

    for split, s in summary_arms.items():
        print(f"{split}: {s['papers_sampled']}/{s['papers_total']} papers, "
              f"{s['reviews_sampled']} reviews, {s['items_answered']}/{s['items_total']} items answered")
        print(f"  characteristic {s['characteristic_answered']} answered, "
              f"{s['characteristic_judged']} judged (weight {s['characteristic_weight']:.4f})")
        print(f"  result         {s['result_answered']} answered, {s['result_judged']} judged")
        print(f"  judged total   {s['judged_total']}")
        print('  items by review online month:',
              ', '.join(f'{m} {n}' for m, n in s['items_by_online_month'].items()))
    if args.dry_run:
        return

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(SAMPLE_OUT, 'w', encoding='utf-8', newline='\n') as f:
        for r in all_rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    summary = {
        'design': 'two-phase: papers sampled per arm; all result + a random share of '
                  'characteristic answers judged; weight = n_char / n_char_judged',
        'paper_share': PAPER_SHARE,
        'judge_share_characteristic': JUDGE_SHARE_CHARACTERISTIC,
        'seed_papers': args.seed_papers, 'seed_judge': args.seed_judge,
        'arms': summary_arms,
        'inputs_sha256': inputs,
        'sample_sha256': sha256(SAMPLE_OUT),
    }
    with open(SUMMARY_OUT, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(summary, f, indent=2)
    print(f'wrote {os.path.relpath(SAMPLE_OUT, BASE)} ({len(all_rows)} rows) and '
          f'{os.path.relpath(SUMMARY_OUT, BASE)}')


if __name__ == '__main__':
    main()
