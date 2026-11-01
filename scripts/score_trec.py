"""
score_trec.py
-------------
Score a retrieval run against SR-QA relevance labels, both in TREC format. Standard
library only, so it works without the rest of the pipeline.

  qrels  "<query_id> 0 <passage_id> <grade>"   grade 2 = phrase match, 1 = numbers match
  run    "<query_id> Q0 <passage_id> <rank> <score> <tag>"   (ordered by score, high first)

Metrics (the definitions used in the paper, as in scripts/eval/retrieval.py):
  Success@5, Success@10  1 if at least one relevant passage is in the top k
  MRR                    reciprocal rank of the first relevant passage
  nDCG@10                graded, gain 2^grade - 1
Queries in the qrels that are missing from the run score 0.

Usage:
  python scripts/score_trec.py data/v2/retrieval_labels/test-2026/qrels.trec \
                               data/v2/eval/retrieval_full/runs/bm25.trec
"""
import collections
import math
import sys


def read_qrels(path):
    qrels = collections.defaultdict(dict)
    with open(path, encoding='utf-8') as f:
        for line in f:
            if line.strip():
                qid, _, pid, grade = line.split()
                if int(grade) > 0:
                    qrels[qid][pid] = int(grade)
    return qrels


def read_run(path):
    rows = collections.defaultdict(list)
    with open(path, encoding='utf-8') as f:
        for line in f:
            if line.strip():
                qid, _, pid, _, score, _ = line.split()
                rows[qid].append((float(score), pid))
    return {qid: [pid for _, pid in sorted(r, key=lambda x: -x[0])] for qid, r in rows.items()}


def metrics(ranking, qrel):
    hits = [qrel.get(pid, 0) for pid in ranking]
    first = next((i for i, g in enumerate(hits, 1) if g), None)
    dcg = sum((2 ** g - 1) / math.log2(i + 1) for i, g in enumerate(hits[:10], 1))
    ideal = sorted(qrel.values(), reverse=True)[:10]
    idcg = sum((2 ** g - 1) / math.log2(i + 1) for i, g in enumerate(ideal, 1))
    return {'Success@5': float(any(hits[:5])), 'Success@10': float(any(hits[:10])),
            'MRR': 1.0 / first if first else 0.0, 'nDCG@10': dcg / idcg if idcg else 0.0}


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    qrels, run = read_qrels(sys.argv[1]), read_run(sys.argv[2])
    scores = [metrics(run.get(qid, []), q) for qid, q in qrels.items()]
    print(f'queries: {len(scores):,}')
    for k in scores[0]:
        print(f'{k:<11} {sum(s[k] for s in scores) / len(scores):.4f}')


if __name__ == '__main__':
    main()
