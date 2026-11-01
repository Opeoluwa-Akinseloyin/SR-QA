"""
retrieval.py
------------
Stage 5.R of the v2 pipeline (agreed 2026-10-10): the evidence-retrieval baseline.
Task: given a question and ITS source paper, rank that paper's passages so the
one holding the answer comes first. No LLM, no judge - scored against
relevance labels made by string matching.

  passages  each paper (document order, the layout the answering runs use) split
            into: one passage per prose sentence (>= 15 chars), one per table
            cell line "[row=... | header = value]", one per table DATA ROW of the
            Markdown grid (added 2026-10-10: cell lines exist only for cells with a
            digit), and table captions / footnotes (otherwise v1's
            old/scripts/rag/chunk_papers.py rules)
  labels    a passage is relevant to an item if it holds a fragment of the gold
            (fragments split on " | "):
              grade 2  the normalised fragment occurs in the passage as a whole
                       word / phrase (so "2" does not match inside "2020")
              grade 1  every number of the fragment occurs in the passage, counting
                       only stand-alone numbers (not the 6 in G6PD) and only when
                       they are distinctive (>= 2 numbers, or one with a decimal
                       point or >= 3 digits)
            (the matching of 18_answerability_check.py, at passage scale and
            tightened 2026-10-10; its "80% of words" level is not used - too loose
            for single passages). Items whose gold is "not reported", has no
            letter or digit (review notation such as a tick or arrow; v2.0.1), is
            only yes/no (2026-10-15), has
            no relevant passage, or has > 10 relevant passages (non-specific gold)
            are left out and counted.
  queries   the question text; the sample's test-2026 items (eval_sample.jsonl)
  metrics   Recall@5, Recall@10 (>= 1 relevant passage in the top k), MRR (first
            relevant), nDCG@10 (graded); overall, by content type, and by where
            the best evidence sits (table cell or row vs text)

Commands:
  build     -> data/v2/eval/retrieval/{passages.jsonl.gz, queries.jsonl, qrels.jsonl}
  label-all -> data/v2/retrieval_labels/<split>/{passages.jsonl.gz, queries.jsonl, qrels.jsonl,
               qrels.trec, excluded.jsonl, stats.json} for EVERY split (the release labels;
               same rules as build; each qrel keeps grade, match kind and passage type)
  bm25      -> runs/bm25.jsonl   BM25 (bm25s, Lucene variant, k1=1.5, b=0.75, English
                                 stopwords), one index per paper
  dense     -> runs/<model>.jsonl  dense retrieval, --model medcpt | bge-m3 (fp32, laptop
                                 GPU); --benchmark N times N passages first
  hybrid    -> runs/hybrid_<a>+<b>.jsonl  reciprocal-rank fusion (k=60) of two runs
  rerank    -> runs/<run>+rerank.jsonl  bge-reranker-v2-m3 cross-encoder on the top --depth (50)
                                 of a first-stage run; --benchmark N times N pairs first
  score     a run file -> metrics (printed, and saved next to the run)

Usage:
  python scripts/eval/retrieval.py build
  python scripts/eval/retrieval.py bm25
  python scripts/eval/retrieval.py score --run bm25
"""
import argparse
import collections
import gzip
import importlib.util
import json
import math
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(os.path.dirname(HERE))
SOURCE_DIR = os.path.join(BASE, 'SOURCE_XML')
DATA_DIR = os.path.join(BASE, 'release', 'v2', 'data')
SAMPLE = os.path.join(BASE, 'data', 'v2', 'eval', 'eval_sample.jsonl')
OUT = os.path.join(BASE, 'data', 'v2', 'eval', 'retrieval')
RUNS = os.path.join(OUT, 'runs')
SPLIT = 'test-2026'
MAX_SOURCE_CHARS = 200_000
TOP_K = 100            # passages kept per query in a run
RRF_K = 60


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


jats_source = _load('jats_source', os.path.join(HERE, 'jats_source.py'))
answerability = _load('answerability', os.path.join(BASE, 'scripts', '18_answerability_check.py'))

try:
    sys.stdout.reconfigure(encoding='utf-8')
except (AttributeError, ValueError):
    pass

# --------------------------------------------------------------------------- #
# passages (v1's chunk_papers rules)
# --------------------------------------------------------------------------- #
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\"'])")


def split_sentences(text):
    out, start = [], 0
    for m in SENT_SPLIT.finditer(text):
        seg = text[start:m.start()].strip()
        if seg:
            out.append(seg)
        start = m.end()
    tail = text[start:].strip()
    if tail:
        out.append(tail)
    return out


def chunk_paper(pmcid):
    src = jats_source.parse_source(os.path.join(SOURCE_DIR, pmcid + '.xml'),
                                   MAX_SOURCE_CHARS, layout='linear')
    passages = []
    for s in src['sections']:
        text = s.get('text') or ''
        is_table = s.get('zone') == 'tables' and 'caption' not in (s.get('title') or '').lower()
        if is_table:
            lines = [ln.strip() for ln in text.split('\n')]
            for i, line in enumerate(lines):
                following = lines[i + 1] if i + 1 < len(lines) else ''
                if line.startswith('[row='):
                    passages.append(('cell', s['id'], line))
                elif line.startswith('|'):
                    # Markdown grid: keep each DATA row as a passage (added 2026-10-10:
                    # cell lines exist only for cells with a digit, so v1 lost non-numeric
                    # and multi-value table answers); skip the header and separator lines
                    if line.startswith('|---') or following.startswith('|---'):
                        continue
                    passages.append(('row', s['id'], line))
                elif len(line) >= 8:
                    passages.append(('table_meta', s['id'], line))
        else:
            for seg in split_sentences(text):
                if len(seg) >= 15:
                    passages.append(('sentence', s['id'], seg))
    return [{'pid': f'{pmcid}:{i}', 'type': t, 'section': sec, 'zone': None, 'text': txt}
            for i, (t, sec, txt) in enumerate(passages)]


# --------------------------------------------------------------------------- #
# labels
# --------------------------------------------------------------------------- #
# not after a letter or digit, nor after "letter-" (IL-6, COVID-19); not before a letter
STANDALONE_NUMBER = re.compile(r'(?<![a-z0-9])(?<![a-z]-)\d+(?:[.,]\d+)*(?![a-z])')
MAX_RELEVANT = 10      # (c) more relevant passages than this = non-specific gold
# (d) agreed 2026-10-14: a gold with no letter or digit (review notation such as a tick or an
# arrow) cannot be found in a paper by string matching, so it gets no relevance labels
HAS_ALNUM = re.compile(r'[A-Za-z0-9]')
# (e) agreed 2026-10-15 (after the human label check): a yes/no gold would be 'found' in any
# passage containing the word yes or no, so it gets no relevance labels either
YES_NO = {'yes', 'no', 'y', 'n'}


def is_yes_no(gold):
    frags = [f.strip().lower().rstrip('.') for f in gold.split(' | ') if f.strip()]
    return bool(frags) and all(f in YES_NO for f in frags)


def standalone_numbers(text):
    """(a) Numbers that stand alone - not digits inside a word such as G6PD or IL6 -
    canonicalised as in 18_answerability_check (thousands separators dropped)."""
    out = set()
    for n in STANDALONE_NUMBER.findall(text):
        if re.fullmatch(r'\d{1,3}(,\d{3})+', n):
            n = n.replace(',', '')
        out.add(n.rstrip('.,'))
    return out


def distinctive(numbers):
    """(b) Numbers specific enough to identify a passage: two or more, or one with a
    decimal point or at least three digits ('54.2', '1068', '30/31'; not '0' or '6')."""
    return len(numbers) >= 2 or any('.' in n or len(n.replace(',', '')) >= 3 for n in numbers)


def fragment_grade(fragment, passage_norm, passage_numbers):
    """2 = whole-phrase match, 1 = every (distinctive, stand-alone) number present, 0 = neither."""
    if not fragment:
        return 0
    if re.search(r'(?<![a-z0-9])' + re.escape(fragment) + r'(?![a-z0-9])', passage_norm):
        return 2
    numbers = standalone_numbers(fragment)
    if numbers and distinctive(numbers) and numbers <= passage_numbers:
        return 1
    return 0


def label_item(gold, passages_norm):
    """{passage_id: grade} over every passage that holds some fragment of the gold."""
    fragments = [answerability.normalise(f) for f in gold.split(answerability.SEPARATOR) if f.strip()]
    qrel = {}
    for pid, norm, numbers in passages_norm:
        grade = max((fragment_grade(f, norm, numbers) for f in fragments), default=0)
        if grade:
            qrel[pid] = grade
    return qrel


def load_test_items():
    with open(SAMPLE, encoding='utf-8') as f:
        wanted = {r['item_id'] for r in map(json.loads, f) if r['split'] == SPLIT}
    items = []
    with gzip.open(os.path.join(DATA_DIR, SPLIT + '.jsonl.gz'), 'rt', encoding='utf-8') as f:
        for line in f:
            it = json.loads(line)
            if it['item_id'] in wanted:
                items.append(it)
    return items


def cmd_build(args):
    items = load_test_items()
    by_paper = collections.defaultdict(list)
    for it in items:
        by_paper[it['source_pmcid']].append(it)
    os.makedirs(OUT, exist_ok=True)
    funnel = collections.Counter(items=len(items))
    n_passages = collections.Counter()
    with gzip.open(os.path.join(OUT, 'passages.jsonl.gz'), 'wt', encoding='utf-8', newline='\n') as fp, \
            open(os.path.join(OUT, 'queries.jsonl'), 'w', encoding='utf-8', newline='\n') as fq, \
            open(os.path.join(OUT, 'qrels.jsonl'), 'w', encoding='utf-8', newline='\n') as fr:
        for pmcid in sorted(by_paper):
            passages = chunk_paper(pmcid)
            for p in passages:
                fp.write(json.dumps(p, ensure_ascii=False) + '\n')
                n_passages[p['type']] += 1
            norm = [(p['pid'], answerability.normalise(p['text']),
                     standalone_numbers(answerability.normalise(p['text']))) for p in passages]
            for it in sorted(by_paper[pmcid], key=lambda it: it['item_id']):
                if it['gold_is_absence']:
                    funnel['gold_not_reported'] += 1
                    continue
                if not HAS_ALNUM.search(it['gold']):
                    funnel['symbol_only_gold'] += 1
                    continue
                if is_yes_no(it['gold']):
                    funnel['yes_no_gold'] += 1
                    continue
                qrel = label_item(it['gold'], norm)
                if not qrel:
                    funnel['no_relevant_passage'] += 1
                    continue
                if len(qrel) > MAX_RELEVANT:
                    funnel['non_specific_gold'] += 1
                    continue
                funnel['kept'] += 1
                fq.write(json.dumps({'item_id': it['item_id'], 'source_pmcid': pmcid,
                                     'content': it['content'], 'query': it['question']},
                                    ensure_ascii=False) + '\n')
                fr.write(json.dumps({'item_id': it['item_id'], 'qrel': qrel}) + '\n')
    stats = {'split': SPLIT, 'papers': len(by_paper), 'passages': dict(n_passages),
             'passages_total': sum(n_passages.values()), 'funnel': dict(funnel)}
    with open(os.path.join(OUT, 'build_stats.json'), 'w', encoding='utf-8', newline='\n') as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))


# --------------------------------------------------------------------------- #
# label-all: release labels for every split (agreed 2026-10-10, brief M2 / G1 / G2)
# --------------------------------------------------------------------------- #
LABELS_OUT = os.path.join(BASE, 'data', 'v2', 'retrieval_labels')
SPLITS = ['train-2025', 'train-2026', 'dev-2026', 'test-2026', 'test-2025']


def label_paper(job):
    """Worker: chunk one paper and label its items with the same rules as `build`.
    Returns (passages, kept, excluded); kept qrels carry grade, match kind and passage type."""
    pmcid, items = job
    passages = chunk_paper(pmcid)
    types = {p['pid']: p['type'] for p in passages}
    norm = [(p['pid'], answerability.normalise(p['text']),
             standalone_numbers(answerability.normalise(p['text']))) for p in passages]
    kept, excluded = [], []
    for it in sorted(items, key=lambda it: it['item_id']):
        if it['gold_is_absence']:
            excluded.append({'item_id': it['item_id'], 'reason': 'gold_not_reported'})
            continue
        if not HAS_ALNUM.search(it['gold']):
            excluded.append({'item_id': it['item_id'], 'reason': 'symbol_only_gold'})
            continue
        if is_yes_no(it['gold']):
            excluded.append({'item_id': it['item_id'], 'reason': 'yes_no_gold'})
            continue
        qrel = label_item(it['gold'], norm)
        if not qrel:
            excluded.append({'item_id': it['item_id'], 'reason': 'no_relevant_passage'})
            continue
        if len(qrel) > MAX_RELEVANT:
            excluded.append({'item_id': it['item_id'], 'reason': 'non_specific_gold',
                             'n_matching': len(qrel)})
            continue
        kept.append({'item_id': it['item_id'], 'source_pmcid': pmcid, 'content': it['content'],
                     'query': it['question'],
                     'qrel': {pid: {'grade': g, 'match': 'phrase' if g == 2 else 'numbers',
                                    'passage_type': types[pid]} for pid, g in qrel.items()}})
    return passages, kept, excluded


def cmd_label_all(args):
    import multiprocessing
    splits = args.splits or SPLITS
    summary = {}
    for split in splits:
        by_paper = collections.defaultdict(list)
        with gzip.open(os.path.join(DATA_DIR, split + '.jsonl.gz'), 'rt', encoding='utf-8') as f:
            for line in f:
                it = json.loads(line)
                by_paper[it['source_pmcid']].append(it)
        jobs = sorted(by_paper.items())
        if args.limit:
            jobs = jobs[:args.limit]
        out = os.path.join(LABELS_OUT, split)
        os.makedirs(out, exist_ok=True)
        funnel = collections.Counter(items=sum(len(v) for _, v in jobs))
        n_passages, n_grades = collections.Counter(), collections.Counter()
        t0 = time.time()
        with multiprocessing.Pool(args.workers) as pool, \
                gzip.open(os.path.join(out, 'passages.jsonl.gz'), 'wt', encoding='utf-8', newline='\n') as fp, \
                open(os.path.join(out, 'queries.jsonl'), 'w', encoding='utf-8', newline='\n') as fq, \
                open(os.path.join(out, 'qrels.jsonl'), 'w', encoding='utf-8', newline='\n') as fr, \
                open(os.path.join(out, 'qrels.trec'), 'w', encoding='utf-8', newline='\n') as ft, \
                open(os.path.join(out, 'excluded.jsonl'), 'w', encoding='utf-8', newline='\n') as fx:
            # imap keeps paper order, so the files are deterministic
            for i, (passages, kept, excluded) in enumerate(pool.imap(label_paper, jobs, chunksize=4), 1):
                for p in passages:
                    fp.write(json.dumps(p, ensure_ascii=False) + '\n')
                    n_passages[p['type']] += 1
                for k in kept:
                    fq.write(json.dumps({key: k[key] for key in ('item_id', 'source_pmcid', 'content', 'query')},
                                        ensure_ascii=False) + '\n')
                    fr.write(json.dumps({'item_id': k['item_id'], 'qrel': k['qrel']}) + '\n')
                    for pid, q in k['qrel'].items():
                        ft.write(f"{k['item_id']} 0 {pid} {q['grade']}\n")
                        n_grades[q['grade']] += 1
                    funnel['kept'] += 1
                for x in excluded:
                    fx.write(json.dumps(x) + '\n')
                    funnel[x['reason']] += 1
                if i % 500 == 0:
                    print(f'  {split}: {i:,}/{len(jobs):,} papers, {time.time() - t0:.0f} s', flush=True)
        stats = {'split': split, 'papers': len(jobs), 'passages': dict(n_passages),
                 'passages_total': sum(n_passages.values()), 'funnel': dict(funnel),
                 'relevant_pairs_by_grade': {str(g): n for g, n in sorted(n_grades.items())},
                 'seconds': round(time.time() - t0)}
        with open(os.path.join(out, 'stats.json'), 'w', encoding='utf-8', newline='\n') as f:
            json.dump(stats, f, indent=2)
        summary[split] = stats
        print(json.dumps(stats, indent=2), flush=True)
    return summary


# --------------------------------------------------------------------------- #
# runs
# --------------------------------------------------------------------------- #
def read_jsonl(path, opener=open):
    with opener(path, 'rt', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def load_corpus():
    passages = collections.defaultdict(list)
    for p in read_jsonl(os.path.join(OUT, 'passages.jsonl.gz'), gzip.open):
        passages[p['pid'].rsplit(':', 1)[0]].append(p)
    return passages, read_jsonl(os.path.join(OUT, 'queries.jsonl'))


def write_run(name, ranked, meta):
    os.makedirs(RUNS, exist_ok=True)
    path = os.path.join(RUNS, name + '.jsonl')
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        f.write(json.dumps({'_meta': meta}) + '\n')
        for item_id, ranking in ranked.items():
            f.write(json.dumps({'item_id': item_id, 'ranking': ranking}) + '\n')
    print(f'-> {os.path.relpath(path, BASE)}  ({len(ranked):,} queries)')


def cmd_bm25(args):
    import bm25s
    passages, queries = load_corpus()
    by_paper = collections.defaultdict(list)
    for q in queries:
        by_paper[q['source_pmcid']].append(q)
    ranked = {}
    for pmcid, qs in by_paper.items():
        docs = passages[pmcid]
        retriever = bm25s.BM25(method='lucene', k1=1.5, b=0.75)
        retriever.index(bm25s.tokenize([d['text'] for d in docs], stopwords='en', show_progress=False),
                        show_progress=False)
        k = min(TOP_K, len(docs))
        hits, scores = retriever.retrieve(
            bm25s.tokenize([q['query'] for q in qs], stopwords='en', show_progress=False),
            k=k, show_progress=False)
        for q, row, srow in zip(qs, hits, scores):
            ranked[q['item_id']] = [[docs[int(i)]['pid'], round(float(s), 4)] for i, s in zip(row, srow)]
    write_run('bm25', ranked, {'method': 'BM25 (bm25s, lucene)', 'k1': 1.5, 'b': 0.75,
                               'stopwords': 'en', 'index': 'one per paper', 'top_k': TOP_K})


# Dense retrievers (agreed 2026-10-10, run on the laptop GPU). Each passage and query is
# encoded once; ranking is a dot product within the query's own paper.
DENSE = {
    # NCBI's MedCPT: separate query / article encoders, [CLS] vector, dot product
    'medcpt': {'query': 'ncbi/MedCPT-Query-Encoder', 'passage': 'ncbi/MedCPT-Article-Encoder',
               'query_len': 64, 'passage_len': 256, 'normalise': False, 'pair_input': True},
    # BAAI's BGE-M3, dense head: [CLS] vector, L2-normalised, cosine
    'bge-m3': {'query': 'BAAI/bge-m3', 'passage': 'BAAI/bge-m3',
               'query_len': 128, 'passage_len': 256, 'normalise': True, 'pair_input': False},
}


def encode(texts, model_id, max_len, normalise, pair_input, batch=32, label=''):
    """[CLS] embeddings (fp32; stored as fp16), batched shortest-first (less padding)."""
    import numpy as np
    import torch
    from transformers import AutoModel, AutoTokenizer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tok = AutoTokenizer.from_pretrained(model_id)
    # fp32: on the T600 fp16 measured 4x SLOWER (93 vs 380 passages/s, 2026-10-10)
    model = AutoModel.from_pretrained(model_id)
    model.to(device).eval()
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    out = np.zeros((len(texts), model.config.hidden_size), dtype=np.float16)
    t0, done = time.time(), 0
    with torch.no_grad():
        for start in range(0, len(order), batch):
            idx = order[start:start + batch]
            chunk = [texts[i] for i in idx]
            if pair_input:          # MedCPT's article encoder takes [title, text]; no title here
                enc = tok([['', t] for t in chunk], truncation=True, padding=True,
                          max_length=max_len, return_tensors='pt')
            else:
                enc = tok(chunk, truncation=True, padding=True, max_length=max_len,
                          return_tensors='pt')
            vec = model(**enc.to(device)).last_hidden_state[:, 0]
            if normalise:
                vec = torch.nn.functional.normalize(vec, dim=-1)
            out[idx] = vec.float().cpu().numpy().astype(np.float16)
            done += len(idx)
            if label and (start // batch) % 200 == 0:
                rate = done / max(time.time() - t0, 1e-6)
                print(f'  {label}: {done:,}/{len(texts):,}  {rate:,.0f}/s  '
                      f'eta {(len(texts) - done) / rate / 60:.0f} min', flush=True)
    return out, time.time() - t0


def cmd_dense(args):
    import numpy as np
    spec = DENSE[args.model]
    passages, queries = load_corpus()
    # only papers with a query need vectors (a query ranks only its own paper; 2026-10-19)
    with_query = {q['source_pmcid'] for q in queries}
    flat = [p for pmcid in sorted(passages) if pmcid in with_query for p in passages[pmcid]]
    if args.benchmark:
        sample = [p['text'] for p in flat[::max(1, len(flat) // args.benchmark)]][:args.benchmark]
        _, secs = encode(sample, spec['passage'], spec['passage_len'], spec['normalise'],
                         spec['pair_input'], batch=args.batch)
        _, secs = encode(sample, spec['passage'], spec['passage_len'], spec['normalise'],
                         spec['pair_input'], batch=args.batch)   # second pass: warm
        print(f'{args.model}: {len(sample):,} passages in {secs:.1f} s = {len(sample) / secs:,.0f}/s; '
              f'all {len(flat):,} would take ~{len(flat) / (len(sample) / secs) / 60:.0f} min')
        return
    p_vec, p_secs = encode([p['text'] for p in flat], spec['passage'], spec['passage_len'],
                           spec['normalise'], spec['pair_input'], batch=args.batch, label='passages')
    q_vec, q_secs = encode([q['query'] for q in queries], spec['query'], spec['query_len'],
                           spec['normalise'], False, batch=args.batch, label='queries')
    row = {p['pid']: i for i, p in enumerate(flat)}
    ranked = {}
    for q, qv in zip(queries, q_vec):
        docs = passages[q['source_pmcid']]
        mat = p_vec[[row[d['pid']] for d in docs]].astype(np.float32)
        scores = mat @ qv.astype(np.float32)
        top = np.argsort(-scores)[:TOP_K]
        ranked[q['item_id']] = [[docs[int(i)]['pid'], round(float(scores[i]), 4)] for i in top]
    write_run(args.model, ranked, {'method': f'dense {args.model}', 'models': [spec['query'], spec['passage']],
                                   'max_len': [spec['query_len'], spec['passage_len']],
                                   'normalise': spec['normalise'], 'dtype': 'fp32 (vectors stored fp16)',
                                   'encode_seconds': [round(p_secs), round(q_secs)], 'top_k': TOP_K})


RERANKER = 'BAAI/bge-reranker-v2-m3'     # cross-encoder (spec 5.R: rerank the top 50)


def cmd_rerank(args):
    """Re-score the top --depth passages of a first-stage run with a cross-encoder.
    Passages below the depth keep their first-stage order after the reranked block."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    passages, queries = load_corpus()
    if args.limit:
        queries = queries[:args.limit]
    text = {p['pid']: p['text'] for docs in passages.values() for p in docs}
    meta, run = load_run(args.run)
    pairs = [(q['item_id'], pid) for q in queries for pid in run[q['item_id']][:args.depth]]
    qtext = {q['item_id']: q['query'] for q in queries}
    if args.benchmark:
        pairs = pairs[::max(1, len(pairs) // args.benchmark)][:args.benchmark]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tok = AutoTokenizer.from_pretrained(RERANKER)
    model = AutoModelForSequenceClassification.from_pretrained(RERANKER).to(device).eval()

    # Token-budget batches (added 2026-10-12): the full test-2026 run stalled in its last 2%
    # when 16 long pairs filled the 4 GB GPU and Windows spilled into system memory. A batch
    # now holds at most --batch pairs AND at most --max-tokens padded tokens.
    lengths = [len(ids) for ids in tok([[qtext[q], text[p]] for q, p in pairs], truncation=True,
                                       max_length=args.max_len)['input_ids']]
    order = sorted(range(len(pairs)), key=lambda i: lengths[i])               # less padding
    batches, cur = [], []
    for i in order:
        if cur and (len(cur) >= args.batch or (len(cur) + 1) * lengths[i] > args.max_tokens):
            batches.append(cur)
            cur = []
        cur.append(i)
    if cur:
        batches.append(cur)

    # Checkpoint (added 2026-10-12): every batch's scores are appended to a .partial file; a
    # restart with the same run / depth / max_len / pairs resumes instead of starting over.
    scores = [None] * len(pairs)
    out_name = f'{args.run}+rerank' + (f'_limit{args.limit}' if args.limit else '')  # tests never overwrite
    partial = os.path.join(RUNS, out_name + '.partial.jsonl')
    header = {'run': args.run, 'depth': args.depth, 'max_len': args.max_len, 'pairs': len(pairs),
              'model': RERANKER}
    if not args.benchmark and os.path.exists(partial):
        rows = read_jsonl(partial)
        if rows and rows[0] == {'_header': header}:
            for r in rows[1:]:
                scores[r['i']] = r['s']
            print(f'  resuming: {sum(s is not None for s in scores):,} of {len(pairs):,} pairs already scored')
        else:
            print('  partial file is for a different setting; starting over')
            os.remove(partial)
    ckpt = None
    if not args.benchmark:
        os.makedirs(RUNS, exist_ok=True)
        fresh = not os.path.exists(partial)
        ckpt = open(partial, 'a', encoding='utf-8', newline='\n')
        if fresh:
            ckpt.write(json.dumps({'_header': header}) + '\n')
    todo = [b for b in batches if any(scores[i] is None for i in b)]
    t0, done = time.time(), 0
    with torch.no_grad():
        for n, idx in enumerate(todo):
            enc = tok([[qtext[pairs[i][0]], text[pairs[i][1]]] for i in idx], truncation=True,
                      padding=True, max_length=args.max_len, return_tensors='pt').to(device)
            logits = model(**enc).logits.view(-1).float().cpu().tolist()
            for i, s in zip(idx, logits):
                scores[i] = s
                if ckpt:
                    ckpt.write(json.dumps({'i': i, 's': s}) + '\n')
            if ckpt:
                ckpt.flush()
            done += len(idx)
            if n % 200 == 0 and n:
                rate = done / (time.time() - t0)
                left = sum(len(b) for b in todo[n + 1:])
                print(f'  rerank: {done:,}/{sum(len(b) for b in todo):,}  {rate:,.1f}/s  '
                      f'eta {left / rate / 60:.0f} min  (batch of {len(idx)})', flush=True)
    if ckpt:
        ckpt.close()
    secs = time.time() - t0
    if args.benchmark:
        total = sum(min(args.depth, len(run[q['item_id']])) for q in queries)
        print(f'rerank: {len(pairs):,} pairs in {secs:.1f} s = {len(pairs) / secs:,.1f}/s; '
              f'all {total:,} pairs (depth {args.depth}) would take ~{total / (len(pairs) / secs) / 60:.0f} min')
        return
    by_q = collections.defaultdict(dict)
    for (qid, pid), s in zip(pairs, scores):
        by_q[qid][pid] = s
    ranked = {}
    for q in queries:
        qid = q['item_id']
        top = sorted(by_q[qid], key=lambda pid: -by_q[qid][pid])
        # same [pid, score] rows as other runs; the un-reranked tail has no score
        ranked[qid] = [[pid, round(by_q[qid][pid], 4)] for pid in top] + \
                      [[pid, None] for pid in run[qid][args.depth:]]
    write_run(out_name, ranked,
              {'method': 'cross-encoder rerank', 'model': RERANKER, 'first_stage': args.run,
               'depth': args.depth, 'max_len': args.max_len, 'seconds': round(secs)})
    os.remove(partial)          # the finished run replaces the checkpoint


def load_run(name):
    rows = read_jsonl(os.path.join(RUNS, name + '.jsonl'))
    return rows[0]['_meta'], {r['item_id']: [pid for pid, _ in r['ranking']] for r in rows[1:]}


def cmd_hybrid(args):
    a_meta, a = load_run(args.a)
    b_meta, b = load_run(args.b)
    ranked = {}
    for item_id in a:
        score = collections.Counter()
        for run in (a, b):
            for rank, pid in enumerate(run.get(item_id, []), 1):
                score[pid] += 1.0 / (RRF_K + rank)
        ranked[item_id] = [[pid, round(s, 6)] for pid, s in score.most_common(TOP_K)]
    write_run(f'hybrid_{args.a}+{args.b}', ranked,
              {'method': 'reciprocal-rank fusion', 'k': RRF_K, 'runs': [args.a, args.b]})


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #
def metrics_for(ranking, qrel):
    hits = [qrel.get(pid, 0) for pid in ranking]
    first = next((i for i, g in enumerate(hits, 1) if g), None)
    dcg = sum((2 ** g - 1) / math.log2(i + 1) for i, g in enumerate(hits[:10], 1))
    ideal = sorted(qrel.values(), reverse=True)[:10]
    idcg = sum((2 ** g - 1) / math.log2(i + 1) for i, g in enumerate(ideal, 1))
    return {'recall@5': float(any(hits[:5])), 'recall@10': float(any(hits[:10])),
            'mrr': 1.0 / first if first else 0.0, 'ndcg@10': dcg / idcg if idcg else 0.0}


def cmd_score(args):
    meta, run = load_run(args.run)
    queries = {q['item_id']: q for q in read_jsonl(os.path.join(OUT, 'queries.jsonl'))}
    # release labels (--set full) store {grade, match, passage_type}; the sample stores the grade
    qrels = {r['item_id']: {p: (g['grade'] if isinstance(g, dict) else g) for p, g in r['qrel'].items()}
             for r in read_jsonl(os.path.join(OUT, 'qrels.jsonl'))}
    ptype = {p['pid']: p['type'] for p in read_jsonl(os.path.join(OUT, 'passages.jsonl.gz'), gzip.open)}
    groups = collections.defaultdict(list)
    for item_id, qrel in qrels.items():
        m = metrics_for(run.get(item_id, []), qrel)
        best = max(qrel.values())
        where = 'table' if any(ptype[p] in ('cell', 'row') for p, g in qrel.items() if g == best) else 'text'
        for g in ('all', 'content=' + queries[item_id]['content'], 'evidence=' + where):
            groups[g].append(m)
    result = {'run': args.run, 'meta': meta, 'groups': {}}
    print(f"run {args.run}: {meta.get('method')}")
    for g in sorted(groups, key=lambda g: (g != 'all', g)):
        ms = groups[g]
        avg = {k: round(sum(m[k] for m in ms) / len(ms), 4) for k in ms[0]}
        result['groups'][g] = {'n': len(ms), **avg}
        print(f"  {g:<26} n={len(ms):>5}  R@5 {avg['recall@5']:.3f}  R@10 {avg['recall@10']:.3f}  "
              f"MRR {avg['mrr']:.3f}  nDCG@10 {avg['ndcg@10']:.3f}")
    with open(os.path.join(RUNS, args.run + '.metrics.json'), 'w', encoding='utf-8', newline='\n') as f:
        json.dump(result, f, indent=2)


def main():
    global OUT, RUNS
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--set', choices=['sample', 'full', 'dev'], default='sample',
                    help='sample: the QA evaluation sample (build); full: ALL test-2026 queries, '
                         'from the release labels (label-all), runs in data/v2/eval/retrieval_full/; '
                         'dev: ALL dev-2026 queries (selecting the reranker\'s first stage, '
                         '2026-10-19), runs in data/v2/eval/retrieval_dev/')
    sub = ap.add_subparsers(dest='command', required=True)
    sub.add_parser('build').set_defaults(func=cmd_build)
    sp = sub.add_parser('label-all', help='relevance labels for every split (release)')
    sp.add_argument('--splits', nargs='*', choices=SPLITS)
    sp.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    sp.add_argument('--limit', type=int, default=0, help='first N papers per split (test run)')
    sp.set_defaults(func=cmd_label_all)
    sub.add_parser('bm25').set_defaults(func=cmd_bm25)
    sp = sub.add_parser('dense')
    sp.add_argument('--model', choices=sorted(DENSE), required=True)
    sp.add_argument('--benchmark', type=int, default=0, help='time N passages and stop')
    sp.add_argument('--batch', type=int, default=32, help='encoding batch size (4 GB GPU: keep small)')
    sp.set_defaults(func=cmd_dense)
    sp = sub.add_parser('hybrid')
    sp.add_argument('--a', required=True)
    sp.add_argument('--b', required=True)
    sp.set_defaults(func=cmd_hybrid)
    sp = sub.add_parser('rerank')
    sp.add_argument('--run', required=True, help='first-stage run to rerank')
    sp.add_argument('--depth', type=int, default=50)
    sp.add_argument('--max-len', type=int, default=384)
    sp.add_argument('--batch', type=int, default=16, help='at most this many pairs per batch')
    sp.add_argument('--max-tokens', type=int, default=4096,
                    help='at most this many padded tokens per batch (4 GB GPU: 16 x 384 = 6,144 stalled)')
    sp.add_argument('--benchmark', type=int, default=0, help='time N pairs and stop')
    sp.add_argument('--limit', type=int, default=0, help='first N queries only (testing)')
    sp.set_defaults(func=cmd_rerank)
    sp = sub.add_parser('score')
    sp.add_argument('--run', required=True)
    sp.set_defaults(func=cmd_score)
    args = ap.parse_args()
    if args.set == 'full':
        if args.command == 'build':
            ap.error('build makes the sample set; the full set comes from label-all')
        OUT = os.path.join(LABELS_OUT, 'test-2026')
        RUNS = os.path.join(BASE, 'data', 'v2', 'eval', 'retrieval_full', 'runs')
    elif args.set == 'dev':
        if args.command in ('build', 'label-all'):
            ap.error('dev reads the release labels of dev-2026; build and label-all do not apply')
        OUT = os.path.join(LABELS_OUT, 'dev-2026')
        RUNS = os.path.join(BASE, 'data', 'v2', 'eval', 'retrieval_dev', 'runs')
    args.func(args)


if __name__ == '__main__':
    main()
