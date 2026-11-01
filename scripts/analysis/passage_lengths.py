"""
passage_lengths.py
------------------
How many test-2026 passages the dense retrievers truncate (asked by a supervisor,
2026-10-29: "is it possible to have passages longer than 256 tokens?"). Each dense
model in scripts/eval/retrieval.py cuts a passage at 256 tokens, special tokens
included; this counts, with each model's own tokenizer and its own input form (MedCPT
encodes a passage as an empty title paired with the text), the passages longer than
that, overall and among relevant passages, by passage type. CPU only, about a minute.

Output: data/v2/analysis/passage_lengths.json

Usage:  python scripts/analysis/passage_lengths.py
"""
import collections
import gzip
import json
import os

from transformers import AutoTokenizer

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LABELS = os.path.join(BASE, 'data', 'v2', 'retrieval_labels', 'test-2026')
OUT = os.path.join(BASE, 'data', 'v2', 'analysis', 'passage_lengths.json')
MAX_LEN = 256                                      # retrieval.py DENSE[*]['passage_len']
MODELS = {'medcpt': ('ncbi/MedCPT-Article-Encoder', True), 'bge-m3': ('BAAI/bge-m3', False)}


def main():
    relevant = set()
    with open(os.path.join(LABELS, 'qrels.jsonl'), encoding='utf-8') as f:
        for r in map(json.loads, f):
            relevant |= set(r['qrel'])
    texts, kinds = [], []
    with gzip.open(os.path.join(LABELS, 'passages.jsonl.gz'), 'rt', encoding='utf-8') as f:
        for p in map(json.loads, f):
            texts.append(p['text'])
            kinds.append((p['type'], p['pid'] in relevant))

    out = {'split': 'test-2026', 'max_len': MAX_LEN, 'passages': len(texts),
           'relevant_passages': len(relevant), 'models': {}}
    for name, (model_id, pair_input) in MODELS.items():
        tok = AutoTokenizer.from_pretrained(model_id)
        total, over = collections.Counter(), collections.Counter()
        for i in range(0, len(texts), 20000):
            chunk = texts[i:i + 20000]
            ids = tok([['', t] for t in chunk] if pair_input else chunk)['input_ids']
            for (kind, rel), n in zip(kinds[i:i + 20000], map(len, ids)):
                for key in ('all', kind) + (('relevant', 'relevant:' + kind) if rel else ()):
                    total[key] += 1
                    over[key] += n > MAX_LEN
        out['models'][name] = {key: {'passages': total[key], 'over': over[key],
                                     'rate': round(over[key] / total[key], 5)} for key in sorted(total)}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(out, f, indent=1)
    for name, m in out['models'].items():
        print(name, {k: f"{v['over']}/{v['passages']}" for k, v in m.items()})


if __name__ == '__main__':
    main()
