"""
crossjudge.py
-------------
Stage 5.4b of the v2 pipeline (agreed 2026-10-11): does the Gemini judge favour Gemma,
a model from its own lab? A second judge from another lab - Claude Sonnet 5.5 - re-judges
answers Gemini already judged, with v1's two judge prompts and schemas unchanged.

Design: 100 random test-2026 papers from the evaluation sample, the SAME papers for every
model (paired); every judged item of those papers; models Gemma 4 31B (contested) and three
controls from other labs (DeepSeek V3.2, GLM-5, Qwen3.5-27B). The requests are rebuilt by
judge_answers.build_requests, so Sonnet sees exactly what Gemini saw.

Settings: claude-sonnet-5-5, thinking off (`between_tools`: Sonnet 5.5 rejects
`disabled`), effort low, structured output (the verdict schema, additionalProperties
false), no temperature (Sonnet 5.5 rejects non-default sampling), Anthropic batch API.

Steps:
  prepare   -> data/v2/eval/crossjudge/requests.jsonl, keys.json   (no API call)
  smoke     send the first N requests directly; print verdicts and tokens
  submit    one Anthropic batch                                    -> batch_state.json
  status    batch progress
  collect   -> verdicts.jsonl, then the comparison -> crossjudge_result.json

Usage:
  python scripts/eval/crossjudge.py prepare
  python scripts/eval/crossjudge.py smoke --n 4
  python scripts/eval/crossjudge.py submit
  python scripts/eval/crossjudge.py collect
"""
import argparse
import collections
import copy
import importlib.util
import json
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(BASE, 'scripts'))
import _env  # noqa: E402,F401


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


judge = _load('judge_answers', os.path.join(HERE, 'judge_answers.py'))

try:
    sys.stdout.reconfigure(encoding='utf-8')
except (AttributeError, ValueError):
    pass

ANSWERS = os.path.join(BASE, 'data', 'v2', 'eval', 'answers', 'eval_sample_linear')
OUT = os.path.join(BASE, 'data', 'v2', 'eval', 'crossjudge')
MODEL = 'claude-sonnet-5-5'
MODELS = ['gemma-4-31b-it', 'deepseek-v3.2', 'glm-5', 'qwen3.5-27b']
CONTESTED = 'gemma-4-31b-it'
N_PAPERS = 100
SEED = 20261011
MAX_TOKENS = 2048
N_BOOT = 2000


def strict(schema):
    """The judge's JSON schema with additionalProperties false (structured outputs needs it)."""
    s = copy.deepcopy(schema['json_schema'])
    s['additionalProperties'] = False
    return s


SCHEMAS = {'F': judge.FAITH_SCHEMA, 'C': judge.CORRECT_SCHEMA}
SYSTEMS = {'F': judge.FAITH_SYSTEM, 'C': judge.CORRECT_SYSTEM}


def params(axis, text):
    return {'model': MODEL, 'max_tokens': MAX_TOKENS,
            'system': SYSTEMS[axis],
            'messages': [{'role': 'user', 'content': text}],
            'thinking': {'type': 'between_tools'},
            'output_config': {'effort': 'low',
                              'format': {'type': 'json_schema', 'schema': strict(SCHEMAS[axis])}}}


def paths():
    return {k: os.path.join(OUT, v) for k, v in (
        ('requests', 'requests.jsonl'), ('keys', 'keys.json'), ('state', 'batch_state.json'),
        ('verdicts', 'verdicts.jsonl'), ('result', 'crossjudge_result.json'), ('smoke', 'smoke.json'))}


def cmd_prepare(args):
    p = paths()
    # the paper sample: test-2026 papers of the evaluation sample, seeded
    with open(os.path.join(BASE, 'data', 'v2', 'eval', 'eval_sample.jsonl'), encoding='utf-8') as f:
        papers = sorted({r['source_pmcid'] for r in map(json.loads, f) if r['split'] == 'test-2026'})
    chosen = set(random.Random(SEED).sample(papers, N_PAPERS))
    requests, keys = [], {}
    for model in MODELS:
        summary, items = judge.load_items(os.path.join(ANSWERS, model, 'answers.json'))
        items = [it for it in items if it['source_pmcid'] in chosen]
        lines, _refs = judge.build_requests(summary, items)
        for ln in lines:
            item_id, axis = ln['key'].rsplit('|', 1)
            text = ln['request']['contents'][0]['parts'][0]['text']
            custom_id = f'r{len(requests):05d}'          # custom ids: [A-Za-z0-9_-], <= 64 chars
            keys[custom_id] = {'model': model, 'item_id': item_id, 'axis': axis}
            requests.append({'custom_id': custom_id, 'params': params(axis, text)})
    os.makedirs(OUT, exist_ok=True)
    with open(p['requests'], 'w', encoding='utf-8', newline='\n') as f:
        for r in requests:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    with open(p['keys'], 'w', encoding='utf-8', newline='\n') as f:
        json.dump({'papers': sorted(chosen), 'keys': keys}, f)
    per = collections.Counter((k['model'], k['axis']) for k in keys.values())
    chars = sum(len(r['params']['system']) + len(r['params']['messages'][0]['content']) for r in requests)
    # rough: ~4 chars per token for this prose; Sonnet 5.5 batch $1 / $5 per M; ~200 output tokens
    est = chars / 4 * 1.0 / 1e6 + len(requests) * 200 * 5.0 / 1e6
    print(f'papers {len(chosen)}; requests {len(requests):,}: ' +
          ', '.join(f'{m} F {per[(m, "F")]} C {per[(m, "C")]}' for m in MODELS))
    print(f'estimate ~${est:.2f} at batch prices')


def load_requests():
    with open(paths()['requests'], encoding='utf-8') as f:
        return [json.loads(line) for line in f]


def client():
    import anthropic
    return anthropic.Anthropic()


def parse(message, axis):
    if message.stop_reason == 'refusal':
        raise ValueError('refusal')
    text = next((b.text for b in message.content if b.type == 'text'), '')
    return judge.parse_verdict(text, SCHEMAS[axis])


def cmd_smoke(args):
    c = client()
    with open(paths()['keys'], encoding='utf-8') as f:
        keys = json.load(f)['keys']
    out = []
    for r in load_requests()[:args.n]:
        msg = c.messages.create(**r['params'])
        k = keys[r['custom_id']]
        v = parse(msg, k['axis'])
        out.append({'custom_id': r['custom_id'], **k, 'verdict': v,
                    'usage': {'in': msg.usage.input_tokens, 'out': msg.usage.output_tokens}})
        print(f"{k['model']} {k['item_id']} {k['axis']}: {json.dumps(v)[:200]}  "
              f"in {msg.usage.input_tokens} out {msg.usage.output_tokens} stop {msg.stop_reason}")
    with open(paths()['smoke'], 'w', encoding='utf-8', newline='\n') as f:
        json.dump(out, f, indent=1)


def cmd_submit(args):
    p = paths()
    if os.path.exists(p['state']):
        sys.exit(f'already submitted: see {p["state"]}')
    batch = client().messages.batches.create(requests=load_requests())
    with open(p['state'], 'w', encoding='utf-8', newline='\n') as f:
        json.dump({'batch_id': batch.id, 'model': MODEL, 'submitted': time.strftime('%Y-%m-%d %H:%M')}, f)
    print(f'submitted {batch.id} ({batch.processing_status})')


def cmd_status(args):
    with open(paths()['state'], encoding='utf-8') as f:
        state = json.load(f)
    b = client().messages.batches.retrieve(state['batch_id'])
    print(state['batch_id'], b.processing_status, dict(b.request_counts))


def bootstrap(papers, diff_of, rng):
    boots = sorted(diff_of([papers[rng.randrange(len(papers))] for _ in papers]) for _ in range(N_BOOT))
    return [round(boots[int(0.025 * N_BOOT)], 4), round(boots[int(0.975 * N_BOOT) - 1], 4)]


def cmd_collect(args):
    p = paths()
    with open(p['state'], encoding='utf-8') as f:
        state = json.load(f)
    with open(p['keys'], encoding='utf-8') as f:
        meta = json.load(f)
    keys = meta['keys']
    c = client()
    if c.messages.batches.retrieve(state['batch_id']).processing_status != 'ended':
        sys.exit('batch not ended yet')
    sonnet, errors, tokens = {}, [], collections.Counter()
    with open(p['verdicts'], 'w', encoding='utf-8', newline='\n') as f:
        for res in c.messages.batches.results(state['batch_id']):
            k = keys[res.custom_id]
            if res.result.type != 'succeeded':
                errors.append((res.custom_id, res.result.type))
                continue
            msg = res.result.message
            tokens['in'] += msg.usage.input_tokens
            tokens['out'] += msg.usage.output_tokens
            try:
                v = parse(msg, k['axis'])
            except ValueError as e:
                errors.append((res.custom_id, str(e)[:100]))
                continue
            sonnet[(k['model'], k['item_id'], k['axis'])] = v
            f.write(json.dumps({**k, 'verdict': v}, ensure_ascii=False) + '\n')

    # pair with Gemini's verdicts on the same items
    pairs = collections.defaultdict(list)       # (model, axis) -> [(paper, gemini_ok, sonnet_ok)]
    for model in MODELS:
        with open(os.path.join(ANSWERS, model, 'judge', 'judged.json'), encoding='utf-8') as f:
            gem = {r['item_id']: r for r in json.load(f)['items']}
        for (m, item_id, axis), v in sonnet.items():
            if m != model:
                continue
            g = gem.get(item_id, {})
            g_v = g.get('faithfulness' if axis == 'F' else 'correctness')
            if not g_v:
                continue
            ok = judge.is_faithful if axis == 'F' else judge.is_correct
            paper = g['instance_id'].split('__')[1]
            pairs[(model, axis)].append((paper, ok(g_v), ok(v)))

    result = {'judge_a': judge.MODEL, 'judge_b': MODEL, 'papers': len(meta['papers']),
              'errors': len(errors), 'tokens': dict(tokens),
              'cost_usd_batch': round((tokens['in'] * 1.0 + tokens['out'] * 5.0) / 1e6, 4),
              'by_model': {}, 'gap_vs_controls': {}}
    for (model, axis), rows in sorted(pairs.items()):
        n = len(rows)
        g_rate = sum(r[1] for r in rows) / n
        s_rate = sum(r[2] for r in rows) / n
        agree = sum(r[1] == r[2] for r in rows) / n
        result['by_model'].setdefault(model, {})[axis] = {
            'n': n, 'gemini': round(g_rate, 4), 'sonnet': round(s_rate, 4),
            'gemini_minus_sonnet': round(g_rate - s_rate, 4), 'agreement': round(agree, 4)}
    # difference-in-differences: (Gemini - Sonnet) for Gemma minus the same for the controls,
    # paired by paper, bootstrap over the 100 papers
    rng = random.Random(SEED)
    for axis in ('F', 'C'):
        by_paper = collections.defaultdict(lambda: collections.defaultdict(list))
        for model in MODELS:
            for paper, g_ok, s_ok in pairs.get((model, axis), []):
                by_paper[paper][model].append(g_ok - s_ok)

        def did(papers):
            gm = [d for pp in papers for d in by_paper[pp][CONTESTED]]
            ct = [d for pp in papers for m in MODELS if m != CONTESTED for d in by_paper[pp][m]]
            return (sum(gm) / len(gm) if gm else 0) - (sum(ct) / len(ct) if ct else 0)
        papers = sorted(by_paper)
        result['gap_vs_controls'][axis] = {'did': round(did(papers), 4),
                                           'ci95': bootstrap(papers, did, rng)}
    # The judges disagree almost only on answers whose quote is NOT found verbatim: the
    # faithfulness prompt says "a missing quote is itself a grounding failure", which Gemini
    # applies and Sonnet largely does not. Models differ in how often that happens, so the
    # same-lab test is repeated WITHIN each stratum (2026-10-11).
    found = {}
    for model in MODELS:
        with open(os.path.join(ANSWERS, model, 'answers.json'), encoding='utf-8') as f:
            for inst in json.load(f)['instances']:
                for q in inst['questions']:
                    for item_id in q['item_ids']:
                        found[(model, item_id)] = bool((q['prediction']['provenance'] or {}).get('quote_found'))
    strata = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(list)))
    by_stratum_model = collections.defaultdict(lambda: [0, 0, 0])
    for model in MODELS:
        with open(os.path.join(ANSWERS, model, 'judge', 'judged.json'), encoding='utf-8') as f:
            gem = {r['item_id']: r for r in json.load(f)['items']}
        for (m, item_id, axis), v in sonnet.items():
            if m != model or axis != 'F' or not gem.get(item_id, {}).get('faithfulness'):
                continue
            stratum = 'quote_found' if found.get((model, item_id)) else 'quote_not_found'
            g_ok, s_ok = judge.is_faithful(gem[item_id]['faithfulness']), judge.is_faithful(v)
            strata[stratum][gem[item_id]['instance_id'].split('__')[1]][model].append(g_ok - s_ok)
            cell = by_stratum_model[(stratum, model)]
            cell[0] += 1
            cell[1] += g_ok
            cell[2] += s_ok
    result['faithfulness_by_quote'] = {}
    for stratum in ('quote_found', 'quote_not_found'):
        by_paper = strata[stratum]

        def did(papers, by_paper=by_paper):
            gm = [d for pp in papers for d in by_paper[pp][CONTESTED]]
            ct = [d for pp in papers for m in MODELS if m != CONTESTED for d in by_paper[pp][m]]
            return (sum(gm) / len(gm) if gm else 0) - (sum(ct) / len(ct) if ct else 0)
        papers = sorted(by_paper)
        result['faithfulness_by_quote'][stratum] = {
            'models': {m: {'n': by_stratum_model[(stratum, m)][0],
                           'gemini': round(by_stratum_model[(stratum, m)][1] / by_stratum_model[(stratum, m)][0], 4),
                           'sonnet': round(by_stratum_model[(stratum, m)][2] / by_stratum_model[(stratum, m)][0], 4)}
                       for m in MODELS if by_stratum_model[(stratum, m)][0]},
            'did': round(did(papers), 4), 'ci95': bootstrap(papers, did, rng)}
    with open(p['result'], 'w', encoding='utf-8', newline='\n') as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    if errors:
        print(f'{len(errors)} errors; first 3: {errors[:3]}')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='command', required=True)
    for name, func in (('prepare', cmd_prepare), ('smoke', cmd_smoke), ('submit', cmd_submit),
                       ('status', cmd_status), ('collect', cmd_collect)):
        sp = sub.add_parser(name)
        sp.set_defaults(func=func)
        if name == 'smoke':
            sp.add_argument('--n', type=int, default=4)
    args = ap.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
