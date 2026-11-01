"""
providers.py
------------
Stage 5.1 of the v2 pipeline: the evaluation roster and, per model, the
approved list of OpenRouter providers allowed to serve it.

Rules (agreed 2026-10-09, option A). A provider endpoint is approved only if ALL hold:
  * precision fp8 or better (fp8 / bf16 / fp16 / fp32); fp4, int4 and "unknown"
    are excluded - lower precision can lower answer quality, and "unknown" cannot
    be reported;
  * context length >= 128k tokens (our longest request is ~50k tokens; a few
    providers serve 32k and would reject long papers);
  * strict structured output supported (OpenRouter's `structured_outputs` flag),
    since every answer comes back through a strict JSON schema.

The runner sends `provider.only` = the approved tags with fallbacks off, so a
request can never route outside the list, and it records which provider served
each request. The list is snapshotted to a dated JSON file and the runner reads
that file, so a run is reproducible and the paper can quote exactly what ran.

Roster (agreed 2026-10-07): the 10 API-served models. Qwen3.5-4B and Gemma 4 E4B
are not on OpenRouter; they run on a local GPU later (JAMIA, or ECIR if time allows).

Output:  data/v2/eval/providers.json   {snapshot_date, rules, models: {slug: {approved:
                                       [{tag, provider, quantization, context,
                                         reasoning_param, price_in_per_m, price_out_per_m}],
                                       rejected: [{tag, reason}]}}}

Usage:
  python scripts/eval/providers.py            # query OpenRouter, write the snapshot
  python scripts/eval/providers.py --show     # print the saved snapshot
"""
import argparse
import datetime
import json
import os
import urllib.request

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_PATH = os.path.join(BASE, 'data', 'v2', 'eval', 'providers.json')
ENDPOINTS_API = 'https://openrouter.ai/api/v1/models/{slug}/endpoints'

ROSTER = [
    'qwen/qwen3.5-397b-a17b',
    'qwen/qwen3.5-122b-a10b',
    'qwen/qwen3.5-35b-a3b',
    'qwen/qwen3.5-27b',
    'qwen/qwen3.5-9b',
    'google/gemma-4-31b-it',
    'z-ai/glm-5',                   # replaces GLM-4.7 (no qualifying provider), 2026-10-09
    'deepseek/deepseek-v3.2',
    'meta-llama/llama-4-maverick',
    'meta-llama/llama-4-scout',
]

GOOD_PRECISION = {'fp8', 'bf16', 'fp16', 'fp32'}
MIN_CONTEXT = 128_000
# 'structured_outputs' is OpenRouter's flag for strict json_schema support (sent via
# response_format); a bare 'response_format' only means loose JSON mode.
NEEDED_PARAMETERS = {'structured_outputs'}


def fetch_endpoints(slug):
    with urllib.request.urlopen(ENDPOINTS_API.format(slug=slug), timeout=60) as r:
        return json.loads(r.read())['data']['endpoints']


def judge_endpoint(endpoint):
    """Return '' if the endpoint passes every rule, else the first reason it fails."""
    quantization = (endpoint.get('quantization') or 'unknown').lower()
    if quantization not in GOOD_PRECISION:
        return f'precision {quantization}'
    if (endpoint.get('context_length') or 0) < MIN_CONTEXT:
        return f"context {endpoint.get('context_length')}"
    missing = NEEDED_PARAMETERS - set(endpoint.get('supported_parameters') or [])
    if missing:
        return 'no ' + '/'.join(sorted(missing))
    return ''


def snapshot():
    models = {}
    for slug in ROSTER:
        approved, rejected = [], []
        for e in fetch_endpoints(slug):
            reason = judge_endpoint(e)
            if reason:
                rejected.append({'tag': e['tag'], 'reason': reason})
            else:
                approved.append({'tag': e['tag'], 'provider': e['provider_name'],
                                 'quantization': e['quantization'],
                                 'context': e['context_length'],
                                 # can thinking be switched off here? models with no
                                 # reasoning mode do not list the parameter at all
                                 'reasoning_param': 'reasoning' in (e.get('supported_parameters') or []),
                                 'price_in_per_m': float(e['pricing']['prompt']) * 1e6,
                                 'price_out_per_m': float(e['pricing']['completion']) * 1e6})
        models[slug] = {'approved': approved, 'rejected': rejected}
    return {
        'snapshot_date': datetime.date.today().isoformat(),
        'rules': {'precision': sorted(GOOD_PRECISION), 'min_context': MIN_CONTEXT,
                  'parameters': sorted(NEEDED_PARAMETERS)},
        'models': models,
    }


def show(data):
    print(f"snapshot {data['snapshot_date']}")
    for slug, m in data['models'].items():
        ok = ', '.join(f"{a['tag']} ({a['context'] // 1000}k)" for a in m['approved']) or 'NONE'
        print(f'{slug:30s} approved: {ok}')
        for r in m['rejected']:
            print(f"{'':30s}   rejected {r['tag']}: {r['reason']}")


def load():
    """The saved snapshot, for the runner."""
    with open(OUT_PATH, encoding='utf-8') as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--show', action='store_true')
    args = ap.parse_args()
    if args.show:
        show(load())
        return
    data = snapshot()
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(data, f, indent=2)
    show(data)
    empty = [s for s, m in data['models'].items() if not m['approved']]
    if empty:
        raise SystemExit(f'no approved provider for: {", ".join(empty)}')


if __name__ == '__main__':
    main()
