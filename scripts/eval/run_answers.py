"""
run_answers.py
--------------
Stage 5.3 of the v2 pipeline: zero-shot question answering over a source paper's
full text, through OpenRouter, with provenance.

One INSTANCE = one (systematic review x source paper) pair. Every sampled question
the review asks about that paper goes to the model in a single call, against the
paper's full text. For each question the model answers (or abstains with
not_reported) and cites a section label plus a verbatim quote; each quote is then
located in the parsed paper offline, so provenance is machine-checked.

Ported from v1 (old/scripts/run_zeroshot_qa.py + run_zeroshot_qa_openrouter.py).
Kept unchanged, as agreed 2026-10-09: SYSTEM_PROMPT, the answer schema,
build_prompt (questions before AND after the text; review title/objectives as
context only), locate_quote, assemble, temperature 0, one retry at temperature
0.5 when a reply covers under half the questions. Changes:

  * inputs are v2's frozen release and a sample file from 21_eval_sample.py;
    one question is asked once per instance and its answer fans out to every
    item_id that shares the question text;
  * reasoning OFF for every model (agreed): sent only where the approved
    providers accept the parameter - models with no reasoning mode (Llama 4)
    do not list it, and the reply's reasoning-token count is recorded so the
    smoke test can confirm it is zero;
  * providers: `provider.only` = the model's approved list from
    data/v2/eval/providers.json (fp8 or better, >=128k context, strict
    schema), fallbacks off; the serving provider is recorded per request;
  * the source cap is 200,000 chars (agreed) and v1's 40,960-token budget is
    gone; instead every request's size is bounded before sending
    (chars x 0.30 tokens/char + max_tokens, above the measured 0.20-0.22) and
    the run stops if any would exceed the model's smallest approved context;
  * --layout sectioned|linear selects the reading (the layout test).

Inputs:  release/v2/data/<split>.jsonl.gz, release/v2/review_context.jsonl,
         SOURCE_XML/, data/v2/eval/providers.json, a sample file
         (data/v2/eval/eval_sample.jsonl or layout_test_sample.jsonl)
Outputs: data/v2/eval/answers/<tag>/<model>/raw.jsonl     append-only, one instance
                                                          per line (enables --resume)
         data/v2/eval/answers/<tag>/<model>/answers.json  summary + all instances

Usage:
  python scripts/eval/run_answers.py --model qwen/qwen3.5-27b --dry-run
  python scripts/eval/run_answers.py --model qwen/qwen3.5-27b --limit 3 --tag smoke
  python scripts/eval/run_answers.py --model qwen/qwen3.5-27b \
      --sample data/v2/eval/layout_test_sample.jsonl --layout linear --tag layout-linear
"""
import argparse
import collections
import gzip
import importlib.util
import json
import os
import random
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(BASE, 'scripts'))
import _env  # noqa: E402,F401  (loads OPENROUTER_API_KEY from .env if not set)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


jats_source = _load('jats_source', os.path.join(HERE, 'jats_source.py'))
providers = _load('providers', os.path.join(HERE, 'providers.py'))

try:
    sys.stdout.reconfigure(encoding='utf-8')
except (AttributeError, ValueError):
    pass

DATA_DIR = os.path.join(BASE, 'release', 'v2', 'data')
REVIEW_CONTEXT = os.path.join(BASE, 'release', 'v2', 'review_context.jsonl')
SOURCE_DIR = os.path.join(BASE, 'SOURCE_XML')
DEFAULT_SAMPLE = os.path.join(BASE, 'data', 'v2', 'eval', 'eval_sample.jsonl')
OUT_ROOT = os.path.join(BASE, 'data', 'v2', 'eval', 'answers')
API = 'https://openrouter.ai/api/v1/chat/completions'

MAX_SOURCE_CHARS = 200_000
# Upper bound for the pre-send size check. v1 reported 0.20-0.22 input tokens per
# prompt character, but the full v2 Qwen3.5-27B run measured 0.31 (17.0M tokens over
# ~55M request chars, 2026-10-09; v1's figure divided by source chars only). 0.40
# leaves a margin, so a request that passes cannot overflow.
TOKENS_PER_CHAR_BOUND = 0.40
# For the cost estimate only: the measured v2 rate.
TOKENS_PER_CHAR_EST = 0.31
OUTPUT_TOKENS_PER_INSTANCE_EST = 1100  # measured, Qwen3.5-27B full run, reasoning off

# --------------------------------------------------------------------------- #
# prompt and schema: v1, unchanged
# --------------------------------------------------------------------------- #
SYSTEM_PROMPT = (
    "You are a meticulous biomedical data-extraction assistant supporting a systematic "
    "review. You are given the full text of ONE study, split into numbered sections, and "
    "a list of the review's data-extraction questions about that study. Answer all of "
    "them in one go.\n\n"
    "Rules:\n"
    "1. Answer every question using ONLY the full text provided. Do not use outside "
    "knowledge and do not guess.\n"
    "2. If the study does not report what a question asks for, set not_reported=true and "
    "leave answer and evidence_quote empty. Abstaining is correct and expected; never "
    "fabricate a value to fill a gap.\n"
    "3. When the study does report it, extract the value. Prefer the study's own wording "
    "verbatim for concrete facts (country, sample size, ages, measures); for questions "
    "that ask for a summary, write a short faithful summary grounded in the text.\n"
    "4. For every question you answer, give its provenance: evidence_section is the "
    "section label exactly as shown (e.g. \"S3\"), and evidence_quote is a span of AT "
    "MOST 300 characters copied VERBATIM, character for character, from that section. "
    "The quote must be text you can actually see in the section, not a paraphrase and "
    "not stitched together from separate places. It must contain or directly support "
    "the value you extracted. Quotes are checked automatically against the paper.\n"
    "5. Tables are given as their own sections, each holding the table's caption, a "
    "Markdown grid, a list of its cell values written as "
    "[row=<row label> | <column header> = <value>], and its footnotes. Many extracted "
    "values live only in a table, so read them. A cell means nothing without its row "
    "and column header, so read the three together, and check the footnotes for the "
    "units and abbreviation expansions before reporting a number. Either the grid line "
    "or the bracketed cell line is valid to quote verbatim.\n"
    "6. Echo each question_id exactly as given so answers can be matched up. Return one "
    "entry per question, and answer every question you were asked.\n"
    "Return your answers by calling the submit_answers tool once."
)

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "answers": {
            "type": "array",
            "description": "One entry per question asked, in the same order.",
            "items": {
                "type": "object",
                "properties": {
                    "question_id": {
                        "type": "string",
                        "description": 'The exact question id given, e.g. "q3".',
                    },
                    "not_reported": {
                        "type": "boolean",
                        "description": "True if the study does not report this.",
                    },
                    "answer": {
                        "type": "string",
                        "description": "The extracted value; empty when not_reported is true.",
                    },
                    "evidence_section": {
                        "type": "string",
                        "description": 'Section label the answer came from, e.g. "S3". Empty when not_reported.',
                    },
                    "evidence_quote": {
                        "type": "string",
                        "description": "Verbatim span (<=300 chars) copied from that section that supports the answer. Empty when not_reported.",
                    },
                },
                "required": ["question_id", "not_reported", "answer",
                             "evidence_section", "evidence_quote"],
            },
        }
    },
    "required": ["answers"],
}


def build_prompt(inst, ctx, source_text, repeat_questions=True):
    """Assemble the user turn (v1, unchanged).

    The questions are stated BEFORE the full text and repeated after it. Stating
    them first means the paper is read already knowing what is being looked for;
    repeating them last puts them in the recency zone, where a long prompt is
    attended to most reliably.
    """
    parts = []
    if ctx:
        scope = [f"# SYSTEMATIC REVIEW SCOPE (context only - do not answer from it)\nTitle: {ctx.get('title', '')}"]
        if ctx.get("objectives"):
            scope.append("Background / objectives:\n" + ctx["objectives"])
        parts.append("\n\n".join(scope))

    qs = "\n".join(f'{q["question_id"]}. {q["question"]}' for q in inst["questions"])

    parts.append(
        "# QUESTIONS TO ANSWER\nThese are the questions you must answer about the study "
        "that follows. Read them first, then read the study with them in mind. They are "
        "repeated after the full text.\n" + qs
    )

    parts.append("# STUDY FULL TEXT\n" + source_text)

    if repeat_questions:
        parts.append(
            "# QUESTIONS (repeated)\nAnswer each question below about the study above, or "
            "abstain (not_reported=true) if the study does not report it. For each answer, "
            "cite the section label and a verbatim supporting quote from that section.\n" + qs
        )
    return "\n\n".join(parts)


# --------------------------------------------------------------------------- #
# provenance verification (v1, unchanged)
# --------------------------------------------------------------------------- #
def _norm(s):
    return re.sub(r"\s+", " ", (s or "")).strip().lower()


def locate_quote(src, quote):
    """Find `quote` in the paper and report where. Whitespace/case-insensitive.

    Returns the section the quote actually lives in and its offsets within that
    section's text - independent of whichever section the model claimed.
    """
    result = {"quote_found": False, "located_section_id": None,
              "located_section_title": None, "char_start": None, "char_end": None}
    nq = _norm(quote)
    if not nq:
        return result
    for s in src["sections"]:
        idx = _norm(s["text"]).find(nq)
        if idx >= 0:
            result.update(quote_found=True, located_section_id=s["id"],
                          located_section_title=s["title"],
                          char_start=idx, char_end=idx + len(nq))
            return result
    return result


# --------------------------------------------------------------------------- #
# instances from v2 items
# --------------------------------------------------------------------------- #
def load_sample(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(line) for line in f]


def load_instances(sample_rows):
    """Group the sampled items into (review x source paper) instances.

    Within an instance the same question text can occur more than once (the same
    column in two tables, or one paper filling several study rows). Each DISTINCT
    question is asked once; its answer fans out to every item_id that shares it.
    """
    wanted = {r['item_id'] for r in sample_rows}
    splits = sorted({r['split'] for r in sample_rows})
    groups = collections.defaultdict(list)
    for split in splits:
        with gzip.open(os.path.join(DATA_DIR, split + '.jsonl.gz'), 'rt', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                if item['item_id'] in wanted:
                    groups[(item['review_pmcid'], item['source_pmcid'])].append(item)
    found = sum(len(v) for v in groups.values())
    if found != len(wanted):
        raise SystemExit(f'sample names {len(wanted)} items but the release has {found}')

    instances = []
    for (review, source), items in sorted(groups.items()):
        by_question = collections.defaultdict(list)
        for it in sorted(items, key=lambda it: it['item_id']):
            by_question[it['question']].append(it['item_id'])
        questions = [{'question_id': f'q{i}', 'question': text, 'item_ids': ids}
                     for i, (text, ids) in enumerate(by_question.items(), 1)]
        instances.append({'instance_id': f'{review}__{source}', 'review_pmcid': review,
                          'source_pmcid': source, 'split': items[0]['split'],
                          'n_questions': len(questions), 'n_items': len(items),
                          'questions': questions})
    return instances


def assemble(inst, src, answers):
    """Merge model answers into the instance record, verifying every quote (v1 logic).

    A schema-constrained reply can still hold a malformed entry; those are counted
    and dropped, and the questions they should have answered fall through to
    missing_from_response.
    """
    by_id, n_malformed = {}, 0
    for a in answers or []:
        if not isinstance(a, dict):
            n_malformed += 1
            continue
        qid = (a.get('question_id') or '').strip().lower()
        if qid and qid not in by_id:
            by_id[qid] = a

    out_questions = []
    for q in inst['questions']:
        a = by_id.get(q['question_id'].lower())
        if a is None:
            pred = {'answered': False, 'missing_from_response': True, 'not_reported': None,
                    'answer': None, 'provenance': None}
        else:
            answer = (a.get('answer') or '').strip()
            not_reported = bool(a.get('not_reported')) or answer == ''
            quote = (a.get('evidence_quote') or '').strip()
            claimed = (a.get('evidence_section') or '').strip()
            prov = None
            if not not_reported:
                located = locate_quote(src, quote) if quote else None
                prov = {'claimed_section_id': claimed, 'quote': quote,
                        **(located or {'quote_found': False, 'located_section_id': None,
                                       'located_section_title': None, 'char_start': None,
                                       'char_end': None})}
                prov['section_agrees'] = (prov['located_section_id'] is not None
                                          and prov['located_section_id'].lower() == claimed.lower())
            pred = {'answered': True, 'missing_from_response': False,
                    'not_reported': not_reported, 'answer': '' if not_reported else answer,
                    'provenance': prov}
        out_questions.append({**q, 'prediction': pred})

    return {
        'instance_id': inst['instance_id'], 'review_pmcid': inst['review_pmcid'],
        'source_pmcid': inst['source_pmcid'], 'split': inst['split'],
        'source_title': src['title'], 'source_chars': src['n_chars'],
        'source_truncated': src['truncated'], 'source_layout': src.get('layout'),
        'source_tables_found': src.get('n_tables_found'),
        'source_tables_kept': src.get('n_tables_kept'),
        'source_sections': [{'id': s['id'], 'title': s['title'], 'zone': s.get('zone'),
                             'n_chars': len(s['text'])} for s in src['sections']],
        'n_questions': inst['n_questions'], 'n_items': inst['n_items'],
        'malformed_answer_entries': n_malformed,
        'questions': out_questions,
    }


# --------------------------------------------------------------------------- #
# the request
# --------------------------------------------------------------------------- #
def model_settings(snapshot, model):
    """Approved provider tags, smallest context, whether reasoning can be switched off."""
    entry = snapshot['models'].get(model)
    if not entry or not entry['approved']:
        raise SystemExit(f'{model}: no approved provider in providers.json')
    approved = entry['approved']
    flags = {a['reasoning_param'] for a in approved}
    if len(flags) != 1:
        raise SystemExit(f'{model}: approved providers disagree on the reasoning parameter')
    return {'tags': [a['tag'] for a in approved],
            'min_context': min(a['context'] for a in approved),
            'reasoning_param': flags.pop(),
            'price_in_per_m': max(a['price_in_per_m'] for a in approved),
            'price_out_per_m': max(a['price_out_per_m'] for a in approved)}


def request_body(model, settings, user, temperature, max_tokens):
    body = {
        'model': model,
        'messages': [{'role': 'system', 'content': SYSTEM_PROMPT},
                     {'role': 'user', 'content': user}],
        'max_tokens': max_tokens,
        'temperature': temperature,
        'response_format': {'type': 'json_schema', 'json_schema': {
            'name': 'submit_answers', 'strict': True, 'schema': ANSWER_SCHEMA}},
        'provider': {'only': settings['tags'], 'allow_fallbacks': False},
        'usage': {'include': True},
    }
    if settings['reasoning_param']:
        # Reasoning off for every model (agreed 2026-10-09). Both spellings are sent
        # because providers disagree on which they honour (v1).
        body['reasoning'] = {'enabled': False}
        body['chat_template_kwargs'] = {'enable_thinking': False}
    return body


def call(body, key, timeout, attempts=6):
    """One chat completion. Retries transport/5xx/429 with backoff; other 4xx fail fast.

    Rate limits (429) need long waits: in the layout test, Llama 4 Scout's single
    approved provider throttled two parallel runs and a 2-8 s backoff lost 7 of 102
    papers. Honour Retry-After when sent, else wait 5, 10, 20, 40, 60 s.
    """
    payload = json.dumps(body).encode()
    last = None
    for i in range(attempts):
        req = urllib.request.Request(API, data=payload, headers={
            'Authorization': f'Bearer {key}', 'Content-Type': 'application/json'})
        wait = min(60, 5 * 2 ** i)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            detail = (e.read() or b'').decode(errors='replace')[:300]
            last = f'HTTP {e.code}: {detail}'
            # 402 'in_flight_budget_exhausted' is temporary: OpenRouter reserves each
            # running request's maximum cost against the balance (group 1, 2026-10-09)
            transient = e.code == 429 or (e.code == 402 and 'in_flight' in detail)
            if e.code < 500 and not transient:
                return {'_error': last}
            try:
                wait = max(wait, float(e.headers.get('Retry-After') or 0))
            except ValueError:
                pass
        except Exception as e:  # noqa: BLE001 - transport, DNS, timeout
            last = f'{type(e).__name__}: {e}'
        if i < attempts - 1:
            time.sleep(wait)
    return {'_error': last}


def openrouter_balance(key):
    """Remaining OpenRouter credit in USD (total credits minus total usage)."""
    req = urllib.request.Request('https://openrouter.ai/api/v1/credits',
                                 headers={'Authorization': f'Bearer {key}'})
    with urllib.request.urlopen(req, timeout=60) as r:
        data = json.loads(r.read())['data']
    return float(data['total_credits']) - float(data['total_usage'])


def parse_answers(res):
    try:
        text = (res['choices'][0]['message'] or {}).get('content')
        return (json.loads(text) or {}).get('answers', []) if text else []
    except Exception:  # noqa: BLE001
        return []


def usage_of(res):
    u = res.get('usage') or {}
    details = u.get('completion_tokens_details') or {}
    choice = (res.get('choices') or [{}])[0]
    return {'in': int(u.get('prompt_tokens') or 0), 'out': int(u.get('completion_tokens') or 0),
            'reasoning': int(details.get('reasoning_tokens') or 0),
            'cost': float(u.get('cost') or 0),
            'finish_reason': choice.get('finish_reason')}


def failed_record(r):
    """No provider served it, or the reply held no usable answer at all (a reply cut
    off at max_tokens is unparseable JSON; an empty reply has nothing)."""
    return not r.get('served_by') or all(q['prediction']['missing_from_response']
                                         for q in r['questions'])


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', required=True, help='OpenRouter slug from the roster')
    ap.add_argument('--sample', default=DEFAULT_SAMPLE)
    # linear = document order, tables at the end: chosen by the layout test (2026-10-09)
    ap.add_argument('--layout', choices=('sectioned', 'linear'), default='linear')
    ap.add_argument('--tag', default=None, help='output folder name (default: from sample + layout)')
    ap.add_argument('--limit', type=int, default=0, help='first N instances (smoke tests)')
    ap.add_argument('--random-limit', action='store_true',
                    help='with --limit: a seeded random N instead of the first N')
    # v1 used 4,096; the full Qwen3.5-27B run (2026-10-09) showed papers with 42-69
    # questions need more - 7 replies were cut off there and lost. Tokens are billed
    # as used, so a higher cap costs nothing on normal replies.
    ap.add_argument('--max-tokens', type=int, default=16384)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--timeout', type=int, default=900)
    ap.add_argument('--retry-coverage', type=float, default=0.5)
    ap.add_argument('--retry-temp', type=float, default=0.5)
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--retry-failed', action='store_true',
                    help='re-send only instances no provider answered, or whose reply held no '
                         'usable answer (see failed_record); implies --resume; the failed '
                         'records are moved to raw_failed.jsonl')
    ap.add_argument('--dry-run', action='store_true', help='build every request, send none')
    ap.add_argument('--show-prompt', action='store_true', help='with --dry-run: print one prompt')
    args = ap.parse_args()

    settings = model_settings(providers.load(), args.model)
    key = os.environ.get('OPENROUTER_API_KEY', '').strip()
    if not key and not args.dry_run:
        sys.exit('OPENROUTER_API_KEY is not set')

    instances = load_instances(load_sample(args.sample))
    if args.limit:
        if args.random_limit:
            instances = sorted(random.Random(20261009).sample(instances, args.limit),
                               key=lambda i: i['instance_id'])
        else:
            instances = instances[:args.limit]

    sample_name = os.path.splitext(os.path.basename(args.sample))[0]
    tag = args.tag or f'{sample_name}_{args.layout}'
    out_dir = os.path.join(OUT_ROOT, tag, args.model.split('/')[-1])
    raw_path = os.path.join(out_dir, 'raw.jsonl')
    out_path = os.path.join(out_dir, 'answers.json')

    if args.retry_failed and os.path.exists(raw_path) and not args.dry_run:
        # keep answered records; set aside the failed ones, to be re-sent
        with open(raw_path, encoding='utf-8') as f:
            records = [json.loads(line) for line in f if line.strip()]
        kept = [r for r in records if not failed_record(r)]
        failed = [r for r in records if failed_record(r)]
        with open(os.path.join(out_dir, 'raw_failed.jsonl'), 'a', encoding='utf-8', newline='\n') as f:
            for r in failed:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
        with open(raw_path, 'w', encoding='utf-8', newline='\n') as f:
            for r in kept:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
        print(f'retry-failed: {len(failed)} failed instance(s) set aside for re-sending')
        args.resume = True

    done = set()
    if args.resume and os.path.exists(raw_path):
        with open(raw_path, encoding='utf-8') as f:
            done = {json.loads(line)['instance_id'] for line in f if line.strip()}
        instances = [i for i in instances if i['instance_id'] not in done]

    ctx = {}
    with open(REVIEW_CONTEXT, encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            ctx[r['review_pmcid']] = r

    print(f'model      : {args.model}  providers: {", ".join(settings["tags"])}')
    print(f'reasoning  : {"off (sent)" if settings["reasoning_param"] else "no reasoning mode"}')
    print(f'sample     : {args.sample}  layout: {args.layout}  out: {out_dir}')
    print(f'instances  : {len(instances)}  questions: {sum(i["n_questions"] for i in instances):,}  '
          f'items: {sum(i["n_items"] for i in instances):,}' + (f'  (resume: {len(done)} done)' if done else ''))

    # ---- build every request, and bound its size before anything is sent ---- #
    built, sizes = [], []
    limit = settings['min_context']
    for inst in instances:
        src = jats_source.parse_source(os.path.join(SOURCE_DIR, inst['source_pmcid'] + '.xml'),
                                       MAX_SOURCE_CHARS, layout=args.layout)
        user = build_prompt(inst, ctx.get(inst['review_pmcid']), jats_source.render_source(src))
        chars = len(SYSTEM_PROMPT) + len(user)
        bound = int(chars * TOKENS_PER_CHAR_BOUND) + args.max_tokens
        if bound > limit:
            sys.exit(f'{inst["instance_id"]}: request bound {bound:,} tokens exceeds the '
                     f'smallest approved context {limit:,}; nothing was sent')
        sizes.append(chars)
        built.append((inst, src, user))
    if not built:
        print('nothing to do')
        return

    s = sorted(sizes)
    pick = lambda q: s[min(int(len(s) * q), len(s) - 1)]  # noqa: E731
    est_in = sum(sizes) * TOKENS_PER_CHAR_EST
    est_out = len(built) * OUTPUT_TOKENS_PER_INSTANCE_EST
    est_cost = (est_in * settings['price_in_per_m'] + est_out * settings['price_out_per_m']) / 1e6
    print(f'request chars: p50 {pick(.5):,}  p90 {pick(.9):,}  max {s[-1]:,}  '
          f'(bound {int(s[-1] * TOKENS_PER_CHAR_BOUND) + args.max_tokens:,} tokens '
          f'vs smallest approved context {limit:,})')
    print(f'truncated sources: {sum(1 for _, src, _ in built if src["truncated"])}')
    print(f'estimate   : ~{est_in / 1e6:.2f}M input + ~{est_out / 1e6:.2f}M output tokens, '
          f'~${est_cost:.2f} at the dearest approved provider')

    if not args.dry_run:
        # never start a run the account cannot pay for (group 1 ran the balance to
        # zero mid-run on 2026-10-09): require the estimate plus 20%
        balance = openrouter_balance(key)
        print(f'balance    : ${balance:.2f} OpenRouter credit')
        if balance < 1.2 * est_cost:
            sys.exit(f'stopping: balance ${balance:.2f} is below the estimate plus 20% '
                     f'(${1.2 * est_cost:.2f}); top up OpenRouter first')

    if args.dry_run:
        if args.show_prompt:
            inst, _, user = built[0]
            print(f'\n--- prompt for {inst["instance_id"]} ({len(user):,} chars) ---')
            print(user[:3000] + '\n[...]\n' + user[-1500:])
        print('\ndry run: no API calls made')
        return

    os.makedirs(out_dir, exist_ok=True)
    lock = threading.Lock()
    served, errors = collections.Counter(), []
    totals = collections.Counter()
    n_retried = n_recovered = 0

    def one(item):
        nonlocal n_retried, n_recovered
        inst, src, user = item
        res = call(request_body(args.model, settings, user, 0.0, args.max_tokens), key, args.timeout)
        answers, provider, retried, usages = [], None, False, []
        if '_error' in res:
            with lock:
                errors.append((inst['instance_id'], res['_error']))
        else:
            answers, provider = parse_answers(res), res.get('provider')
            usages.append(usage_of(res))
        # a schema-valid but empty/short reply is retried once, at temperature > 0
        # so the request is not a byte-identical repeat of the one that failed (v1)
        if len(answers) < args.retry_coverage * inst['n_questions']:
            retried = True
            res2 = call(request_body(args.model, settings, user, args.retry_temp, args.max_tokens),
                        key, args.timeout)
            if '_error' not in res2:
                usages.append(usage_of(res2))
                again = parse_answers(res2)
                if len(again) > len(answers):
                    answers, provider = again, res2.get('provider') or provider
                    with lock:
                        n_recovered += 1
        with lock:
            n_retried += retried
            served[provider or '(none)'] += 1
            for u in usages:
                totals.update({'in': u['in'], 'out': u['out'], 'reasoning': u['reasoning']})
                totals['cost_micro'] += int(u['cost'] * 1e6)
        rec = assemble(inst, src, answers)
        rec.update(served_by=provider, retried=retried, usage=usages)
        return rec

    t0 = time.time()
    with open(raw_path, 'a', encoding='utf-8', newline='\n') as raw, \
            ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(one, b) for b in built]
        for k, fut in enumerate(as_completed(futures), 1):
            rec = fut.result()
            raw.write(json.dumps(rec, ensure_ascii=False) + '\n')
            if k % 25 == 0 or k == len(futures):
                raw.flush()
                el = time.time() - t0
                print(f'  [{k}/{len(futures)}] {el / 60:.1f} min | '
                      f'${totals["cost_micro"] / 1e6:.2f} | eta {(len(futures) - k) * el / k / 60:.0f} min',
                      flush=True)

    # collate every record of this cell, including earlier resumed runs
    seen, records = set(), []
    with open(raw_path, encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            if r['instance_id'] not in seen:
                seen.add(r['instance_id'])
                records.append(r)
    records.sort(key=lambda r: r['instance_id'])

    preds = [q['prediction'] for r in records for q in r['questions']]
    answered = [p for p in preds if p['answered'] and not p['not_reported']]
    with_quote = [p for p in answered if p['provenance'] and p['provenance']['quote']]
    verified = [p for p in with_quote if p['provenance']['quote_found']]
    all_usage = [u for r in records for u in r.get('usage', [])]
    summary = {
        'model': args.model, 'sample': os.path.relpath(args.sample, BASE).replace(os.sep, '/'),
        'layout': args.layout, 'max_source_chars': MAX_SOURCE_CHARS,
        'providers_snapshot': providers.load()['snapshot_date'],
        'approved_providers': settings['tags'],
        'reasoning': 'off' if settings['reasoning_param'] else 'no reasoning mode',
        'instances': len(records), 'questions': len(preds),
        'items': sum(r['n_items'] for r in records),
        'answered_with_value': len(answered),
        'not_reported': sum(1 for p in preds if p['not_reported']),
        'missing_from_response': sum(1 for p in preds if p['missing_from_response']),
        'malformed_answer_entries': sum(r['malformed_answer_entries'] for r in records),
        'provenance': {
            'answers_with_quote': len(with_quote),
            'quotes_verified_verbatim': len(verified),
            'quote_verification_rate': round(len(verified) / len(with_quote), 4) if with_quote else None,
            'cited_section_matches_located_section': sum(
                1 for p in verified if p['provenance'].get('section_agrees')),
        },
        'served_by': dict(collections.Counter(r.get('served_by') or '(none)' for r in records)),
        'retries': {'under_covered_instances': sum(1 for r in records if r.get('retried')),
                    'coverage_threshold': args.retry_coverage, 'temperature': args.retry_temp},
        'api_errors_this_session': len(errors),
        'tokens': {'input': sum(u['in'] for u in all_usage),
                   'output': sum(u['out'] for u in all_usage),
                   'reasoning': sum(u['reasoning'] for u in all_usage)},
        'cost_usd': round(sum(u['cost'] for u in all_usage), 4),
    }
    failed_path = os.path.join(out_dir, 'raw_failed.jsonl')
    if os.path.exists(failed_path):
        with open(failed_path, encoding='utf-8') as f:
            failed_usage = [u for line in f if line.strip()
                            for u in json.loads(line).get('usage', [])]
        # what was actually spent: the kept answers plus the discarded failed attempts
        summary['failed_attempts'] = {'records': sum(1 for _ in open(failed_path, encoding='utf-8')),
                                      'cost_usd': round(sum(u['cost'] for u in failed_usage), 4)}
        summary['cost_usd_total_spent'] = round(summary['cost_usd'] + summary['failed_attempts']['cost_usd'], 4)
    with open(out_path, 'w', encoding='utf-8', newline='\n') as f:
        json.dump({'summary': summary, 'instances': records}, f, indent=1, ensure_ascii=False)
    print('\n=== summary ===')
    print(json.dumps(summary, indent=2))
    if errors:
        print(f'\n{len(errors)} instance(s) errored; first 3:')
        for iid, e in errors[:3]:
            print(f'  {iid}: {e}')


if __name__ == '__main__':
    main()
