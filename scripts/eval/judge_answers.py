"""
judge_answers.py
----------------
Stage 5.4 of the v2 pipeline: two-axis LLM-judge scoring of one answers file
(written by run_answers.py). Two separate judges, each blind to the other's
private input, so the 2x2 stays clean:

  1. FAITHFULNESS - is the answer supported by the SOURCE it cites?
       sees : question + answer + the model's cited quote + its section (<=1,600 chars)
       blind: the gold answer
       faithful = fully_supported or partially_supported (v1's definition)
  2. CORRECTNESS  - does the answer convey the same fact the review RECORDED?
       sees : question + the item's gold + answer
       blind: the source paper
       correct = match_level >= minor_deviation, applied here, never by the judge

Ported from v1 (old/scripts/judge_answers.py); the two system prompts, the two
verdict schemas, parse_verdict, section_context and the span self-audit are
unchanged. Agreed 2026-10-09 (plan 5.4):

  * judge = gemini-3.7-flash through Google's batch API only (v1 found the
    transport changes verdicts), thinking at its default (it cannot be disabled);
  * every item is judged against ITS OWN gold (v1 judged a question against the
    list of golds sharing it); only items the sample marks judge=true are sent;
  * correctness is skipped when gold_is_absence (no fact to match: an answer
    there is an over-answer, scored in the abstention table);
  * "not reported" answers never go to a judge: abstention is scored
    deterministically on every sampled item from the release's gold_is_absence;
  * pooled rates use the sample's weights (characteristic ~2.78); per-content
    rates are unweighted.

Steps (each resumable; files live next to the answers, in judge/):
  prepare   build the batch requests  -> judge/requests.jsonl, judge/items.json
  smoke     send the first N requests directly (not batched): checks the model
            id, the JSON verdicts and the token use before paying for a batch
  submit    send judge/requests.jsonl as one Gemini batch (half price)
  status    batch progress
  topup     judge DIRECTLY any request the batch has no usable verdict for (items
            answered after it was sent; its unparseable verdicts); then re-run collect
  collect   -> judge/judged.json: per-item verdicts, self-audit, 2x2,
            abstention table, rates (per content + weighted pooled)

Key: GEMINI_API_KEY from the environment (the Windows user-level key; the newer
key in .env has no prepaid credit - see 16_design_labels.py).

Usage:
  python scripts/eval/judge_answers.py prepare --answers data/v2/eval/answers/<tag>/<model>/answers.json
  python scripts/eval/judge_answers.py smoke   --answers ... --n 6
  python scripts/eval/judge_answers.py submit  --answers ...
  python scripts/eval/judge_answers.py status  --answers ...
  python scripts/eval/judge_answers.py collect --answers ...
"""
import argparse
import collections
import gzip
import importlib.util
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(os.path.dirname(HERE))
DATA_DIR = os.path.join(BASE, 'release', 'v2', 'data')
SOURCE_DIR = os.path.join(BASE, 'SOURCE_XML')

MODEL = 'gemini-3.7-flash'
MAX_OUTPUT_TOKENS = 4096        # Gemini counts thinking tokens toward this cap
SECTION_CHARS = 1600            # v1's cap on the section shown to the faithfulness judge
CORRECT_THRESHOLD = 3           # match_level >= minor_deviation


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


jats_source = _load('jats_source', os.path.join(HERE, 'jats_source.py'))
run_answers = _load('run_answers', os.path.join(HERE, 'run_answers.py'))

try:
    sys.stdout.reconfigure(encoding='utf-8')
except (AttributeError, ValueError):
    pass

# --------------------------------------------------------------------------- #
# Judge 1: FAITHFULNESS (answer <- source; blind to gold) - v1, unchanged
# --------------------------------------------------------------------------- #
FAITH_SYSTEM = (
    "You check whether a model's answer to a systematic-review data-extraction "
    "question is GROUNDED in the source study it was shown.\n\n"
    "You are given: the QUESTION, the model's ANSWER, a quote the model cited from "
    "the source (VERBATIM unless flagged otherwise), and the surrounding section.\n\n"
    "You are NOT given any gold/reference answer. Grounding is about support by the "
    "evidence, not about whether the value is 'right' in any external sense - do not "
    "guess a correct value. Use the question only to interpret what the answer is "
    "claiming and to check the cited evidence is actually about what the question "
    "asks; do NOT rate answer quality.\n\n"
    "Decide evidence_supports_answer:\n"
    "  fully_supported     - the cited evidence clearly states or directly entails "
    "the answer, and the evidence is on-topic for the question.\n"
    "  partially_supported - the evidence supports part of the answer, or is on-topic "
    "but weaker than the answer claims.\n"
    "  unsupported         - the evidence does not support the answer, is off-topic "
    "for the question, or (if flagged) the quote was not found in the source.\n"
    "  contradicted        - the evidence states something that conflicts with the answer.\n"
    "Set answer_adds_unsupported_detail true if the answer asserts specific facts "
    "(numbers, names, qualifiers) that the cited evidence does not contain.\n\n"
    "IMPORTANT - source papers are sometimes internally inconsistent (e.g. the "
    "abstract and a table say 6/61 while one sentence in the results says 5/61, where "
    "only one can match the stated percentage). If the answer matches ONE of the "
    "values the source gives and the source disagrees with itself, set "
    "source_internally_inconsistent true and judge support against the value the "
    "source states most consistently - do NOT mark the answer contradicted merely "
    "because it disagrees with one erroneous mention.\n"
    "Finally, return relied_on_span: copy VERBATIM the shortest span of the cited quote "
    "or the surrounding section that your grounding verdict turned on. It is checked "
    "automatically against the evidence you were shown, so it must be text you can see, "
    "not a paraphrase.\n"
    "Return your verdict as a single JSON object matching the required schema."
)

FAITH_SCHEMA = {
    "name": "judge_faithfulness",
    "description": "Grounding verdict for one answer against its cited source evidence.",
    "json_schema": {
        "type": "object",
        "properties": {
            "evidence_supports_answer": {
                "type": "string",
                "enum": ["fully_supported", "partially_supported", "unsupported", "contradicted"],
            },
            "answer_adds_unsupported_detail": {"type": "boolean"},
            "source_internally_inconsistent": {
                "type": "boolean",
                "description": "True if the source states this fact inconsistently in different places.",
            },
            "relied_on_span": {
                "type": "string",
                "description": ("VERBATIM shortest span of the cited quote or surrounding "
                                "section this verdict turned on; checked against the evidence shown."),
            },
            "rationale": {"type": "string", "description": "One sentence, <=300 chars."},
        },
        "required": ["evidence_supports_answer", "answer_adds_unsupported_detail",
                     "source_internally_inconsistent", "relied_on_span", "rationale"],
    },
}

# --------------------------------------------------------------------------- #
# Judge 2: CORRECTNESS (answer <-> gold; blind to source) - v1, unchanged
# --------------------------------------------------------------------------- #
CORRECT_SYSTEM = (
    "You check whether a model's answer conveys the SAME underlying fact as the "
    "reference (gold) value a systematic review recorded for a data-extraction "
    "question.\n\n"
    "You are given: the QUESTION, the gold value(s), and the model's ANSWER. You are "
    "NOT given the source paper - judge only against the gold.\n\n"
    "Gold cells are terse, author-chosen shorthand; the model may answer in prose or "
    "at a different granularity. Credit a match when the answer and the gold refer to "
    "the same fact even if wording, format, or granularity differ. Interpret both "
    "relative to the question.\n\n"
    "Choose relation (pick the single best fit):\n"
    "  equivalent          - same fact, same granularity (allowing format/wording), "
    "e.g. gold 'USA' vs 'United States'; gold 'hiPSC-CMs' vs the spelled-out name.\n"
    "  model_more_specific - answer is a FINER-GRAINED version of THE SAME fact "
    "(e.g. gold 'Geriatric', answer '55-85 years'; gold 'China', answer 'China (Tianjin)').\n"
    "  model_broader       - answer covers the gold's fact but ALSO adds other things "
    "the gold did not record (e.g. gold 'Serum', answer 'culture media and serum'). "
    "This is a superset, NOT more_specific - do not confuse the two.\n"
    "  model_less_specific - answer is incomplete relative to the gold: it drops "
    "detail the gold records (e.g. gold lists 19 symptoms with percentages, answer "
    "gives 2). Use this whenever the answer omits substantive gold content.\n"
    "  different_selection - both are plausible values for this field but describe "
    "different rows/arms/subgroups/measures (e.g. gold reports one arm, answer another).\n"
    "  contradictory       - answer conflicts with the gold.\n"
    "  unrelated           - answer does not address the fact the gold records.\n"
    "If several gold values are given, match against any and report matched_gold_index "
    "(0-based; -1 if none).\n\n"
    "ALSO rate match_level. This measures ONE thing only: HOW MUCH of the gold's "
    "content the answer recovers. It is not about what kind of difference there is - "
    "`relation` already records that. In particular, extra material the answer adds is "
    "NOT penalised here (that is what model_broader is for); only missing or "
    "incompatible content lowers the level. Use these anchors:\n"
    "  complete        - the gold's fact is fully conveyed; nothing material missing. "
    "(gold 'USA' / answer 'United States'; gold 'hiPSC-CMs' / answer spelled out in "
    "full; gold 'Serum' / answer 'culture media and serum' - the gold's fact is all "
    "there, the surplus is recorded by relation, not here)\n"
    "  minor_deviation - the gold's fact is conveyed with a small immaterial "
    "difference: rounding, units, or a dropped qualifier. (gold '121.5' / answer "
    "'121.47 ug/L'; gold 'miR-194-3p(down)' / answer 'miR-194-3p')\n"
    "  partial         - only SOME of the gold's content is present; substantive gold "
    "content is missing. (gold lists 19 symptoms with percentages / answer gives 2)\n"
    "  wrong_value     - the answer addresses the SAME attribute the gold records, but "
    "the value is incompatible: a different number, a different arm/subgroup, a "
    "different measure, or a direct contradiction. (gold '1048' / answer '1,068'; "
    "gold 'China' / answer 'Japan')\n"
    "  no_match        - the answer does not address the gold's attribute at all: it "
    "answers a different field, or is too vague to constitute a value. (gold '1048' / "
    "answer 'patients were recruited from three hospitals')\n\n"
    "Judge relation and match_level only. Whether a level counts as 'correct' is decided "
    "downstream by a threshold, so do not optimise for that.\n\n"
    "ALSO emit failure_mechanism - the KIND of genuine difference, a sub-label that "
    "refines relation. Set it to \"not_applicable\" UNLESS relation is different_selection "
    "or contradictory.\n"
    "  If relation is different_selection, choose one:\n"
    "    WRONG_ARM         - gold reports one arm/group, the answer another.\n"
    "    WRONG_TIMEPOINT   - a different follow-up or measurement occasion.\n"
    "    WRONG_COHORT      - a different population, site, or subgroup within the paper.\n"
    "    AGGREGATION_LEVEL - pooled vs per-arm, total vs per-group (e.g. a pooled sample "
    "size vs one group's).\n"
    "  If relation is contradictory, choose one:\n"
    "    UNIT_OR_SCALE       - right quantity, wrong unit/scale/denominator (% vs n, mg vs g).\n"
    "    DISPERSION_CONFUSION- SD reported as SE, CI as IQR, and the like.\n"
    "    TRANSCRIPTION       - digits wrong or rounding beyond tolerance (gold '1048' vs '1068').\n"
    "    SUBSTANTIVE         - a genuinely different fact, none of the above.\n"
    "  Use OTHER only when relation is one of those two but no code fits.\n"
    "Finally, return relied_on_span: copy VERBATIM the shortest span of the GOLD value(s) "
    "or the ANSWER that your verdict turned on. It is checked automatically against those "
    "strings, so it must be text that appears in them, not a paraphrase.\n"
    "Return your verdict as a single JSON object matching the required schema."
)

CORRECT_SCHEMA = {
    "name": "judge_correctness",
    "description": "Same-fact verdict for one answer against the gold value(s).",
    "json_schema": {
        "type": "object",
        "properties": {
            "relation": {
                "type": "string",
                "enum": ["equivalent", "model_more_specific", "model_broader",
                         "model_less_specific", "different_selection", "contradictory",
                         "unrelated"],
            },
            "match_level": {
                "type": "string",
                "enum": ["complete", "minor_deviation", "partial", "wrong_value", "no_match"],
                "description": (
                    "How much of the GOLD'S CONTENT the answer recovers; extra material "
                    "is not penalised here. complete = gold's fact fully conveyed "
                    "(including when the answer adds more); minor_deviation = conveyed "
                    "with rounding/units/dropped qualifier; partial = substantive gold "
                    "content missing; wrong_value = same attribute but incompatible "
                    "value (incl. contradiction); no_match = does not address the gold's "
                    "attribute at all."
                ),
            },
            "matched_gold_index": {"type": "integer", "description": "0-based; -1 if none."},
            "failure_mechanism": {
                "type": "string",
                "enum": ["WRONG_ARM", "WRONG_TIMEPOINT", "WRONG_COHORT", "AGGREGATION_LEVEL",
                         "UNIT_OR_SCALE", "DISPERSION_CONFUSION", "TRANSCRIPTION",
                         "SUBSTANTIVE", "OTHER", "not_applicable"],
                "description": ("The KIND of genuine difference; sub-label refining `relation`. "
                                "\"not_applicable\" unless relation is different_selection or "
                                "contradictory."),
            },
            "relied_on_span": {
                "type": "string",
                "description": ("VERBATIM shortest span of the GOLD value(s) or the ANSWER this "
                                "verdict turned on; checked against those strings."),
            },
            "rationale": {"type": "string", "description": "One sentence, <=300 chars."},
        },
        "required": ["relation", "match_level", "matched_gold_index", "failure_mechanism",
                     "relied_on_span", "rationale"],
    },
}

MATCH_LEVELS = {"complete": 4, "minor_deviation": 3, "partial": 2, "wrong_value": 1, "no_match": 0}
FAITHFUL = ("fully_supported", "partially_supported")


def is_faithful(verdict):
    return bool(verdict) and verdict['evidence_supports_answer'] in FAITHFUL


def is_correct(verdict, threshold=CORRECT_THRESHOLD):
    return bool(verdict) and MATCH_LEVELS.get(verdict.get('match_level'), -1) >= threshold


def parse_verdict(text, schema):
    """Parse a verdict and enforce the schema's required keys and enums (v1).
    Never accept an enum value the rubric does not define."""
    try:
        v = json.loads(text or '')
    except ValueError as e:
        raise ValueError(f"{schema['name']}: unparseable JSON ({e})")
    if not isinstance(v, dict):
        raise ValueError(f"{schema['name']}: expected a JSON object, got {type(v).__name__}")
    js = schema['json_schema']
    missing = [k for k in js['required'] if k not in v]
    if missing:
        raise ValueError(f"{schema['name']}: missing keys {missing}")
    for key, spec in js['properties'].items():
        if 'enum' in spec and v.get(key) not in spec['enum']:
            raise ValueError(f"{schema['name']}: {key}={v.get(key)!r} not in enum")
    return v


def _norm(s):
    return re.sub(r'\s+', ' ', (s or '')).strip().lower()


def span_verified(verdict, reference):
    """Judge self-audit: was the judge's cited span in the material it was shown?
    A span that is not is invented evidence. None when there is no span."""
    if not verdict:
        return None
    span = _norm(verdict.get('relied_on_span'))
    if not span:
        return None
    return span in _norm(reference)


def section_context(src, prov, max_chars=SECTION_CHARS):
    """Text of the section the quote lives in (or the claimed one), quote centred (v1).
    The quote is re-located in the freshly parsed source; section ids are positional,
    so they are only a fallback."""
    if src is None:
        return '', None
    sec_id = prov.get('located_section_id') or prov.get('claimed_section_id')
    start = prov.get('char_start')
    relocated = run_answers.locate_quote(src, prov.get('quote') or '')
    if relocated['quote_found']:
        sec_id, start = relocated['located_section_id'], relocated['char_start']
    section = next((s for s in src['sections'] if s['id'] == sec_id), None)
    if section is None:
        return '', sec_id
    text = section['text']
    if isinstance(start, int) and len(text) > max_chars:
        lo = max(0, start - max_chars // 2)
        text = ('...' if lo else '') + text[lo:lo + max_chars] + '...'
    elif len(text) > max_chars:
        text = text[:max_chars] + '...'
    return text, sec_id


def build_faith_prompt(question, answer, prov, context, section_id):
    note = '' if prov.get('quote_found') else (
        "\n\n[NOTE: the model's cited quote could NOT be located verbatim in the "
        "source. Weigh the surrounding section, but a missing quote is itself a "
        "grounding failure.]")
    return (f"QUESTION:\n{question}\n\n"
            f"MODEL ANSWER:\n{answer}\n\n"
            f"MODEL'S CITED QUOTE:\n\"{prov.get('quote') or ''}\"\n\n"
            f"SURROUNDING SOURCE SECTION [{section_id}]:\n"
            f"{context or '(section text unavailable)'}"
            f"{note}")


def build_correct_prompt(question, gold, answer):
    return (f"QUESTION:\n{question}\n\n"
            f"GOLD VALUE(S) (the review recorded one of these):\n  [0] {gold!r}\n\n"
            f"MODEL ANSWER:\n{answer}")


def generation_config(schema):
    """Thinking is left at the model's default: it cannot be disabled on 3.7 Flash."""
    return {'temperature': 0.0, 'max_output_tokens': MAX_OUTPUT_TOKENS,
            'response_mime_type': 'application/json',
            'response_json_schema': schema['json_schema']}


# --------------------------------------------------------------------------- #
# items: the answers file joined to the release and the sample
# --------------------------------------------------------------------------- #
def paths(answers_path):
    work = os.path.join(os.path.dirname(os.path.abspath(answers_path)), 'judge')
    return {'dir': work, 'requests': os.path.join(work, 'requests.jsonl'),
            'items': os.path.join(work, 'items.json'), 'state': os.path.join(work, 'batch_state.json'),
            'smoke': os.path.join(work, 'smoke.json'), 'judged': os.path.join(work, 'judged.json'),
            'results': os.path.join(work, 'batch_results.jsonl'),
            'topup': os.path.join(work, 'topup.jsonl'), 'topup_refs': os.path.join(work, 'topup_refs.json')}


def load_items(answers_path):
    """One record per sampled ITEM: its answer (shared by items with the same question
    in an instance), its own gold, and the sample's judge flag and weight."""
    with open(answers_path, encoding='utf-8') as f:
        answers = json.load(f)
    sample_path = os.path.join(BASE, answers['summary']['sample'])
    with open(sample_path, encoding='utf-8') as f:
        sample = {r['item_id']: r for r in map(json.loads, f)}
    gold = {}
    for split in sorted({r['split'] for r in sample.values()}):
        with gzip.open(os.path.join(DATA_DIR, split + '.jsonl.gz'), 'rt', encoding='utf-8') as f:
            for line in f:
                it = json.loads(line)
                if it['item_id'] in sample:
                    gold[it['item_id']] = it
    items = []
    for inst in answers['instances']:
        for q in inst['questions']:
            for item_id in q['item_ids']:
                rel, s = gold[item_id], sample[item_id]
                items.append({'item_id': item_id, 'instance_id': inst['instance_id'],
                              'source_pmcid': inst['source_pmcid'], 'question': q['question'],
                              'gold': rel['gold'], 'gold_is_absence': rel['gold_is_absence'],
                              'content': rel['content'], 'judge': s['judge'], 'weight': s['weight'],
                              'prediction': q['prediction']})
    return answers['summary'], items


def abstention_verdict(item):
    """Deterministic, from gold_is_absence (no judge)."""
    p = item['prediction']
    if not p['answered']:
        return 'missing_from_response'
    absent, abstained = item['gold_is_absence'], p['not_reported']
    return ('correct_abstention' if absent and abstained else
            'over_answer' if absent and not abstained else
            'missed_value' if not absent and abstained else 'answered')


def build_requests(summary, items):
    """One batch line per (item, judge), keyed '<item_id>|F' / '<item_id>|C'."""
    layout = summary['layout']
    sources, lines, refs = {}, [], {}

    def line(key, system, schema, text):
        return {'key': key, 'request': {
            'contents': [{'role': 'user', 'parts': [{'text': text}]}],
            'system_instruction': {'parts': [{'text': system}]},
            'generation_config': generation_config(schema)}}

    for it in items:
        p = it['prediction']
        if not it['judge'] or not p['answered'] or p['not_reported']:
            continue
        pmcid = it['source_pmcid']
        if pmcid not in sources:
            sources[pmcid] = jats_source.parse_source(os.path.join(SOURCE_DIR, pmcid + '.xml'),
                                                      run_answers.MAX_SOURCE_CHARS, layout=layout)
        prov = p['provenance'] or {}
        context, sec_id = section_context(sources[pmcid], prov)
        lines.append(line(f"{it['item_id']}|F", FAITH_SYSTEM, FAITH_SCHEMA,
                          build_faith_prompt(it['question'], p['answer'], prov, context, sec_id)))
        refs[it['item_id']] = {'faith_ref': (prov.get('quote') or '') + '\n' + (context or ''),
                               'correct_ref': f"{it['gold']} {p['answer']}"}
        if not it['gold_is_absence']:
            lines.append(line(f"{it['item_id']}|C", CORRECT_SYSTEM, CORRECT_SCHEMA,
                              build_correct_prompt(it['question'], it['gold'], p['answer'])))
    return lines, refs


# --------------------------------------------------------------------------- #
# Gemini
# --------------------------------------------------------------------------- #
def gemini_client():
    key = os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY')
    if not key:
        sys.exit('no GEMINI_API_KEY in the environment')
    from google import genai
    return genai.Client(api_key=key)


def response_text(response):
    """Non-thought text of a raw GenerateContentResponse dict from a batch result."""
    try:
        parts = response['candidates'][0]['content']['parts']
    except (KeyError, IndexError, TypeError):
        return ''
    return ''.join(p.get('text', '') for p in parts if not p.get('thought'))


def job_state(job):
    return (job.state.name if job.state else '?').rsplit('STATE_', 1)[-1]


SCHEMAS = {'F': FAITH_SCHEMA, 'C': CORRECT_SCHEMA}


# --------------------------------------------------------------------------- #
# commands
# --------------------------------------------------------------------------- #
def cmd_prepare(args):
    p = paths(args.answers)
    summary, items = load_items(args.answers)
    lines, refs = build_requests(summary, items)
    os.makedirs(p['dir'], exist_ok=True)
    with open(p['requests'], 'w', encoding='utf-8', newline='\n') as f:
        for ln in lines:
            f.write(json.dumps(ln, ensure_ascii=False) + '\n')
    with open(p['items'], 'w', encoding='utf-8', newline='\n') as f:
        json.dump({'answers': os.path.relpath(args.answers, BASE).replace(os.sep, '/'),
                   'model_judged': summary['model'], 'judge_model': MODEL, 'refs': refs}, f)
    n_f = sum(1 for ln in lines if ln['key'].endswith('|F'))
    n_c = len(lines) - n_f
    chars = sum(len(ln['request']['contents'][0]['parts'][0]['text'])
                + len(ln['request']['system_instruction']['parts'][0]['text']) for ln in lines)
    # v1 measured ~2,200 input + ~950 output (thinking included) tokens per judged item
    # (both axes); priced at gemini-3.7-flash batch rates ($0.375 / $1.875 per M)
    est = n_f * (2200 * 0.375 + 950 * 1.875) / 1e6
    abst = collections.Counter(abstention_verdict(it) for it in items)
    print(f'answers    : {summary["model"]}  ({summary["layout"]}, {summary["sample"]})')
    print(f'items      : {len(items):,} sampled, {sum(it["judge"] for it in items):,} marked for judging')
    print(f'abstention : ' + ', '.join(f'{k} {v:,}' for k, v in abst.most_common()))
    print(f'requests   : {len(lines):,} ({n_f:,} faithfulness, {n_c:,} correctness); '
          f'{chars / max(len(lines), 1):,.0f} chars each on average')
    print(f'estimate   : ~${est:.2f} at batch prices (v1 token rates)')
    print(f'-> {os.path.relpath(p["requests"], BASE)}')
    if args.show and lines:
        for key_suffix in ('|F', '|C'):
            ln = next((x for x in lines if x['key'].endswith(key_suffix)), None)
            if ln:
                print(f"\n--- {ln['key']} ---\n" + ln['request']['contents'][0]['parts'][0]['text'])


def cmd_smoke(args):
    """The first N requests, sent directly: verdicts, token use, a cost per item."""
    p = paths(args.answers)
    with open(p['requests'], encoding='utf-8') as f:
        lines = [json.loads(x) for x in f][:args.n]
    c = gemini_client()
    out, tokens = [], collections.Counter()
    for ln in lines:
        req = ln['request']
        resp = c.models.generate_content(
            model=MODEL, contents=req['contents'][0]['parts'][0]['text'],
            config={'system_instruction': req['system_instruction']['parts'][0]['text'],
                    **req['generation_config']})
        u = resp.usage_metadata
        used = {'in': u.prompt_token_count or 0, 'out': u.candidates_token_count or 0,
                'thinking': u.thoughts_token_count or 0}
        tokens.update(used)
        verdict = parse_verdict(resp.text, SCHEMAS[ln['key'][-1]])
        out.append({'key': ln['key'], 'verdict': verdict, 'tokens': used})
        print(f"{ln['key']}: {json.dumps(verdict, ensure_ascii=False)[:300]}  tokens {used}")
    with open(p['smoke'], 'w', encoding='utf-8', newline='\n') as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    n = max(len(lines), 1)
    print(f'per request: in {tokens["in"] / n:,.0f}, out {tokens["out"] / n:,.0f}, '
          f'thinking {tokens["thinking"] / n:,.0f} tokens')


def cmd_submit(args):
    p = paths(args.answers)
    if os.path.exists(p['state']):
        sys.exit(f'already submitted: see {p["state"]}')
    c = gemini_client()
    uploaded = c.files.upload(file=p['requests'],
                              config={'display_name': 'judge_requests', 'mime_type': 'jsonl'})
    job = c.batches.create(model=MODEL, src=uploaded.name,
                           config={'display_name': 'srqa-v2-judge-' + os.path.basename(os.path.dirname(p['dir']))})
    with open(p['state'], 'w', encoding='utf-8', newline='\n') as f:
        json.dump({'batch_id': job.name, 'model': MODEL,
                   'submitted': time.strftime('%Y-%m-%d %H:%M')}, f)
    print(f'submitted {job.name}')


def cmd_status(args):
    with open(paths(args.answers)['state'], encoding='utf-8') as f:
        state = json.load(f)
    c = gemini_client()          # keep a reference: a discarded client closes its connection
    print(state['batch_id'], job_state(c.batches.get(name=state['batch_id'])))


def batch_results(p, state):
    """The batch's raw result lines, downloaded once and kept in judge/ so collect can
    be re-run (e.g. after a top-up) without depending on Google keeping the file."""
    if not os.path.exists(p['results']):
        c = gemini_client()
        job = c.batches.get(name=state['batch_id'])
        if job_state(job) != 'SUCCEEDED':
            sys.exit(f'batch state: {job_state(job)}')
        raw = c.files.download(file=job.dest.file_name).decode('utf-8')
        with open(p['results'], 'w', encoding='utf-8', newline='\n') as f:
            f.write(raw)
    with open(p['results'], encoding='utf-8') as f:
        return f.read()


def cmd_topup(args):
    """Judge, directly (same Google API, not batched), every request the batch has no
    usable verdict for: items answered after the batch was sent (e.g. a paper re-sent
    with --retry-failed) and the batch's own unparseable verdicts. Run collect after."""
    p = paths(args.answers)
    with open(p['state'], encoding='utf-8') as f:
        state = json.load(f)
    have = set()
    for row_text in batch_results(p, state).splitlines():
        if row_text.strip():
            row = json.loads(row_text)
            try:
                if row.get('response'):
                    parse_verdict(response_text(row['response']), SCHEMAS[row['key'][-1]])
                    have.add(row['key'])
            except ValueError:
                pass
    if os.path.exists(p['topup']):
        with open(p['topup'], encoding='utf-8') as f:
            have |= {json.loads(x)['key'] for x in f}
    summary, items = load_items(args.answers)
    lines, refs = build_requests(summary, items)
    todo = [ln for ln in lines if ln['key'] not in have]
    print(f'top-up: {len(todo)} request(s) without a usable verdict')
    if not todo:
        return
    c = gemini_client()
    with open(p['topup'], 'a', encoding='utf-8', newline='\n') as out:
        for ln in todo:
            req = ln['request']
            for attempt in range(3):
                resp = c.models.generate_content(
                    model=MODEL, contents=req['contents'][0]['parts'][0]['text'],
                    config={'system_instruction': req['system_instruction']['parts'][0]['text'],
                            **req['generation_config']})
                try:
                    verdict = parse_verdict(resp.text, SCHEMAS[ln['key'][-1]])
                    break
                except ValueError as e:
                    print(f"  {ln['key']}: {e} (attempt {attempt + 1})")
            else:
                continue
            u = resp.usage_metadata
            out.write(json.dumps({'key': ln['key'], 'verdict': verdict, 'tokens': {
                'in': u.prompt_token_count or 0, 'out': u.candidates_token_count or 0,
                'thinking': u.thoughts_token_count or 0}}, ensure_ascii=False) + '\n')
            print(f"  {ln['key']}: ok")
    topup_ids = {ln['key'].rsplit('|', 1)[0] for ln in todo}
    old = json.load(open(p['topup_refs'], encoding='utf-8')) if os.path.exists(p['topup_refs']) else {}
    old.update({i: refs[i] for i in topup_ids if i in refs})
    with open(p['topup_refs'], 'w', encoding='utf-8', newline='\n') as f:
        json.dump(old, f)


def rates(items, verdicts, axis, test, weighted):
    """Share of judged items passing `test`; weighted by the sample's weights if asked."""
    pool = [it for it in items if verdicts.get((it['item_id'], axis))]
    if not pool:
        return None, 0
    w = (lambda it: it['weight']) if weighted else (lambda it: 1.0)
    total = sum(w(it) for it in pool)
    return round(sum(w(it) for it in pool if test(verdicts[(it['item_id'], axis)])) / total, 4), len(pool)


def cmd_collect(args):
    p = paths(args.answers)
    with open(p['state'], encoding='utf-8') as f:
        state = json.load(f)
    with open(p['items'], encoding='utf-8') as f:
        meta = json.load(f)
    raw = batch_results(p, state)
    verdicts, errors, tokens = {}, [], collections.Counter()
    for row_text in raw.splitlines():
        if not row_text.strip():
            continue
        row = json.loads(row_text)
        item_id, axis = row['key'].rsplit('|', 1)
        if row.get('error') or not row.get('response'):
            errors.append((row['key'], json.dumps(row.get('error'))[:120]))
            continue
        usage = row['response'].get('usageMetadata') or {}
        tokens['in'] += usage.get('promptTokenCount', 0)
        tokens['out'] += usage.get('candidatesTokenCount', 0) + usage.get('thoughtsTokenCount', 0)
        try:
            verdicts[(item_id, axis)] = parse_verdict(response_text(row['response']), SCHEMAS[axis])
        except ValueError as e:
            errors.append((row['key'], str(e)[:120]))

    # verdicts judged directly by `topup` fill keys the batch lacks (never override it)
    topped = 0
    if os.path.exists(p['topup']):
        with open(p['topup'], encoding='utf-8') as f:
            for row in map(json.loads, f):
                item_id, axis = row['key'].rsplit('|', 1)
                if (item_id, axis) not in verdicts:
                    verdicts[(item_id, axis)] = row['verdict']
                    tokens['in'] += row['tokens']['in']
                    tokens['out'] += row['tokens']['out'] + row['tokens']['thinking']
                    topped += 1
        errors = [e for e in errors if tuple(e[0].rsplit('|', 1)) not in verdicts]
    if topped:
        meta['refs'].update(json.load(open(p['topup_refs'], encoding='utf-8')))

    summary, items = load_items(args.answers)
    records = []
    for it in items:
        f_v, c_v = verdicts.get((it['item_id'], 'F')), verdicts.get((it['item_id'], 'C'))
        ref = meta['refs'].get(it['item_id'], {})
        records.append({k: it[k] for k in ('item_id', 'instance_id', 'content', 'judge', 'weight',
                                           'gold_is_absence')}
                       | {'abstention': abstention_verdict(it), 'faithfulness': f_v, 'correctness': c_v,
                          'faithful': is_faithful(f_v) if f_v else None,
                          'correct': is_correct(c_v) if c_v else None,
                          'faith_span_verified': span_verified(f_v, ref.get('faith_ref')),
                          'correct_span_verified': span_verified(c_v, ref.get('correct_ref'))})

    out = {'judge_model': MODEL, 'transport': 'google-batch',
           'topup_direct_verdicts': topped, 'batch_id': state['batch_id'],
           'answers': meta['answers'], 'model_judged': summary['model'],
           'correct_threshold': 'match_level >= minor_deviation', 'faithful': list(FAITHFUL)}
    result = {}
    for scope, pool in (('characteristic', [it for it in items if it['content'] == 'characteristic']),
                        ('result', [it for it in items if it['content'] == 'result']),
                        ('pooled_weighted', items)):
        weighted = scope == 'pooled_weighted'
        fr, fn = rates(pool, verdicts, 'F', is_faithful, weighted)
        cr, cn = rates(pool, verdicts, 'C', is_correct, weighted)
        result[scope] = {'faithful': fr, 'n_faithfulness': fn, 'correct': cr, 'n_correctness': cn}
    both = [r for r in records if r['faithful'] is not None and r['correct'] is not None]
    result['two_by_two'] = dict(collections.Counter(
        ('faithful' if r['faithful'] else 'unfaithful') + '_' + ('correct' if r['correct'] else 'incorrect')
        for r in both))
    result['abstention'] = dict(collections.Counter(r['abstention'] for r in records))
    audited = [r[k] for r in records for k in ('faith_span_verified', 'correct_span_verified')
               if r[k] is not None]
    result['judge_span_verified_rate'] = round(sum(audited) / len(audited), 4) if audited else None
    result['judge_errors'] = len(errors)
    result['tokens'] = dict(tokens)
    result['cost_usd_batch'] = round((tokens['in'] * 0.375 + tokens['out'] * 1.875) / 1e6, 4)
    out['summary'] = result
    out['items'] = records
    with open(p['judged'], 'w', encoding='utf-8', newline='\n') as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(json.dumps(result, indent=2))
    if errors:
        print(f'{len(errors)} judge errors; first 3: {errors[:3]}')
    print(f'-> {os.path.relpath(p["judged"], BASE)}')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='command', required=True)
    for name, func in (('prepare', cmd_prepare), ('smoke', cmd_smoke), ('submit', cmd_submit),
                       ('status', cmd_status), ('collect', cmd_collect), ('topup', cmd_topup)):
        sp = sub.add_parser(name)
        sp.add_argument('--answers', required=True, help='answers.json from run_answers.py')
        sp.set_defaults(func=func)
        if name == 'prepare':
            sp.add_argument('--show', action='store_true', help='print one prompt of each kind')
        if name == 'smoke':
            sp.add_argument('--n', type=int, default=6)
    args = ap.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
