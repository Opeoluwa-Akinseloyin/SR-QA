"""Tests for scripts/eval/judge_answers.py: verdict checking and the scoring rules."""
import importlib.util
import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location(
    'judge_answers', os.path.join(ROOT, 'scripts', 'eval', 'judge_answers.py'))
ja = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ja)

GOOD_FAITH = ('{"evidence_supports_answer": "partially_supported", '
              '"answer_adds_unsupported_detail": false, "source_internally_inconsistent": false, '
              '"relied_on_span": "in Bangladesh", "rationale": "ok"}')


def test_parse_verdict_accepts_a_schema_shaped_verdict():
    v = ja.parse_verdict(GOOD_FAITH, ja.FAITH_SCHEMA)
    assert ja.is_faithful(v)


def test_parse_verdict_rejects_an_enum_the_rubric_does_not_define():
    with pytest.raises(ValueError):
        ja.parse_verdict(GOOD_FAITH.replace('partially_supported', 'mostly_supported'),
                         ja.FAITH_SCHEMA)


def test_parse_verdict_rejects_missing_keys():
    with pytest.raises(ValueError):
        ja.parse_verdict('{"relation": "equivalent"}', ja.CORRECT_SCHEMA)


@pytest.mark.parametrize('level,correct', [('complete', True), ('minor_deviation', True),
                                           ('partial', False), ('wrong_value', False),
                                           ('no_match', False)])
def test_correct_is_minor_deviation_or_better(level, correct):
    assert ja.is_correct({'match_level': level}) is correct


def item(absent, answered=True, not_reported=False):
    return {'gold_is_absence': absent,
            'prediction': {'answered': answered, 'not_reported': not_reported}}


def test_abstention_verdicts():
    assert ja.abstention_verdict(item(True, not_reported=True)) == 'correct_abstention'
    assert ja.abstention_verdict(item(True)) == 'over_answer'
    assert ja.abstention_verdict(item(False, not_reported=True)) == 'missed_value'
    assert ja.abstention_verdict(item(False)) == 'answered'
    assert ja.abstention_verdict(item(False, answered=False)) == 'missing_from_response'


def test_span_self_audit():
    assert ja.span_verified({'relied_on_span': 'IN  Bangladesh'}, 'trial in Bangladesh, 2010')
    assert ja.span_verified({'relied_on_span': 'in Nepal'}, 'trial in Bangladesh') is False
    assert ja.span_verified({'relied_on_span': ''}, 'anything') is None


def test_pooled_rate_uses_weights_per_content_rate_does_not():
    items = [{'item_id': 'r1', 'weight': 1.0}, {'item_id': 'c1', 'weight': 3.0}]
    verdicts = {('r1', 'C'): {'match_level': 'complete'}, ('c1', 'C'): {'match_level': 'no_match'}}
    assert ja.rates(items, verdicts, 'C', ja.is_correct, weighted=False) == (0.5, 2)
    assert ja.rates(items, verdicts, 'C', ja.is_correct, weighted=True) == (0.25, 2)
