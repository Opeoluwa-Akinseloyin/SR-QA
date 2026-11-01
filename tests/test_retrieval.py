"""Tests for scripts/eval/retrieval.py: the relevance-label rules and the IR metrics."""
import importlib.util
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location(
    'retrieval', os.path.join(ROOT, 'scripts', 'eval', 'retrieval.py'))
rt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rt)


def test_numbers_inside_words_do_not_count():
    assert rt.standalone_numbers('g6pd deficient only') == set()
    assert rt.standalone_numbers('il-6 levels 12.5') == {'12.5'}
    assert rt.standalone_numbers('covid-19 cases') == set()
    assert rt.standalone_numbers('n = 1,068') == {'1068'}


def test_distinctive_numbers():
    assert not rt.distinctive({'0'})
    assert not rt.distinctive({'74'})
    assert rt.distinctive({'54.2'})
    assert rt.distinctive({'1068'})
    assert rt.distinctive({'30', '31'})


def grade(fragment, passage):
    norm = rt.answerability.normalise(passage)
    return rt.fragment_grade(rt.answerability.normalise(fragment), norm, rt.standalone_numbers(norm))


def test_fragment_grades():
    assert grade('Bangladesh', 'The trial ran in Bangladesh.') == 2
    assert grade('2', 'Recruited in 2020.') == 0                 # no match inside 2020
    assert grade('54.2 (3.1)', '[row=Placebo | Age = 54.2 ± 3.1]') == 1
    assert grade('6', 'Six groups; IL-6 measured at week 6.') == 2   # a phrase match is still exact
    assert grade('G6PD deficient', 'six patients, 6 sites') == 0


def test_metrics():
    m = rt.metrics_for(['a', 'b', 'c'], {'b': 2})
    assert m['recall@5'] == 1.0 and m['mrr'] == 0.5
    assert rt.metrics_for(['x'] * 12 + ['b'], {'b': 2})['recall@10'] == 0.0
