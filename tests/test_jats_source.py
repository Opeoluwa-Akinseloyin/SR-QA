"""Tests for scripts/eval/jats_source.py, on one small synthetic JATS paper.

    front:  title + abstract
    body:   Introduction | Methods > Statistical analysis | Results | Discussion
            Results cites <xref ref-type="table" rid="T1">1</xref>
    floats-group (outside <body>):
            Table 1  thead: Group | Age (years) - Mean | n
                     tbody: Placebo | 54.2 | 30
                            Drug    | 51.8 | 31
"""
import importlib.util
import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location(
    'jats_source', os.path.join(ROOT, 'scripts', 'eval', 'jats_source.py'))
js = importlib.util.module_from_spec(spec)
spec.loader.exec_module(js)

PAPER = """<?xml version="1.0"?>
<article>
 <front><article-meta>
  <title-group><article-title>A Drug Trial</article-title></title-group>
  <abstract><p>We tested a drug.</p></abstract>
 </article-meta></front>
 <body>
  <sec><title>Introduction</title><p>Background {intro}</p></sec>
  <sec><title>Methods</title><p>Adults were randomised.</p>
   <sec><title>Statistical analysis</title><p>We used t-tests.</p></sec>
  </sec>
  <sec><title>Results</title><p>Ages are in <xref ref-type="table" rid="T1">1</xref>.</p></sec>
  <sec><title>Discussion</title><p>Discussion {disc}</p></sec>
 </body>
 <floats-group>
  <table-wrap id="T1"><label>Table 1</label><caption><p>Baseline</p></caption>
   <table>
    <thead><tr><td>Group</td><td>Age (years)</td><td>n</td></tr>
           <tr><td></td><td>Mean</td><td></td></tr></thead>
    <tbody><tr><td>Placebo</td><td>54.2</td><td>30</td></tr>
           <tr><td>Drug</td><td>51.8</td><td>31</td></tr></tbody>
   </table>
   <table-wrap-foot><fn><p>Age in years.</p></fn></table-wrap-foot>
  </table-wrap>
 </floats-group>
</article>
"""


@pytest.fixture
def paper(tmp_path):
    def write(intro='', disc=''):
        path = tmp_path / 'paper.xml'
        path.write_text(PAPER.format(intro=intro, disc=disc), encoding='utf-8')
        return str(path)
    return write


def titles(src):
    return [s['title'] for s in src['sections']]


def test_table_outside_body_is_found(paper):
    src = js.parse_source(paper(), 200000)
    assert src['n_tables_found'] == 1
    assert src['n_tables_kept'] == 1
    assert 'Table 1' in titles(src)


def test_cell_values_keep_row_and_column(paper):
    table = next(s for s in js.parse_source(paper(), 200000)['sections']
                 if s['title'] == 'Table 1')
    assert '[row=Placebo | Age (years) - Mean = 54.2]' in table['text']
    assert '[row=Drug | n = 31]' in table['text']
    assert 'Notes: Age in years.' in table['text']


def test_xref_resolves_to_table_name(paper):
    results = next(s for s in js.parse_source(paper(), 200000)['sections']
                   if s['title'] == 'Results')
    assert 'Ages are in Table 1.' in results['text']


def test_sectioned_order(paper):
    assert titles(js.parse_source(paper(), 200000)) == [
        'Abstract', 'Methods', 'Methods > Statistical analysis', 'Results',
        'Table 1', 'Introduction', 'Discussion']


def test_linear_order_keeps_document_order_tables_last(paper):
    assert titles(js.parse_source(paper(), 200000, layout='linear')) == [
        'Abstract', 'Introduction', 'Methods', 'Methods > Statistical analysis',
        'Results', 'Discussion', 'Table 1']


def test_cap_drops_discussion_before_results_and_tables(paper):
    path = paper(intro='i' * 300, disc='d' * 3000)
    full = js.parse_source(path, 200000)
    cap = full['n_chars'] - 100          # just over: one section has to go
    cut = js.parse_source(path, cap)
    assert cut['truncated']
    assert 'Discussion' not in titles(cut)
    for kept in ('Methods', 'Results', 'Table 1', 'Introduction'):
        assert kept in titles(cut)


def test_no_truncation_under_cap(paper):
    src = js.parse_source(paper(), 200000)
    assert not src['truncated']
    rendered = js.render_source(src)
    assert rendered.startswith('TITLE: A Drug Trial')
    assert 'truncated' not in rendered


def test_section_ids_follow_presentation_order(paper):
    src = js.parse_source(paper(), 200000)
    assert [s['id'] for s in src['sections']] == [f'S{i}' for i in range(1, 8)]
