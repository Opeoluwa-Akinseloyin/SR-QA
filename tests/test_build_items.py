"""Tests for scripts/10_build_items.py, on the validator tests' small table.

    r0  Study            | Outcome            | Effect estimate
    r1  Saleh 2022 {B18} | Mech. ventilation  | OR 2.54
    r2  Chakrabarti 2022 | Ventilator days    | r = 0.45
    r3  {B20, merged}    | ICU LOS            | r = 0.44
    r4  Adults (banner)
    r5  Konrad 2023 {B22}| 28-day mortality   | HR 1.07 (cf.)
"""
import copy
import importlib.util
import os

from test_table_map_validate import ALIASES, GOOD_MAP, TABLE

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location(
    'build_items', os.path.join(ROOT, 'scripts', '10_build_items.py'))
build_items = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build_items)

XML_IDS = {alias: rid for rid, alias in ALIASES.items()}


def stored_map(change=None):
    """GOOD_MAP as stored after collect: citations as XML ids, not aliases."""
    table_map = copy.deepcopy(GOOD_MAP)
    for row in table_map['rows']:
        row['ref'] = XML_IDS.get(row['ref'], row['ref'])
    if change:
        change(table_map)
    return table_map


TABLE_WITH_REFS = dict(TABLE, review_pmcid='PMC1', table_no=1,
                       refs={rid: {'pmcid': '', 'pmid': '', 'doi': '', 'text': rid}
                             for rid in ALIASES})


class StubLinker:
    def link(self, ref):
        return ('PMC' + ref['text'][1:], 'ccby_full_text') if ref else ('', 'no_ref')


def build(change=None):
    record = {'map': stored_map(change), 'table_type': 'mixed'}
    return build_items.build_table_items(TABLE_WITH_REFS, record, StubLinker())


def by_rows_and_col(items):
    return {(tuple(i['rows']), i['col_index']): i for i in items}


def test_groups_split_a_study_by_outcome_and_merge_continuations():
    groups = build_items.groups_of(stored_map())
    assert [g['rows'] for g in groups] == [[1], [2], [3], [5]]
    # Chakrabarti's rows without outcome labels are continuations: one group.
    def no_labels(m):
        m['rows'][1]['outcome'] = m['rows'][2]['outcome'] = ''
    assert [g['rows'] for g in build_items.groups_of(stored_map(no_labels))] == [[1], [2, 3], [5]]


def test_labels_are_dropped_for_single_row_studies():
    def label_saleh(m):
        m['rows'][0]['outcome'] = 'mechanical ventilation'
    groups = build_items.groups_of(stored_map(label_saleh))
    assert groups[0]['outcome'] == ''                  # Saleh has one row
    assert groups[1]['outcome'] == 'ventilator days'   # Chakrabarti has two


def test_question_carries_the_labels_only_where_needed():
    items = by_rows_and_col(build()[0])
    assert items[((1,), 2)]['question'] == 'What effect estimate was reported?'
    assert items[((2,), 2)]['question'] == ('What effect estimate was reported? '
                                            '(outcome: ventilator days)')


def test_gold_is_verbatim_and_a_merged_cell_counts_once():
    def no_labels(m):
        m['rows'][1]['outcome'] = m['rows'][2]['outcome'] = ''
    items = by_rows_and_col(build(no_labels)[0])
    assert items[((2, 3), 2)]['gold'] == 'r = 0.45 | r = 0.44'
    assert all(i['gold_verbatim'] for i in items.values())
    assert build_items.verify_gold('r = 0.45 | r = 0.46', TABLE, 2) is False


def test_verbatim_check_accepts_a_cell_whose_own_text_contains_the_separator():
    table = dict(TABLE, cells=copy.deepcopy(TABLE['cells']))
    table['cells'][13]['text'] = 'Death | ICD-10 codes'          # Konrad's outcome cell
    assert build_items.verify_gold('Death | ICD-10 codes', table, 1)
    assert build_items.verify_gold('Death | ICD-10 codes | ICU LOS', table, 1)
    assert not build_items.verify_gold('Death', table, 1)          # half a cell


def test_an_item_whose_label_gives_the_answer_away_is_dropped():
    items, dropped = build()
    # Chakrabarti's Outcome-column items: gold "Ventilator days" asked
    # "(outcome: ventilator days)" - the label is the answer.
    assert {(tuple(d['rows']), d['col_index']) for d in dropped} == {((2,), 1)}
    assert dropped[0]['drop_reason'] == 'label_reveals_answer'
    # Its effect-estimate item is kept, and so is the Outcome item of a
    # single-row study (no label, so nothing to leak).
    kept = by_rows_and_col(items)
    assert ((2,), 2) in kept and ((1,), 1) in kept


def test_leak_rule():
    assert build_items.leaks('ICU LOS', '', 'ICU LOS')
    assert not build_items.leaks('OR 2.54', '', 'mortality')
    assert not build_items.leaks('', '', 'mortality')


def test_every_item_is_linked_through_its_study_citation():
    items = build()[0]
    konrad = [i for i in items if i['rows'] == [5]]
    # Konrad's citation is B22 (study column), never B99 (effect column).
    assert {i['ref'] for i in konrad} == {'B22'}
    assert {i['source_status'] for i in items} == {'ccby_full_text'}


def test_a_full_width_study_heading_row_never_supplies_answers():
    # Konrad's row replaced by a full-width heading (one cell over all columns)
    # followed by a data row; both share Konrad's study key, as maps give them.
    table = copy.deepcopy(TABLE_WITH_REFS)
    table['cells'].append({'text': 'Konrad 2023', 'rids': ['B22'], 'is_header': False,
                           'rowspan': 1, 'colspan': 3})
    heading = len(table['cells']) - 1
    table['grid'] = table['grid'][:5] + [[heading] * 3, [12, 13, 14]]
    table['n_rows'] = 7

    def add_heading(m):
        m['rows'].append({'row_index': 6, 'kind': 'study', 'study_key': 2, 'arm': '',
                          'outcome': 'mortality', 'ref': 'B22', 'ref_matches_label': 'yes'})
        m['rows'][4]['outcome'] = ''
    record = {'map': stored_map(add_heading), 'table_type': 'mixed'}
    assert build_items.heading_rows(table) == {4, 5}     # 4 is the 'Adults' banner, also full-width
    items, dropped = build_items.build_table_items(table, record, StubLinker())
    assert all(5 not in i['rows'] for i in items)
    assert all(i['gold'] != 'Konrad 2023' for i in items)
    assert {d['drop_reason'] for d in dropped} >= {'study_heading_row'}


def test_a_study_label_merged_into_another_column_is_not_its_answer():
    # Saleh's label cell spans Study and Outcome; Outcome must not answer "Saleh 2022".
    table = copy.deepcopy(TABLE_WITH_REFS)
    table['grid'][1] = [3, 3, 5]
    record = {'map': stored_map(), 'table_type': 'mixed'}
    items, dropped = build_items.build_table_items(table, record, StubLinker())
    assert not any(i['rows'] == [1] and i['col_index'] == 1 for i in items)
    assert any(d['rows'] == [1] and d['col_index'] == 1 and d['drop_reason'] == 'study_label_cell'
               for d in dropped)
    assert any(i['rows'] == [1] and i['col_index'] == 2 and i['gold'] == 'OR 2.54' for i in items)


def test_unescape_literal_codes_from_the_annotator():
    bs = chr(92)
    assert build_items.unescape_literal(f'Age, mean {bs}u00b1 SD') == 'Age, mean \u00b1 SD'
    assert build_items.unescape_literal(f'mL.kg{bs}u22121') == 'mL.kg\u22121'
    assert build_items.unescape_literal(f'{bs}ud83d{bs}ude00') == '\U0001f600'
    assert build_items.unescape_literal(f'C:{bs}users') == f'C:{bs}users'


def test_unescape_map_reaches_nested_strings_only():
    bs = chr(92)
    table_map = {'accept': True, 'columns': [{'col_index': 2, 'field': f'Age {bs}u00b1 SD'}]}
    out = build_items.unescape_map(table_map)
    assert out == {'accept': True, 'columns': [{'col_index': 2, 'field': 'Age \u00b1 SD'}]}
