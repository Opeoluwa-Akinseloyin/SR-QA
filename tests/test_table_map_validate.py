"""Tests for scripts/table_map_validate.py.

The fixture is a small table with every feature the validator must handle:

    r0  Study            | Outcome            | Effect estimate
    r1  Saleh 2022 {B18} | Mech. ventilation  | OR 2.54
    r2  Chakrabarti 2022 | Ventilator days    | r = 0.45
    r3  {B20, merged}    | ICU LOS            | r = 0.44
    r4  Adults (banner spanning all three columns)
    r5  Konrad 2023 {B22}| 28-day mortality   | HR 1.07 (cf. {B99})

B99 is cited in a NON-study column: v1 could link Konrad's row to it.
"""
import copy
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                'scripts'))
from table_map_validate import validate  # noqa: E402


def cell(text, rids=(), header=False):
    return {'text': text, 'rids': list(rids), 'is_header': header, 'rowspan': 1, 'colspan': 1}


TABLE = {
    'n_rows': 6,
    'n_cols': 3,
    'cells': [
        cell('Study', header=True), cell('Outcome', header=True),            # 0, 1
        cell('Effect estimate', header=True),                                # 2
        cell('Saleh 2022', ['B18']), cell('Mech. ventilation'), cell('OR 2.54'),   # 3, 4, 5
        cell('Chakrabarti 2022', ['B20']), cell('Ventilator days'),          # 6, 7
        cell('r = 0.45'), cell('ICU LOS'), cell('r = 0.44'),                 # 8, 9, 10
        cell('Adults'),                                                      # 11
        cell('Konrad 2023', ['B22']), cell('28-day mortality'),              # 12, 13
        cell('HR 1.07 (cf.)', ['B99']),                                      # 14
    ],
    'grid': [
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8],
        [6, 9, 10],          # cell 6 is merged down: same study
        [11, 11, 11],        # banner
        [12, 13, 14],
    ],
}
ALIASES = {'B18': 'R1', 'B20': 'R2', 'B22': 'R3', 'B99': 'R4'}


def row(index, kind='study', key=0, arm='', outcome='', ref='', match='unclear'):
    return {'row_index': index, 'kind': kind, 'study_key': key, 'arm': arm,
            'outcome': outcome, 'ref': ref, 'ref_matches_label': match}


def column(index, role, field, question='', content='none',
           category='not_applicable', score=0):
    return {'col_index': index, 'role': role, 'field': field, 'question': question,
            'content': content, 'answerability_category': category,
            'answerability_score': score}


GOOD_MAP = {
    'accept': True,
    'reject_reason': '',
    'n_header_rows': 1,
    'study_col_index': 0,
    'rows': [
        row(1, key=0, ref='R1', match='yes'),
        row(2, key=1, outcome='ventilator days', ref='R2', match='yes'),
        row(3, key=1, outcome='ICU length of stay', ref='R2', match='yes'),
        row(4, kind='banner', key=-1),
        row(5, key=2, ref='R3', match='yes'),
    ],
    'columns': [
        column(0, 'identifier', 'Study'),
        column(1, 'extraction', 'Outcome', 'Which outcome was assessed?',
               'characteristic', 'reported', 5),
        column(2, 'extraction', 'Effect estimate', 'What effect estimate was reported?',
               'result', 'reported', 4),
    ],
}


def codes(problems):
    return {code for code, _ in problems}


def broken(change):
    """A deep copy of GOOD_MAP with `change` applied to it."""
    table_map = copy.deepcopy(GOOD_MAP)
    change(table_map)
    return table_map


def test_good_map_has_no_problems():
    assert validate(GOOD_MAP, TABLE, ALIASES) == ([], [])


def test_rejected_table_needs_no_rows_or_columns():
    table_map = {'accept': False, 'reject_reason': 'summary table', 'n_header_rows': 1,
                 'study_col_index': 0, 'rows': [], 'columns': []}
    assert validate(table_map, TABLE, ALIASES) == ([], [])


@pytest.mark.parametrize('change, expected', [
    (lambda m: m.update(n_header_rows=6), 'bad_header_rows'),
    (lambda m: m.update(study_col_index=3), 'bad_study_col'),
    (lambda m: m['rows'].pop(2), 'rows_incomplete'),
    (lambda m: m['rows'].reverse(), 'rows_incomplete'),
    (lambda m: m['rows'][0].update(study_key=-1), 'bad_study_key'),
    (lambda m: m['rows'][3].update(study_key=0), 'bad_study_key'),
    # v1's cross-column linking bug: B99 is cited only in the effect column
    (lambda m: m['rows'][4].update(ref='R4'), 'ref_not_in_study_column'),
    (lambda m: m['rows'][2].update(ref='R1'), 'refs_disagree_within_study'),
    (lambda m: m['columns'].pop(), 'cols_incomplete'),
    (lambda m: m['columns'][2].update(question=' '), 'extraction_missing_question'),
    (lambda m: m['columns'][2].update(content='none'), 'extraction_missing_content'),
    (lambda m: m['columns'][2].update(answerability_score=0), 'extraction_missing_answerability'),
    (lambda m: m['columns'][2].update(answerability_category='not_applicable'),
     'extraction_missing_answerability'),
    # v1's duplicate-header defect; names are compared case- and space-insensitively
    (lambda m: m['columns'][1].update(field='effect  ESTIMATE'), 'duplicate_field_names'),
])
def test_each_error_is_caught(change, expected):
    errors, _ = validate(broken(change), TABLE, ALIASES)
    assert expected in codes(errors)


@pytest.mark.parametrize('change, expected', [
    (lambda m: m.update(reject_reason='oops'), 'accepted_with_reason'),
    (lambda m: m['rows'][3].update(arm='adults'), 'labels_on_non_study_row'),
    (lambda m: m['rows'][0].update(ref='', match='yes'), 'match_without_ref'),
    (lambda m: m['columns'][0].update(question='Who?'), 'extraction_fields_on_other_column'),
    (lambda m: m['columns'][2].update(answerability_category='reviewer_judgement'),
     'answerability_inconsistent'),                     # judgement scored 4
    (lambda m: m['columns'][1].update(answerability_score=1), 'answerability_inconsistent'),
    (lambda m: m['columns'][0].update(role='extraction', question='Which study?',
                                      content='characteristic', answerability_category='bibliographic',
                                      answerability_score=5), 'study_column_is_extraction'),
])
def test_each_warning_is_raised_without_errors(change, expected):
    errors, warnings = validate(broken(change), TABLE, ALIASES)
    assert errors == []
    assert expected in codes(warnings)


def test_ref_from_an_identifier_column_is_allowed():
    # The same citation (R4, column 2) that is an error from a content column
    # is legal once column 2 is an identifier column, e.g. a separate "Ref." column.
    def change(m):
        m['rows'][4]['ref'] = 'R4'
        m['columns'][2] = column(2, 'identifier', 'Reference')
    errors, _ = validate(broken(change), TABLE, ALIASES)
    assert 'ref_not_in_study_column' not in codes(errors)


def test_non_adjacent_rows_with_same_labels_warn():
    # Konrad's row given Saleh's key and labels, with the banner in between.
    # R1 is cited in study 0's study column (row 1), so the ref itself is legal.
    def change(m):
        m['rows'][4].update(study_key=0, ref='R1')
    errors, warnings = validate(broken(change), TABLE, ALIASES)
    assert errors == []
    assert 'same_labels_not_adjacent' in codes(warnings)


def test_continuation_rows_with_same_labels_are_fine():
    # Chakrabarti's two rows with no outcome labels: adjacent, so joined, no warning.
    def change(m):
        m['rows'][1]['outcome'] = m['rows'][2]['outcome'] = ''
    assert validate(broken(change), TABLE, ALIASES) == ([], [])
