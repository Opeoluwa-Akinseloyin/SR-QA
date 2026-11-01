"""
table_map_validate.py
---------------------
Stage 2.4 of the v2 pipeline: check one table map against the table it
describes, before anything uses it.

Structured outputs guarantee the map's SHAPE (every field present, every enum
value legal). They cannot guarantee that the content makes sense for this
particular table - that no row was skipped, that a citation really comes from
the study column, that two columns do not share a name. This module checks
that mechanically.

  errors    the map is unusable: no items are built from it
  warnings  recorded and counted, but the map is still used

Every problem has a short code, so the pilot (2.6) can count them per model.

Descends from sanitise() in old/scripts/apply_structure_map.py, which clamped
out-of-range indices back to v1's defaults. v2 rejects instead of repairing:
a map that points outside the table is a wrong map, and silently patching it
would hide how often the annotator gets things wrong.

Usage (as a library):
  from table_map_validate import validate
  errors, warnings = validate(table_map, table, aliases)
"""
from collections import defaultdict


def study_column_refs(table, table_map, aliases):
    """Map study_key -> the aliases a study's ref may be chosen from.

    That is every citation in the study's rows in the study column or in any
    other column the map calls an "identifier" (e.g. a separate "Ref." column -
    pilot finding, 2026-10-05). Content columns never count: that is v1's
    cross-column linking bug. The error code keeps its original name,
    ref_not_in_study_column, so pilot counts stay comparable.
    """
    allowed_cols = {table_map['study_col_index']}
    allowed_cols |= {c['col_index'] for c in table_map['columns']
                     if c['role'] == 'identifier' and 0 <= c['col_index'] < table['n_cols']}
    refs = defaultdict(set)
    for row in table_map['rows']:
        if row['kind'] != 'study':
            continue
        for col in allowed_cols:
            cell_id = table['grid'][row['row_index']][col]
            if cell_id is not None:
                refs[row['study_key']].update(aliases[rid] for rid in table['cells'][cell_id]['rids'])
    return refs


def validate(table_map, table, aliases):
    """Return (errors, warnings): two lists of (code, detail) tuples.

    `aliases` maps the table's XML citation ids to the short ids (R1, R2, ...)
    the model was shown; `ref` values in the map are those short ids.
    """
    errors, warnings = [], []
    n_rows, n_cols = table['n_rows'], table['n_cols']

    if not table_map['accept']:
        if table_map['rows'] or table_map['columns']:
            warnings.append(('rejected_but_mapped', 'rows/columns given for a rejected table'))
        return errors, warnings
    if table_map['reject_reason']:
        warnings.append(('accepted_with_reason', table_map['reject_reason']))

    # --- table level -------------------------------------------------------
    header = table_map['n_header_rows']
    if not 0 <= header < n_rows:
        errors.append(('bad_header_rows', f'n_header_rows={header}, table has {n_rows} rows'))
        return errors, warnings        # row checks are meaningless without it
    if not 0 <= table_map['study_col_index'] < n_cols:
        errors.append(('bad_study_col',
                       f"study_col_index={table_map['study_col_index']}, table has {n_cols} columns"))
        return errors, warnings

    # --- rows --------------------------------------------------------------
    rows = table_map['rows']
    got = [row['row_index'] for row in rows]
    if got != list(range(header, n_rows)):
        errors.append(('rows_incomplete', f'expected rows {header}..{n_rows - 1}, got {got}'))
        return errors, warnings

    for row in rows:
        r = row['row_index']
        if row['kind'] == 'study':
            if row['study_key'] < 0:
                errors.append(('bad_study_key', f'study row {r} has study_key {row["study_key"]}'))
        else:
            if row['study_key'] != -1:
                errors.append(('bad_study_key', f'{row["kind"]} row {r} has study_key {row["study_key"]}'))
            if row['arm'] or row['outcome'] or row['ref']:
                warnings.append(('labels_on_non_study_row', f'row {r}'))
        if not row['ref'] and row['ref_matches_label'] != 'unclear':
            warnings.append(('match_without_ref', f'row {r}'))

    # A study's ref must come from the study column of that study's rows
    # (v1's cross-column linking bug), and one study points at one paper.
    allowed = study_column_refs(table, table_map, aliases)
    refs_by_study = defaultdict(set)
    for row in rows:
        if row['kind'] != 'study' or not row['ref']:
            continue
        if row['ref'] not in allowed[row['study_key']]:
            errors.append(('ref_not_in_study_column',
                           f"row {row['row_index']} ref {row['ref']} is not cited in the "
                           f"study or identifier columns of study {row['study_key']}"))
        refs_by_study[row['study_key']].add(row['ref'])
    for key, refs in refs_by_study.items():
        if len(refs) > 1:
            errors.append(('refs_disagree_within_study', f'study {key}: {sorted(refs)}'))

    # Rows sharing (study, arm, outcome) are joined; if they are not adjacent,
    # the labels are probably wrong rather than the rows being continuations.
    last_seen = {}
    for position, row in enumerate(rows):
        if row['kind'] != 'study':
            continue
        label = (row['study_key'], row['arm'], row['outcome'])
        if label in last_seen and last_seen[label] != position - 1:
            warnings.append(('same_labels_not_adjacent', f"rows with labels {label}"))
        last_seen[label] = position

    # --- columns -----------------------------------------------------------
    columns = table_map['columns']
    got = [col['col_index'] for col in columns]
    if got != list(range(n_cols)):
        errors.append(('cols_incomplete', f'expected columns 0..{n_cols - 1}, got {got}'))
        return errors, warnings

    fields = defaultdict(list)
    for col in columns:
        c = col['col_index']
        if col['role'] == 'extraction':
            if not col['question'].strip():
                errors.append(('extraction_missing_question', f'column {c}'))
            if col['content'] == 'none':
                errors.append(('extraction_missing_content', f'column {c}'))
            if (col['answerability_category'] == 'not_applicable'
                    or not 1 <= col['answerability_score'] <= 5):
                errors.append(('extraction_missing_answerability',
                               f"column {c}: {col['answerability_category']} "
                               f"{col['answerability_score']}"))
            fields[' '.join(col['field'].lower().split())].append(c)
            # The score should agree with the category (see FIELD_DEFINITIONS).
            category, score = col['answerability_category'], col['answerability_score']
            if ((category in ('reviewer_judgement', 'review_calculated') and score >= 4)
                    or (category == 'reported' and score <= 2)):
                warnings.append(('answerability_inconsistent', f'column {c}: {category} {score}'))
        else:
            # A question or content type here suggests the role is wrong. An
            # answerability category is harmless (it is ignored for these roles)
            # and models sensibly fill it in - not worth a warning.
            if col['question'] or col['content'] != 'none':
                warnings.append(('extraction_fields_on_other_column', f'column {c} ({col["role"]})'))
    for name, cols in fields.items():
        if len(cols) > 1:
            errors.append(('duplicate_field_names', f'"{name}" on columns {cols}'))

    if not fields:
        warnings.append(('accepted_without_extraction_columns', ''))
    if columns[table_map['study_col_index']]['role'] == 'extraction':
        warnings.append(('study_column_is_extraction', f"column {table_map['study_col_index']}"))

    return errors, warnings
