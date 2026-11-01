"""
10_build_items.py
-----------------
Stages 3.1 and 3.2 of the v2 pipeline: turn every accepted table's map into
QA items, and link each item to its source paper. Deterministic - no LLM.

For each accepted table (data/v2/table_maps.jsonl, status "mapped"):

  1. Group the study rows: rows of one study (study_key) with the same arm and
     outcome labels form one group - continuation rows are merged.
  2. Labels only where needed: a study's arm/outcome labels are used only if
     that study has more than one group. Models also label single-row studies
     (seen in the 2.5 smoke test); those labels are ignored.
  3. One item per group x extraction column:
       question  the column's question, plus "(group: ...; outcome: ...)"
                 where labels are needed (table_map_schema.qualified_question)
       gold      the column's cell texts for the group's rows, VERBATIM: each
                 merged cell once, distinct texts in row order, joined " | "
                 (v1's convention). verify_gold() checks every fragment is
                 exactly a cell text of that column - v1's guarantee.
     An item whose label gives its answer away is dropped and logged - e.g.
     the Outcome column asked "(outcome: ICU LOS)" with gold "ICU LOS". Rule:
     at least LEAK_SHARE of the gold's words appear in the labels.
  4. Link (3.2): the group's citation from the map -> its PMCID (the
     reference's own PMCID, else v1's or v2's NCBI lookups) -> source status:
       ccby_full_text   CC-BY and the file has a body - usable
       ccby_no_body     CC-BY, front matter only
       not_ccby         on disk, another licence
       not_downloaded   in PMC, not on disk
       not_in_pmc       NCBI found no PMCID
       unresolved       PMID/DOI never looked up
       no_id            no PMCID, PMID or DOI
       no_ref           the map gave this study no citation

Nothing is filtered here: every candidate item is written with its status,
and 3.3 applies the filters one at a time, logging counts for the funnel.

Inputs:  data/v2/tables.jsonl.gz, data/v2/table_maps.jsonl,
         data/v2/source_licences.csv, data/input/resolved_ids_v1.csv,
         data/v2/resolved_ids_v2.csv
Outputs: data/v2/items_all.jsonl       one line per candidate item
         data/v2/items_dropped.jsonl   dropped items, with drop_reason: label_reveals_answer,
                                       study_heading_row or study_label_cell

Usage:
  python scripts/10_build_items.py
"""
import collections
import csv
import gzip
import importlib.util
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from table_map_schema import qualified_question  # noqa: E402

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TABLES_PATH = os.path.join(BASE, 'data', 'v2', 'tables.jsonl.gz')
MAPS_PATH = os.path.join(BASE, 'data', 'v2', 'table_maps.jsonl')
LICENCES_CSV = os.path.join(BASE, 'data', 'v2', 'source_licences.csv')
ITEMS_PATH = os.path.join(BASE, 'data', 'v2', 'items_all.jsonl')
DROPPED_PATH = os.path.join(BASE, 'data', 'v2', 'items_dropped.jsonl')

SEPARATOR = ' | '
LEAK_SHARE = 0.8
WORD = re.compile(r'[a-z0-9]+')
# A literal "±" (backslash, u, four hex digits) in model-written text: in 28 maps
# the annotator double-escaped non-ASCII characters, so they were stored as escape
# codes instead of characters (found 2026-10-09: 98 questions, 15 reviews).
LITERAL_ESCAPE = re.compile(r'(?:\\u[0-9a-fA-F]{4})+')


def unescape_literal(text):
    """Turn literal \\uXXXX runs back into characters (surrogate pairs included)."""
    def decode(match):
        units = match.group(0).split('\\u')[1:]
        raw = b''.join(int(u, 16).to_bytes(2, 'big') for u in units)
        return raw.decode('utf-16-be', errors='replace')
    return LITERAL_ESCAPE.sub(decode, text)


def unescape_map(value):
    """Apply unescape_literal to every string in a table map. Only model-written
    text lives in the map; gold answers are copied from the table cells later."""
    if isinstance(value, str):
        return unescape_literal(value)
    if isinstance(value, list):
        return [unescape_map(v) for v in value]
    if isinstance(value, dict):
        return {k: unescape_map(v) for k, v in value.items()}
    return value

# The citation lookups are shared with 07_run_tables.py (a digit-led module name).
_spec = importlib.util.spec_from_file_location(
    'run_tables', os.path.join(BASE, 'scripts', '07_run_tables.py'))
run_tables = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_tables)


# ---------------------------------------------------------------------------
# 1. Groups and labels
# ---------------------------------------------------------------------------
def norm(text):
    return ' '.join((text or '').lower().split())


def groups_of(table_map):
    """Study-row groups in table order: {study_key, arm, outcome, rows, ref}.

    Labels are kept only for studies with more than one distinct (arm,
    outcome) - a single-row study needs none, and its label would only risk
    giving the answer away.
    """
    groups = collections.OrderedDict()
    for row in table_map['rows']:
        if row['kind'] != 'study':
            continue
        key = (row['study_key'], norm(row['arm']), norm(row['outcome']))
        group = groups.setdefault(key, {'study_key': row['study_key'], 'arm': row['arm'].strip(),
                                        'outcome': row['outcome'].strip(), 'rows': [], 'ref': ''})
        group['rows'].append(row['row_index'])
        group['ref'] = group['ref'] or row['ref']
    per_study = collections.Counter(g['study_key'] for g in groups.values())
    for group in groups.values():
        if per_study[group['study_key']] == 1:
            group['arm'] = group['outcome'] = ''
    return list(groups.values())


# ---------------------------------------------------------------------------
# 2. Gold and its checks
# ---------------------------------------------------------------------------
def gold_of(table, rows, col, label_cells=frozenset()):
    """Verbatim gold: the column's distinct cell texts over `rows`, in order.

    Cells in `label_cells` - the study-label cells of these rows - are skipped:
    a study's name merged sideways into another column ("Jelinek et al. [47]"
    spanning Study and Time point) is not that column's answer. Found by the
    4.1 hand audit, with the full-width variant handled by heading_rows().
    """
    seen_cells, texts = set(), []
    for r in rows:
        cell_id = table['grid'][r][col]
        if cell_id is None or cell_id in seen_cells or cell_id in label_cells:
            continue
        seen_cells.add(cell_id)
        text = table['cells'][cell_id]['text']
        if text and text not in texts:
            texts.append(text)
    return SEPARATOR.join(texts)


def verify_gold(gold, table, col):
    """True if gold is exactly cell (' | ' cell)* for cell texts of this column.

    Built from cells, this holds by construction; the check runs anyway so the
    guarantee is a measured number, as in v1's apply_structure_map --verify.
    It parses rather than splitting on ' | ', because a cell's own text can
    contain ' | ' (e.g. "Cardiac death | ICD-10 codes") - as v1's
    parses_as_cells() did.
    """
    if not gold:
        return True
    legal = {table['cells'][row[col]]['text'] for row in table['grid'] if row[col] is not None}
    # reachable[i]: gold[:i] is a complete join of legal cell texts
    reachable = [False] * (len(gold) + 1)
    reachable[0] = True
    for start in range(len(gold)):
        if not reachable[start]:
            continue
        for text in legal:
            end = start + len(text)
            if gold.startswith(text, start) and (end == len(gold)
                                                 or gold.startswith(SEPARATOR, end)):
                reachable[end if end == len(gold) else end + len(SEPARATOR)] = True
    return reachable[len(gold)]


def leaks(gold, arm, outcome):
    """True if the labels in the question give the answer away."""
    gold_words = set(WORD.findall(gold.lower()))
    label_words = set(WORD.findall(f'{arm} {outcome}'.lower()))
    if not gold_words or not label_words:
        return False
    return len(gold_words & label_words) / len(gold_words) >= LEAK_SHARE


# ---------------------------------------------------------------------------
# 3. Linking
# ---------------------------------------------------------------------------
class Linker:
    """Citation -> (source PMCID, source status)."""

    def __init__(self):
        self.licences, self.resolved, _ = run_tables.load_inputs()
        with open(LICENCES_CSV, encoding='utf-8') as f:
            self.has_body = {row['pmcid']: row['has_body'] == '1' for row in csv.DictReader(f)}

    def link(self, ref):
        if ref is None:
            return '', 'no_ref'
        pmcid = run_tables.norm_pmcid(ref['pmcid'])
        if not pmcid:
            for key_type in ('pmid', 'doi'):
                value = (ref[key_type] or '').strip().lower()
                if value and (key_type, value) in self.resolved:
                    pmcid = run_tables.norm_pmcid(self.resolved[(key_type, value)])
                    if not pmcid:
                        return '', 'not_in_pmc'
                    break
            else:
                return '', 'unresolved' if (ref['pmid'] or ref['doi']) else 'no_id'
        if pmcid not in self.licences:
            return pmcid, 'not_downloaded'
        if self.licences[pmcid] != 'CC-BY':
            return pmcid, 'not_ccby'
        return pmcid, 'ccby_full_text' if self.has_body.get(pmcid) else 'ccby_no_body'


# ---------------------------------------------------------------------------
# 4. Items
# ---------------------------------------------------------------------------
def heading_rows(table):
    """Rows that are one merged cell spanning the whole table width.

    Some tables open each study with such a row holding only its name
    ("Vandeweerdt et al [86], 2022"), with the data in the rows below. Maps may
    mark it a study row (right study, right citation), but it is a heading: in
    every column its "answer" is the study's name. Found by the 4.1 hand audit
    (537 such rows, 882 items); v1 had the same detector (banner_row).
    """
    if table['n_cols'] < 2:
        return set()
    return {r for r, row in enumerate(table['grid'])
            if None not in row and len(set(row)) == 1}


def build_table_items(table, record, linker):
    """(items, dropped) for one accepted, mapped table."""
    table_map = record['map']
    review, table_no = table['review_pmcid'], table['table_no']
    headings = heading_rows(table)
    items, dropped = [], []
    for g_index, group in enumerate(groups_of(table_map)):
        ref = table['refs'].get(group['ref']) if group['ref'] else None
        source_pmcid, source_status = linker.link(ref)
        # A study's heading row never supplies answers (see heading_rows); the
        # study's citation still comes through its other rows' entries.
        data_rows = [r for r in group['rows'] if r not in headings]
        study_col = table_map['study_col_index']
        label_cells = {table['grid'][r][study_col] for r in data_rows} - {None}
        for col in table_map['columns']:
            if col['role'] != 'extraction':
                continue
            c = col['col_index']
            if not data_rows:
                dropped.append({'item_id': f'{review}-t{table_no}-g{g_index}-c{c}',
                                'review_pmcid': review, 'table_no': table_no,
                                'rows': group['rows'], 'col_index': c,
                                'drop_reason': 'study_heading_row'})
                continue
            gold = gold_of(table, data_rows, c, label_cells if c != study_col else frozenset())
            if not gold and any(table['grid'][r][c] in label_cells for r in data_rows):
                # Every cell of this column here is the study's own label cell.
                dropped.append({'item_id': f'{review}-t{table_no}-g{g_index}-c{c}',
                                'review_pmcid': review, 'table_no': table_no,
                                'rows': data_rows, 'col_index': c,
                                'drop_reason': 'study_label_cell'})
                continue
            item = {
                'item_id': f'{review}-t{table_no}-g{g_index}-c{c}',
                'review_pmcid': review, 'table_no': table_no, 'table_type': record['table_type'],
                'study_key': group['study_key'], 'rows': data_rows, 'col_index': c,
                'field': col['field'],
                'question': qualified_question(col['question'], group['arm'], group['outcome']),
                'arm': group['arm'], 'outcome': group['outcome'],
                'gold': gold, 'gold_verbatim': verify_gold(gold, table, c),
                'content': col['content'],
                'answerability_category': col['answerability_category'],
                'answerability_score': col['answerability_score'],
                'ref': group['ref'], 'source_pmcid': source_pmcid, 'source_status': source_status,
            }
            if leaks(gold, group['arm'], group['outcome']):
                dropped.append(dict(item, drop_reason='label_reveals_answer'))
            else:
                items.append(item)
    return items, dropped


def main():
    with open(MAPS_PATH, encoding='utf-8') as f:
        records = {(r['review_pmcid'], r['table_no']): {**r, 'map': unescape_map(r['map'])}
                   for r in map(json.loads, f)
                   if r['status'] == 'mapped' and r['map']['accept']}
    linker = Linker()
    n_items, n_tables = 0, 0
    drop_reasons = collections.Counter()
    status = collections.Counter()
    not_verbatim = 0
    with gzip.open(TABLES_PATH, 'rt', encoding='utf-8') as f, \
            open(ITEMS_PATH, 'w', encoding='utf-8') as out, \
            open(DROPPED_PATH, 'w', encoding='utf-8') as out_dropped:
        for line in f:
            table = json.loads(line)
            record = records.get((table['review_pmcid'], table['table_no']))
            if record is None:
                continue
            items, dropped = build_table_items(table, record, linker)
            n_tables += 1
            for item in items:
                out.write(json.dumps(item, ensure_ascii=False) + '\n')
                status[item['source_status']] += 1
                not_verbatim += not item['gold_verbatim']
            for item in dropped:
                out_dropped.write(json.dumps(item, ensure_ascii=False) + '\n')
            n_items += len(items)
            drop_reasons.update(d['drop_reason'] for d in dropped)

    print(f'accepted tables: {n_tables:,}')
    print(f'candidate items: {n_items:,}  (dropped: ' +
          ', '.join(f'{reason} {n:,}' for reason, n in drop_reasons.most_common()) + ')')
    print(f'gold not verbatim: {not_verbatim:,}')
    print('items by source status:')
    for name, count in status.most_common():
        print(f'  {name:<16} {count:>8,}  ({count / n_items:.1%})')
    print(f'-> {ITEMS_PATH}')


if __name__ == '__main__':
    main()
