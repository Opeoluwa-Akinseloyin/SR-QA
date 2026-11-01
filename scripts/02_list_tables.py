"""
02_list_tables.py
-----------------
Stage 2.1 of the v2 pipeline: list EVERY table in every primary review, as a
clean rectangular grid, together with the bibliography entries its cells cite.

Every later Phase 2 stage reads this file instead of re-parsing the XML, so the
table the LLM is shown and the table the gold answers are later copied from
are guaranteed to be the same grid.

Ported from old/scripts/parse_chis_tables.py (XML helpers, references, grid
building) and old/scripts/llm_generate_questions.py (table numbering). The grid
code is unchanged. What differs from v1:

  * every table is kept - v1 scored captions and kept one table per review;
    in v2 the LLM decides, so nothing is picked here;
  * merged cells keep their identity on disk - each physical <td>/<th> gets an
    id and the grid stores ids, so "these three rows share one merged cell"
    survives serialisation (v1 relied on Python object identity in memory);
  * table footnotes are kept - they explain abbreviations and arm labels;
  * header rows are only a HINT (rows made entirely of <th> cells, v1's rule,
    which caused the leaked-header defect); the LLM makes the final call.

Output (data/v2/):
  tables.jsonl.gz   one JSON object per table:
    {
      "review_pmcid": "PMC123", "review_title": "...",
      "table_no": 2,                 # 1-based, counts every <table-wrap> with a
                                     # <table>, as v1 did
      "label": "Table 2", "caption": "...", "foot": "NR, not reported; ...",
      "n_rows": 12, "n_cols": 6,
      "header_rows_hint": 1,
      "cells": [{"text": "...", "rids": ["B12"], "is_header": false,
                 "rowspan": 1, "colspan": 1}, ...],
      "grid": [[0, 1, 1, 2, ...], ...],   # cell id per (row, col); null = gap
      "refs": {"B12": {"text": "...", "short": "Saleh et al., 2022. Title...",
                       "doi": ..., "pmid": ..., "pmcid": ...}}
                                     # only the entries this table cites
    }
  A <table-wrap> whose <table> has no rows is still recorded (empty grid), so
  table numbers stay in step and the count of such tables is visible.

Usage:
  python scripts/02_list_tables.py
  python scripts/02_list_tables.py --limit 20     # first 20 reviews only
"""
import argparse
import csv
import gzip
import json
import os
import xml.etree.ElementTree as ET
from multiprocessing import Pool

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REVIEW_XML_DIR = os.path.join(BASE, 'PMC_XML_')
PRIMARY_CSV = os.path.join(BASE, 'data', 'input', 'fulltext_primary_SRs.csv')
OUT_PATH = os.path.join(BASE, 'data', 'v2', 'tables.jsonl.gz')


# ---------------------------------------------------------------------------
# 1. XML helpers (from v1, unchanged)
# ---------------------------------------------------------------------------
def tag_name(tag):
    """Return an XML tag without its namespace prefix."""
    return tag.rsplit('}', 1)[-1]


def element_text(element):
    """Return all text inside an XML element as one clean, single-spaced string."""
    if element is None:
        return ''
    return ' '.join(''.join(element.itertext()).split())


def find_first(root, name):
    """Return the first element named `name` anywhere under `root`, or None."""
    for element in root.iter():
        if tag_name(element.tag) == name:
            return element
    return None


# ---------------------------------------------------------------------------
# 2. References (from v1, unchanged)
# ---------------------------------------------------------------------------
def spaced_text(element):
    """All text inside an element, with a space between child elements.

    element_text() concatenates text nodes directly, which suits table cells but
    runs a reference's name parts together ("SalehN.Y.IbrahemR.A.L.") - a v1 bug
    that makes matching a row label to its bibliography entry needlessly hard.
    """
    return ' '.join(' '.join(part.split()) for part in element.itertext() if part.strip())


def short_reference(ref_element, full_text):
    """A compact display form: 'Saleh et al., 2022. Surfactant protein D: a predictor ...'.

    The LLM only needs first author, year and title to check that a row's label
    ("Saleh 2022") matches its citation; the full entry (author lists, journal,
    pages, ids) made up ~31% of all prompt tokens. Built from the citation's
    structured fields (present for ~95% of references); otherwise the full text
    cut to 150 characters.
    """
    # Explicit None checks: an XML element with no child elements is falsy, so
    # `a or b` would wrongly skip a plain-text <article-title>.
    first = find_first(ref_element, 'person-group')
    if first is None:
        first = ref_element
    names = [e for e in first.iter() if tag_name(e.tag) in ('name', 'string-name', 'collab')]
    surname = find_first(first, 'surname')
    collab = find_first(first, 'collab')
    year = find_first(ref_element, 'year')
    title = find_first(ref_element, 'article-title')
    if title is None:
        title = find_first(ref_element, 'source')
    if (surname is None and collab is None) or year is None:
        return full_text[:150]
    author = element_text(surname if surname is not None else collab)
    if len(names) == 2:
        second = [e for e in first.iter() if tag_name(e.tag) == 'surname'][1:2]
        if second:
            author += ' and ' + element_text(second[0])
    elif len(names) > 2:
        author += ' et al.'
    text = f'{author}, {element_text(year)}.'
    if title is not None:
        title_text = element_text(title)
        text += ' ' + (title_text[:100] + '...' if len(title_text) > 100 else title_text)
    return text


def parse_reference(ref_element):
    """Pull DOI / PMID / PMCID, the full text and a short display form out of one <ref>."""
    doi = pmid = pmcid = None
    for element in ref_element.iter():
        if tag_name(element.tag) != 'pub-id':
            continue
        id_type = (element.get('pub-id-type') or '').lower()
        id_value = (element.text or '').strip()
        if id_type == 'doi' and not doi:
            doi = id_value
        elif id_type == 'pmid' and not pmid:
            pmid = id_value
        elif id_type == 'pmcid' and not pmcid:
            pmcid = id_value
    text = spaced_text(ref_element)[:300]
    return {'text': text, 'short': short_reference(ref_element, text),
            'doi': doi, 'pmid': pmid, 'pmcid': pmcid}


def build_reference_map(root):
    """Map each reference id ('rid') to its parsed reference dict."""
    reference_map = {}
    for element in root.iter():
        if tag_name(element.tag) == 'ref':
            rid = element.get('id')
            if rid:
                reference_map[rid] = parse_reference(element)
    return reference_map


# ---------------------------------------------------------------------------
# 3. Grid building (from v1; the only change is that each cell gets an id)
# ---------------------------------------------------------------------------
def read_span(cell_element, attribute):
    """Read a colspan/rowspan attribute; missing or bad values count as 1."""
    try:
        return max(1, int(cell_element.get(attribute)))
    except (TypeError, ValueError):
        return 1


def make_cell(cell_element, reference_map, cell_id):
    """Convert one <td>/<th> element into a plain dict, with its id."""
    rids = []
    for element in cell_element.iter():
        if tag_name(element.tag) != 'xref':
            continue
        rid = element.get('rid')
        is_citation = element.get('ref-type') == 'bibr' or rid in reference_map
        if rid and is_citation:
            # One rid attribute may list several ids separated by spaces.
            for single_rid in rid.split():
                if single_rid in reference_map:
                    rids.append(single_rid)
    return {
        'id': cell_id,
        'text': element_text(cell_element),
        'colspan': read_span(cell_element, 'colspan'),
        'rowspan': read_span(cell_element, 'rowspan'),
        'is_header': tag_name(cell_element.tag) == 'th',
        'rids': rids,
    }


def build_grid(table_element, reference_map):
    """Reconstruct a rectangular grid from the table, honouring merged cells.

    Returns (grid, cells): grid is a list of rows, each a dict
    {column_index: cell}, with a merged cell repeated at every position it
    covers; cells lists every physical cell once, indexed by its id.
    """
    table_rows = [e for e in table_element.iter() if tag_name(e.tag) == 'tr']
    cells = []
    # Cells hanging down into later rows because of a rowspan:
    # {column_index: {'rows_left': n, 'cell': cell}}
    carried_cells = {}
    grid = []
    for table_row in table_rows:
        physical_cells = [c for c in table_row if tag_name(c.tag) in ('td', 'th')]
        row = {}
        col = 0
        next_physical = 0
        safety_counter = 0
        while next_physical < len(physical_cells) or (carried_cells and col <= max(carried_cells)):
            safety_counter += 1
            if safety_counter > 1000:
                break            # backstop against malformed tables
            if col in carried_cells:
                carried = carried_cells[col]['cell']
                row[col] = carried
                col += carried['colspan']
                continue
            if next_physical < len(physical_cells):
                cell = make_cell(physical_cells[next_physical], reference_map, len(cells))
                cells.append(cell)
                next_physical += 1
                for covered_col in range(col, col + cell['colspan']):
                    row[covered_col] = cell
                    if cell['rowspan'] > 1:
                        carried_cells[covered_col] = {'rows_left': cell['rowspan'], 'cell': cell}
                col += cell['colspan']
                continue
            break

        still_carried = {}
        for covered_col, carried in carried_cells.items():
            if carried['rows_left'] - 1 > 0:
                still_carried[covered_col] = {'rows_left': carried['rows_left'] - 1,
                                              'cell': carried['cell']}
        carried_cells = still_carried
        grid.append(row)
    return grid, cells


def grid_size(grid):
    """Return (number_of_rows, number_of_columns) of a grid."""
    n_cols = max((max(row) + 1 for row in grid if row), default=0)
    return len(grid), n_cols


def header_rows_hint(grid):
    """v1's header rule, kept as a hint only: leading rows made entirely of <th>."""
    count = 0
    for row in grid:
        if not row or not all(cell['is_header'] for cell in row.values()):
            break
        count += 1
    return count


# ---------------------------------------------------------------------------
# 4. One review -> its tables
# ---------------------------------------------------------------------------
def table_foot(table_wrap):
    """Text of the table's footnotes (<table-wrap-foot>), or ''."""
    foot = find_first(table_wrap, 'table-wrap-foot')
    return element_text(foot)


def review_tables(job):
    """Parse one review and return one record per table, in document order.

    table_no counts every <table-wrap> that contains a <table>, exactly as v1
    did, so "table 3" means the same physical table in v1 and v2.
    """
    pmcid, title = job
    try:
        root = ET.parse(os.path.join(REVIEW_XML_DIR, pmcid + '.xml')).getroot()
    except (ET.ParseError, OSError) as error:
        return [{'review_pmcid': pmcid, 'error': str(error)}]
    reference_map = build_reference_map(root)

    records = []
    table_no = 0
    for table_wrap in root.iter():
        if tag_name(table_wrap.tag) != 'table-wrap':
            continue
        table_element = find_first(table_wrap, 'table')
        if table_element is None:
            continue
        table_no += 1
        label = caption = ''
        for child in table_wrap:
            if tag_name(child.tag) == 'label':
                label = element_text(child)
            elif tag_name(child.tag) == 'caption':
                caption = element_text(child)

        grid, cells = build_grid(table_element, reference_map)
        n_rows, n_cols = grid_size(grid)
        cited = sorted({rid for cell in cells for rid in cell['rids']})
        records.append({
            'review_pmcid': pmcid,
            'review_title': title,
            'table_no': table_no,
            'label': label,
            'caption': caption,
            'foot': table_foot(table_wrap),
            'n_rows': n_rows,
            'n_cols': n_cols,
            'header_rows_hint': header_rows_hint(grid),
            'cells': [{k: c[k] for k in ('text', 'rids', 'is_header', 'rowspan', 'colspan')}
                      for c in cells],
            'grid': [[row[col]['id'] if col in row else None for col in range(n_cols)]
                     for row in grid],
            'refs': {rid: reference_map[rid] for rid in cited},
        })
    return records


# ---------------------------------------------------------------------------
# 5. Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=0, help='first N reviews only')
    args = parser.parse_args()

    with open(PRIMARY_CSV, encoding='utf-8') as f:
        jobs = [(row['PMCID'], row['Title']) for row in csv.DictReader(f)]
    if args.limit:
        jobs = jobs[:args.limit]

    n_tables = n_empty = n_errors = n_with_links = 0
    reviews_with_tables = set()
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with gzip.open(OUT_PATH, 'wt', encoding='utf-8') as out, Pool() as pool:
        # imap keeps document order within the output file (review by review).
        for records in pool.imap(review_tables, jobs, chunksize=20):
            for record in records:
                if 'error' in record:
                    n_errors += 1
                    continue
                out.write(json.dumps(record, ensure_ascii=False) + '\n')
                n_tables += 1
                reviews_with_tables.add(record['review_pmcid'])
                if not record['grid']:
                    n_empty += 1
                if record['refs']:
                    n_with_links += 1

    print(f'reviews read            : {len(jobs):,}  (unreadable: {n_errors})')
    print(f'reviews with >=1 table  : {len(reviews_with_tables):,}')
    print(f'tables listed           : {n_tables:,}')
    print(f'  with an empty grid    : {n_empty:,}')
    print(f'  citing >=1 reference  : {n_with_links:,}')
    print(f'-> {OUT_PATH}')


if __name__ == '__main__':
    main()
