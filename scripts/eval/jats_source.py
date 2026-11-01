"""JATS source-paper parsing for the zero-shot QA runner.

This module replaces the flatten-everything-with-itertext() reading of a source
paper that `run_zeroshot_qa.py` used to do. Three problems it fixes:

1. **Tables outside <body> were invisible.** In JATS a <table-wrap> may sit in
   <body>, or in <floats-group> - a sibling of <body> that holds every float the
   publisher pulled out of the running text. A `.//body` walk misses the latter
   entirely. Measured on a 300-paper sample of SOURCE_XML/, 22.6% of the papers
   that have any table have *all* of their tables in <floats-group>: for those
   papers the model was answering table questions from a document with no tables
   in it. We now collect floats from both places.

2. **Flattening destroyed the cell bindings.** ' '.join(itertext()) turns a grid
   into a bag of words: the value loses its row and its column header, which is
   exactly the binding a data-extraction question asks about. We rebuild the
   grid (honouring rowspan/colspan) with the same helpers the ChIS parser uses,
   then serialise it two ways - a Markdown table that preserves the visual
   layout, and an explicit `[row=... | header = value]` cell list that restates
   the binding for every value-bearing cell, so a multi-arm row survives even if
   the model reads the block linearly. Footnotes ride along because that is
   where units and abbreviations are defined.

3. **Answer-bearing text sat in the middle of the prompt.** Long-context recall
   sags in the middle of a document, so sections are re-ordered into zones:
   the sections that answer extraction questions (Methods, Results, tables) go
   in the primacy zone, and Introduction / Discussion / back matter - which
   mostly restate or speculate - go in the low-attention middle. The same
   ranking drives truncation, so when a paper exceeds the character cap it is
   the Discussion that goes, never the Results.

Public API (kept compatible with the old in-runner version):

    parse_source(path, max_chars, ...) -> {title, sections:[{id,title,text}],
                                           n_chars, truncated, ...}
    render_source(src)                 -> the string that goes in the prompt

`sections` stays a flat ordered list of {id, title, text}, so quote location and
the judge's section lookup keep working unchanged.

v2 port (2026-10-09) of old/scripts/jats_source.py. The only change: the grid is
built by v2's own build_grid (scripts/02_list_tables.py, which read the review
tables the items come from) instead of v1's parse_chis_tables, so the review
side and the source side read a table by identical rules. header_rows and
column_headers are copied from v1's parse_chis_tables unchanged.
"""
import importlib.util
import os
import re
import xml.etree.ElementTree as ET

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# The grid reconstruction is shared with the review-table lister rather than
# reimplemented: rowspan/colspan handling is fiddly and having two copies drift
# apart would mean the review side and the source side disagree about what a
# table says. 02_list_tables.py has a digit-led name, hence importlib.
_spec = importlib.util.spec_from_file_location(
    'list_tables', os.path.join(BASE, 'scripts', '02_list_tables.py'))
_list_tables = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_list_tables)


def build_grid(table_element, reference_map):
    """v2's grid: a list of rows, each {column_index: cell}; cells carry text/is_header."""
    grid, _cells = _list_tables.build_grid(table_element, reference_map)
    return grid


def header_rows(grid):
    """Return the leading rows that consist entirely of header (<th>) cells."""
    rows = []
    for row in grid:
        cells = list(row.values())
        if not cells or not all(cell['is_header'] for cell in cells):
            break
        rows.append(row)
    return rows


def column_headers(grid, n_cols):
    """Build one header string per column by joining the stacked header rows.

    A column sitting under 'Age (years)' and then 'Mean' becomes
    'Age (years) - Mean'. Returns (headers, number_of_header_rows).
    """
    top_rows = header_rows(grid)
    headers = []
    for col in range(n_cols):
        parts = []
        for row in top_rows:
            if col in row:
                text = row[col]['text']
                # Skip empty text and immediate repeats (from merged cells).
                if text and (not parts or parts[-1] != text):
                    parts.append(text)
        headers.append(' - '.join(parts))
    return headers, len(top_rows)

# --------------------------------------------------------------------------- #
# small XML helpers
# --------------------------------------------------------------------------- #
FLOAT_TAGS = ("table-wrap", "fig")

# Tags whose subtree must not be inlined into a section's prose: floats are
# emitted separately (with their structure intact) and would otherwise appear
# twice - once flattened, once properly.
PROSE_SKIP_TAGS = frozenset(("table-wrap", "fig", "table-wrap-foot", "graphic",
                             "media", "supplementary-material"))


def _tag(t):
    return t.split("}")[-1]


def _squash(s):
    return re.sub(r"\s+", " ", s or "").strip()


def _text(el):
    """All text under `el`, whitespace-normalised. Used where structure is moot."""
    if el is None:
        return ""
    return _squash("".join(el.itertext()))


def _prose(el, float_names=None, skip=PROSE_SKIP_TAGS):
    """Serialise `el` as running prose.

    Differs from _text in two ways that matter: subtrees listed in `skip` are
    dropped (floats are emitted separately), and a cross-reference to a float is
    rewritten to that float's display name. A bare <xref ref-type="table"> often
    contains just "2", which flattens to a stray digit; rewriting it to "Table 2"
    is what lets the model follow a sentence like "as shown in Table 2" to the
    block where Table 2 actually lives.
    """
    float_names = float_names or {}
    out = []

    def walk(node):
        for child in node:
            name = _tag(child.tag)
            if name in skip:
                if child.tail:
                    out.append(child.tail)
                continue
            if name == "xref" and child.get("ref-type") in ("table", "fig"):
                resolved = [float_names[r] for r in (child.get("rid") or "").split()
                            if r in float_names]
                out.append(", ".join(dict.fromkeys(resolved)) if resolved
                           else "".join(child.itertext()))
                if child.tail:
                    out.append(child.tail)
                continue
            if child.text:
                out.append(child.text)
            walk(child)
            if child.tail:
                out.append(child.tail)

    if el.text:
        out.append(el.text)
    walk(el)
    return _squash("".join(out))


def _child_named(el, name):
    for child in el:
        if _tag(child.tag) == name:
            return child
    return None


# --------------------------------------------------------------------------- #
# 1. float collection: <body> AND <floats-group>
# --------------------------------------------------------------------------- #
def collect_floats(root):
    """Return (tables, figures, names) gathered from <body> and <floats-group>.

    `names` maps every float's @id to its display name ("Table 2"), which is how
    an in-text <xref ref-type="table" rid=...> gets resolved.

    Document order is preserved and a float reachable from both containers is
    kept once - some publishers leave a stub <table-wrap> in the body and the
    real one in <floats-group>, and the two would otherwise both be emitted.
    """
    containers = []
    for name in ("body", "floats-group"):
        for el in root.iter():
            if _tag(el.tag) == name:
                containers.append(el)

    tables, figures, names = [], [], {}
    seen = set()
    counters = {"table-wrap": 0, "fig": 0}

    for container in containers:
        for el in container.iter():
            kind = _tag(el.tag)
            if kind not in FLOAT_TAGS or id(el) in seen:
                continue
            seen.add(id(el))
            counters[kind] += 1

            label = _text(_child_named(el, "label"))
            caption = _text(_child_named(el, "caption"))
            fallback = ("Table " if kind == "table-wrap" else "Figure ") + str(counters[kind])
            display = label or fallback

            entry = {"kind": kind, "xml_id": el.get("id") or "", "label": label,
                     "display": display, "caption": caption, "element": el}
            (tables if kind == "table-wrap" else figures).append(entry)
            if entry["xml_id"]:
                names[entry["xml_id"]] = display

    return tables, figures, names


# --------------------------------------------------------------------------- #
# 2. structure-preserving table serialisation
# --------------------------------------------------------------------------- #
# A cell has to contain a digit to earn a line in the explicit cell list;
# otherwise every "Yes"/"NR" in a 30-column table would get one.
HAS_DIGIT = re.compile(r"\d")
BLANK_CELL = re.compile(r"^\s*[-–—.]*\s*$")

# Upper bound on how many leading columns may be treated as the row's identity.
# Compound stubs are real but shallow; without a cap a malformed header row
# would let the stub eat the whole table and emit no values at all.
MAX_STUB_COLS = 3


def _md_escape(text):
    return (text or "").replace("|", "\\|").replace("\n", " ")


def _grid_width(grid):
    return max((max(row) + 1 for row in grid if row), default=0)


def _thead_row_count(table_element):
    """Number of leading <tr>s that live inside <thead>.

    `column_headers` decides what a header row is by looking for <th> cells, but
    plenty of journals mark their header row up as <thead><tr><td>. Counting the
    <thead> rows recovers the header for those tables instead of composing an
    empty one.
    """
    for el in table_element.iter():
        if _tag(el.tag) == "thead":
            return sum(1 for x in el.iter() if _tag(x.tag) == "tr")
    return 0


def _compose_headers(grid, n_cols, n_header_rows):
    """Join the stacked header rows into one string per column.

    Same rule as parse_chis_tables.column_headers ('Age (years) - Mean'), but
    over an explicit row count so a <thead> of <td>s can be honoured too.
    """
    headers = []
    for col in range(n_cols):
        parts = []
        for row in grid[:n_header_rows]:
            cell = row.get(col)
            if cell is None:
                continue
            text = _squash(cell["text"])
            if text and (not parts or parts[-1] != text):
                parts.append(text)
        headers.append(" - ".join(parts))
    return headers


def serialize_table(entry, max_chars=20000, max_cell_lines=400):
    """Render one <table-wrap> as label + caption + Markdown grid + cell list + notes.

    A <table-wrap> may hold several <table> elements (multi-part tables, "Table
    3a / 3b"); each is rendered as its own part under the shared caption.
    """
    element = entry["element"]
    lines = []
    heading = entry["display"]
    if entry["caption"]:
        heading += ": " + entry["caption"]
    lines.append(heading)

    tables = [el for el in element.iter() if _tag(el.tag) == "table"]
    n_value_cells = 0

    for part_no, table_element in enumerate(tables, 1):
        grid = build_grid(table_element, {})
        if not grid:
            continue
        n_cols = _grid_width(grid)
        if n_cols == 0:
            continue

        n_header_rows = _thead_row_count(table_element)
        if n_header_rows and n_header_rows < len(grid):
            headers = _compose_headers(grid, n_cols, n_header_rows)
        else:
            # No usable <thead>: fall back to the <th>-based rule.
            headers, n_header_rows = column_headers(grid, n_cols)
        if not any(headers):
            headers, n_header_rows = column_headers(grid, n_cols)

        # How many leading columns are the row's identity rather than data. A
        # table may have a compound stub ("Gender | Male", "Gender | Female"):
        # if only column 0 were the row label, the Male and Female rows would
        # both bind to row=Gender and their values would be indistinguishable.
        # An unheaded leading column is the tell-tale of such a qualifier.
        n_stub = 1
        while n_stub < min(n_cols - 1, MAX_STUB_COLS) and not headers[n_stub]:
            n_stub += 1

        headers = [h or f"col{i + 1}" for i, h in enumerate(headers)]
        data_rows = grid[n_header_rows:]

        if len(tables) > 1:
            lines.append("")
            lines.append(f"(part {part_no} of {len(tables)})")

        # --- the visual grid ------------------------------------------------ #
        lines.append("")
        lines.append("| " + " | ".join(_md_escape(h) for h in headers) + " |")
        lines.append("|" + "|".join(["---"] * n_cols) + "|")
        for row in data_rows:
            cells = [_squash(row[c]["text"]) if c in row else "" for c in range(n_cols)]
            lines.append("| " + " | ".join(_md_escape(c) for c in cells) + " |")

        # --- the explicit cell list ----------------------------------------- #
        # Belt and braces for the wide multi-arm tables where a Markdown row is
        # long enough that the value and its column header end up far apart:
        # each line restates the full binding (row label + composed header +
        # value), which is the PropRAG rule - the three never travel separately.
        cell_lines = []
        carried = [""] * n_stub
        for row in data_rows:
            for i in range(n_stub):
                value = _squash(row[i]["text"]) if i in row else ""
                if value:
                    carried[i] = value
                    # A new group heading invalidates the qualifiers beneath it,
                    # so "Age" does not inherit the previous row's "Female".
                    for j in range(i + 1, n_stub):
                        carried[j] = ""
            row_label = ", ".join(x for x in carried if x)
            if not row_label:
                continue
            for col in range(n_stub, n_cols):
                cell = row.get(col)
                if cell is None:
                    continue
                value = _squash(cell["text"])
                if not value or BLANK_CELL.match(value) or value == row_label:
                    continue
                if not HAS_DIGIT.search(value):
                    continue
                cell_lines.append(f"[row={row_label} | {headers[col]} = {value}]")

        cell_lines = list(dict.fromkeys(cell_lines))
        n_value_cells += len(cell_lines)
        if cell_lines:
            lines.append("")
            lines.append(f"Cell values of {entry['display']}"
                         + (f" part {part_no}" if len(tables) > 1 else "") + ":")
            if len(cell_lines) > max_cell_lines:
                lines.extend(cell_lines[:max_cell_lines])
                lines.append(f"[...{len(cell_lines) - max_cell_lines} further cells "
                             f"omitted; see the grid above...]")
            else:
                lines.extend(cell_lines)

    # --- footnotes: units and abbreviation expansions live here -------------- #
    notes = []
    for el in element.iter():
        if _tag(el.tag) in ("table-wrap-foot", "fn"):
            text = _text(el)
            if text and text not in notes:
                notes.append(text)
    if notes:
        lines.append("")
        lines.append("Notes: " + " ".join(notes))

    text = "\n".join(lines).strip()
    truncated = len(text) > max_chars
    if truncated:
        text = text[:max_chars] + "\n[...table truncated for length...]"
    return {"title": entry["display"], "text": text, "n_value_cells": n_value_cells,
            "truncated": truncated}


# --------------------------------------------------------------------------- #
# 3. section walking and zoning
# --------------------------------------------------------------------------- #
def _collect_sections(el, path, out, float_names):
    """Walk a <sec> tree, emitting one entry per section that has its own prose.

    A section's own text excludes its nested <sec> children - those become their
    own entries titled with the full path - so a located quote resolves as
    specifically as the paper's own structure allows.
    """
    title = ""
    for child in el:
        if _tag(child.tag) == "title":
            title = _text(child)
            break
    new_path = path + [title] if title else path

    own = []
    for child in el:
        if _tag(child.tag) in ("sec", "title"):
            continue
        own.append(_prose(child, float_names))
    body_text = _squash(" ".join(x for x in own if x))
    if body_text:
        out.append({"title": " > ".join(new_path) or "Body", "text": body_text})

    for child in el:
        if _tag(child.tag) == "sec":
            _collect_sections(child, new_path, out, float_names)


# Zones, in the order they are presented to the model. `keep_rank` is how hard a
# zone is to dislodge: the LOWEST rank is sacrificed first, so a paper over the
# cap loses its Discussion long before it loses a Results table. Methods and
# Results outrank the Abstract because they are where an extraction question is
# actually answerable; the Abstract is a summary of them.
ZONES = {
    "abstract":   {"order": 0, "keep_rank": 3},
    "methods":    {"order": 1, "keep_rank": 7},
    "results":    {"order": 2, "keep_rank": 6},
    "content":    {"order": 3, "keep_rank": 5},
    "tables":     {"order": 4, "keep_rank": 4},
    "background": {"order": 5, "keep_rank": 1},
    "discussion": {"order": 5, "keep_rank": 0},
    "other":      {"order": 6, "keep_rank": 2},
}

# Matched against the section title, most specific first. Order matters:
# "Statistical analysis" must land in methods before "analysis" can pull it
# anywhere else, and "Results and discussion" is scored as results.
ZONE_PATTERNS = [
    ("methods", (
        "method", "material", "patients and", "subjects and", "study design",
        "study population", "participants", "procedure", "measures", "measurement",
        "instrument", "intervention", "data collection", "data extraction",
        "statistical analys", "statistical method", "experimental", "protocol",
        "sample size", "randomi", "inclusion", "eligib", "setting", "outcome measure",
    )),
    ("results", ("result", "finding", "outcome")),
    ("discussion", (
        "discussion", "conclusion", "limitation", "implication", "interpretation",
        "future direction", "perspective", "comparison with",
    )),
    ("background", ("introduction", "background", "rationale", "objective", "aim")),
    # True back matter - the only things safe to shed early. Everything that
    # matches nothing at all is NOT put here; see the fallback below.
    ("other", (
        "acknowledg", "funding", "conflict", "competing interest", "author contribution",
        "ethic", "consent", "availability", "abbreviation", "supplementary",
        "supporting information", "appendix", "reference", "disclosure",
    )),
]


def classify_section(title):
    """Assign a section to a zone from its title.

    Nested titles arrive as a path ("Methods > Statistical analysis"), and the
    top component is tried first so a subsection inherits its parent's zone -
    that is what keeps Methods contiguous instead of scattering its children.

    An unrecognised title falls back to "content", not to back matter. Plenty of
    article types carry their substance under headings that are not IMRaD at all
    - a case report's "Case presentation", "Final diagnosis", "The Study", or an
    unstructured "Body" - and in the v2 run those sections supplied 52 verified
    answers. Treating "no keyword matched" as "probably disposable" would rank
    exactly that content for early truncation.
    """
    if not title:
        return "content"
    lowered = title.lower()
    top = lowered.split(" > ")[0]
    if top.startswith("abstract"):
        return "abstract"
    for candidate in (top, lowered):
        for zone, needles in ZONE_PATTERNS:
            for needle in needles:
                if needle in candidate:
                    return zone
    return "content"


# --------------------------------------------------------------------------- #
# 4. the public entry point
# --------------------------------------------------------------------------- #
def parse_source(path, max_chars, layout="sectioned", include_tables=True,
                 max_table_chars=20000):
    """Parse a source paper into ordered, labelled sections.

    layout="sectioned"  zone-ordered (Methods/Results/tables first, Introduction
                        and Discussion in the low-attention middle)
    layout="linear"     document order, with any tables appended at the end -
                        the v1 reading, kept so the two can be compared

    Returns {title, sections:[{id,title,text}], n_chars, truncated, ...}; the
    section list is flat and ordered, and `id` is assigned in presentation order.
    """
    root = ET.parse(path).getroot()

    title = ""
    t = root.find(".//front//article-title")
    if t is not None:
        title = _text(t)

    tables, figures, float_names = collect_floats(root)

    sections = []
    abstract = root.find(".//front//abstract")
    if abstract is not None and _prose(abstract, float_names):
        sections.append({"title": "Abstract", "text": _prose(abstract, float_names),
                         "zone": "abstract"})

    body = root.find(".//body")
    if body is not None:
        found = []
        _collect_sections(body, [], found, float_names)
        if not found:
            # A body with no <sec> structure: keep it whole rather than lose it.
            flat = _prose(body, float_names)
            if flat:
                found = [{"title": "Body", "text": flat}]
        for s in found:
            s["zone"] = classify_section(s["title"])
        sections.extend(found)

    n_tables_found = len(tables)
    n_value_cells = 0
    if include_tables:
        for entry in tables:
            rendered = serialize_table(entry, max_chars=max_table_chars)
            if rendered["text"]:
                n_value_cells += rendered["n_value_cells"]
                sections.append({"title": rendered["title"], "text": rendered["text"],
                                 "zone": "tables", "_is_table": True})
        captions = [f"{f['display']}: {f['caption']}" for f in figures if f["caption"]]
        if captions:
            sections.append({"title": "Figure captions",
                             "text": "\n".join(captions), "zone": "tables"})

    # ---- order and truncate ------------------------------------------------ #
    for i, s in enumerate(sections):
        s["_doc_pos"] = i

    if layout == "sectioned":
        ordered = sorted(sections, key=lambda s: (ZONES[s["zone"]]["order"], s["_doc_pos"]))
    else:
        # linear: document order, tables last (they have no position in the prose)
        ordered = sorted(sections, key=lambda s: (s["zone"] == "tables", s["_doc_pos"]))

    truncated = False
    total = sum(len(s["text"]) for s in ordered)
    if total > max_chars:
        # Drop whole sections cheapest-zone-first (Discussion goes before
        # Methods), and within a zone from the back of the paper, until it fits.
        truncated = True
        victims = sorted(
            range(len(ordered)),
            key=lambda i: (ZONES[ordered[i]["zone"]]["keep_rank"], -ordered[i]["_doc_pos"]),
        )
        dropped = set()
        for i in victims:
            if total <= max_chars:
                break
            dropped.add(i)
            total -= len(ordered[i]["text"])
        kept = [s for i, s in enumerate(ordered) if i not in dropped]
        # Everything droppable is gone and it still does not fit: hard-cut the tail.
        if total > max_chars:
            used, hard = 0, []
            for s in kept:
                if used + len(s["text"]) > max_chars:
                    room = max_chars - used
                    if room > 500:
                        hard.append({**s, "text": s["text"][:room]})
                    break
                hard.append(s)
                used += len(s["text"])
            kept = hard
    else:
        kept = list(ordered)

    for i, s in enumerate(kept, 1):
        s["id"] = f"S{i}"

    return {
        "title": title,
        "sections": [{"id": s["id"], "title": s["title"], "text": s["text"],
                      "zone": s["zone"]} for s in kept],
        "n_chars": sum(len(s["text"]) for s in kept),
        "truncated": truncated,
        "layout": layout,
        "n_tables_found": n_tables_found,
        "n_tables_kept": sum(1 for s in kept if s.get("_is_table")),
        "n_table_value_cells": n_value_cells,
    }


# --------------------------------------------------------------------------- #
# 5. rendering
# --------------------------------------------------------------------------- #
ZONE_BANNERS = {
    "abstract": None,
    "methods": None,
    "results": None,
    "content": None,
    "tables": "=== TABLES (values below are bound to their row and column header) ===",
    "background": "=== BACKGROUND / DISCUSSION / OTHER (context; rarely the source of an "
                  "extracted value) ===",
    "discussion": None,
    "other": None,
}
# background and discussion share one banner; 'other' trails them without its own.
_BANNER_ZONE = {"background": "background", "discussion": "background", "other": "background"}


def render_source(src):
    """Render a parsed source into the STUDY FULL TEXT block of the prompt."""
    parts = [f"TITLE: {src['title']}" if src["title"] else "TITLE: (none)"]
    shown_banners = set()
    for s in src["sections"]:
        zone = s.get("zone", "other")
        banner_key = _BANNER_ZONE.get(zone, zone)
        banner = ZONE_BANNERS.get(banner_key)
        if banner and banner_key not in shown_banners:
            shown_banners.add(banner_key)
            parts.append(banner)
        parts.append(f"[{s['id']}] {s['title']}\n{s['text']}")
    if src["truncated"]:
        parts.append("[...lower-priority sections of the full text truncated for length...]")
    return "\n\n".join(parts)
