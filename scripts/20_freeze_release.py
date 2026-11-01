"""
20_freeze_release.py
--------------------
Stage 4.3 of the v2 pipeline: freeze SR-QA v2.0 into release/v2/, the exact
files the ECIR submission and the Zenodo / Hugging Face release describe.

  data/<split>.jsonl.gz   the items, one file per split, in the public schema
                          (PUBLIC_FIELDS): `split` is the date split, `arm` and
                          `outcome` are the item's group labels
  reviews.csv             per review: title, journal, online date, issue date,
                          OpenAlex field, split
  review_context.jsonl    per review: title, objectives, eligibility (the
                          review-side input; extractor ported from
                          old/scripts/construct_qa.py extract_context)
  sources.csv             per source paper: licence, PMID, DOI, design (+ where
                          the label came from), publisher (parent company), year
  table_maps.jsonl.gz     the table maps every item is rebuilt from
  quality/                funnel, bias table, answerability check, integrity
                          checks, audit judgements
  DATASHEET.md, ...       documentation, copied from docs/release/
  MANIFEST.json           version, counts, and SHA-256 of every file above, of
                          every pipeline script, and of each source XML - so
                          v2.0 is tied to the exact code and texts behind it

Source full texts are not copied here (they are packaged for the release in
Phase 8); their checksums are fixed now so that package can be verified.

Usage:
  python scripts/20_freeze_release.py
"""
import collections
import csv
import datetime
import gzip
import hashlib
import io
import json
import os
import re
import shutil
import xml.etree.ElementTree as ET
from multiprocessing import Pool

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V2 = os.path.join(BASE, 'data', 'v2')
OUT = os.path.join(BASE, 'release', 'v2')
REVIEW_XML_DIR = os.path.join(BASE, 'PMC_XML_')
SOURCE_DIR = os.path.join(BASE, 'SOURCE_XML')
VERSION = '2.0.1'   # 2026-10-14: symbol-only gold fixes (11_filter_items)

PUBLIC_FIELDS = ['item_id', 'split', 'review_pmcid', 'review_online_date', 'table_no',
                 'table_type', 'study_key', 'rows', 'col_index', 'field', 'question', 'gold',
                 'gold_is_absence', 'arm', 'outcome', 'content', 'answerability_category',
                 'answerability_score', 'source_pmcid', 'citation_rid']
SPLITS = ['train-2025', 'train-2026', 'dev-2026', 'test-2026', 'test-2025']
QUALITY_FILES = ['funnel.csv', 'bias_table.csv', 'bias_tvd.csv', 'checks_report.json',
                 'answerability_check.csv']


def write_gz(path):
    """A text gzip whose bytes depend only on its content: a fixed header time
    (gzip stores the write time by default) and '\\n' line endings on every OS,
    so rebuilding identical content anywhere reproduces the MANIFEST checksums."""
    raw = open(path, 'wb')
    gz = gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0)
    text = io.TextIOWrapper(gz, encoding='utf-8', newline='\n')
    text.raw_file = raw          # closed with the wrapper, see close_gz
    return text


def close_gz(text):
    text.close()
    text.raw_file.close()


def copy_normalised(src, dst):
    """Copy a file; JSON / JSONL / Markdown get '\\n' line endings so the copy is
    the same on every OS. CSVs are copied byte for byte (the csv module writes
    '\\r\\n' on every OS, as RFC 4180 specifies)."""
    if os.path.splitext(src)[1] in ('.json', '.jsonl', '.md'):
        with open(src, encoding='utf-8', newline=None) as f:
            text = f.read()
        with open(dst, 'w', encoding='utf-8', newline='\n') as f:
            f.write(text)
    else:
        shutil.copy(src, dst)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Review context (from v1's construct_qa.extract_context, unchanged)
# ---------------------------------------------------------------------------
def tag_name(tag):
    return tag.rsplit('}', 1)[-1]


def element_text(element):
    return ' '.join(''.join(element.itertext()).split()) if element is not None else ''


def review_context(pmcid):
    """Title, objectives (abstract as fallback) and eligibility of one review."""
    try:
        root = ET.parse(os.path.join(REVIEW_XML_DIR, pmcid + '.xml')).getroot()
    except (ET.ParseError, OSError):
        return {'review_pmcid': pmcid, 'title': '', 'objectives': '', 'eligibility': ''}
    first = lambda name: next((e for e in root.iter() if tag_name(e.tag) == name), None)
    title, abstract, body = element_text(first('article-title')), element_text(first('abstract')), first('body')
    objectives = eligibility = ''
    if body is not None:
        for section in body.iter():
            if tag_name(section.tag) != 'sec':
                continue
            heading = next((c for c in section if tag_name(c.tag) == 'title'), None)
            heading = ' '.join(element_text(heading).lower().split())
            if not heading:
                continue
            if not objectives and re.search(r'objective|aim|purpose|rationale|background', heading):
                objectives = element_text(section)[:4000]
            if not eligibility and re.search(r'eligib|inclusion|selection criteria|study selection',
                                             heading):
                eligibility = element_text(section)[:4000]
    return {'review_pmcid': pmcid, 'title': title, 'objectives': (objectives or abstract)[:4000],
            'eligibility': eligibility[:4000]}


def hash_source(pmcid):
    return pmcid, sha256(os.path.join(SOURCE_DIR, pmcid + '.xml'))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    if os.path.exists(OUT):
        shutil.rmtree(OUT)                      # a freeze is rebuilt whole, never patched
    os.makedirs(os.path.join(OUT, 'data'))
    os.makedirs(os.path.join(OUT, 'quality'))

    with open(os.path.join(V2, 'items_split.jsonl'), encoding='utf-8') as f:
        items = [json.loads(line) for line in f]
    by_split = collections.defaultdict(list)
    for item in items:
        record = dict(item, review_online_date=item['review_online'], citation_rid=item['ref'])
        by_split[item['split']].append({k: record[k] for k in PUBLIC_FIELDS})
    for split in SPLITS:
        f = write_gz(os.path.join(OUT, 'data', f'{split}.jsonl.gz'))
        for record in sorted(by_split[split], key=lambda r: r['item_id']):
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
        close_gz(f)

    reviews = sorted({i['review_pmcid'] for i in items})
    papers = sorted({i['source_pmcid'] for i in items})
    tables = {(i['review_pmcid'], i['table_no']) for i in items}

    # reviews.csv
    with open(os.path.join(BASE, 'data', 'input', 'fulltext_primary_SRs.csv'), encoding='utf-8') as f:
        meta = {r['PMCID']: r for r in csv.DictReader(f)}
    with open(os.path.join(V2, 'review_splits.csv'), encoding='utf-8') as f:
        splits = {r['review_pmcid']: r for r in csv.DictReader(f)}
    with open(os.path.join(OUT, 'reviews.csv'), 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['review_pmcid', 'doi', 'title', 'journal', 'online_date', 'issue_date',
                         'field', 'split'])
        for r in reviews:
            m, s = meta.get(r, {}), splits[r]
            writer.writerow([r, m.get('DOI', ''), m.get('Title', ''), m.get('Journal', ''),
                             s['online'], s['issue'], s['field'], s['split']])

    with Pool() as pool:
        contexts = pool.map(review_context, reviews, chunksize=50)
        source_hashes = dict(pool.map(hash_source, papers, chunksize=200))
    with open(os.path.join(OUT, 'review_context.jsonl'), 'w', encoding='utf-8', newline='\n') as f:
        for c in contexts:
            f.write(json.dumps(c, ensure_ascii=False) + '\n')

    # sources.csv
    with open(os.path.join(V2, 'source_licences.csv'), encoding='utf-8') as f:
        licences = {r['pmcid']: r for r in csv.DictReader(f)}
    with open(os.path.join(V2, 'source_metadata.csv'), encoding='utf-8') as f:
        source_meta = {r['pmcid']: r for r in csv.DictReader(f) if r['pmcid']}
    gemini = {}
    if os.path.exists(os.path.join(V2, 'design_labels.csv')):
        with open(os.path.join(V2, 'design_labels.csv'), encoding='utf-8') as f:
            gemini = {r['pmid']: r['design'] for r in csv.DictReader(f) if r['design'] != 'cannot tell'}
    with open(os.path.join(OUT, 'sources.csv'), 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['source_pmcid', 'licence', 'licence_version', 'pmid', 'doi', 'design',
                         'design_source', 'publisher', 'parent_publisher', 'year', 'sha256'])
        for p in papers:
            lic, m = licences[p], source_meta.get(p, {})
            design, design_source = m.get('design', 'unknown'), m.get('design_source', 'none')
            if design == 'unknown' and m.get('pmid') in gemini:
                design, design_source = gemini[m['pmid']], 'gemini'
            writer.writerow([p, lic['licence'], lic['version'], m.get('pmid', ''), m.get('doi', ''),
                             design, design_source, m.get('publisher', ''),
                             m.get('parent_publisher', ''), m.get('year', ''), source_hashes[p]])

    # table maps for the tables the items come from
    out = write_gz(os.path.join(OUT, 'table_maps.jsonl.gz'))
    with open(os.path.join(V2, 'table_maps.jsonl'), encoding='utf-8') as f:
        n_maps = 0
        for record in map(json.loads, f):
            if (record['review_pmcid'], record['table_no']) in tables:
                out.write(json.dumps(record, ensure_ascii=False) + '\n')
                n_maps += 1
    close_gz(out)

    # Documentation lives in docs/release/ (a freeze rebuilds release/v2/ whole).
    docs = os.path.join(BASE, 'docs', 'release')
    if os.path.isdir(docs):
        for name in sorted(os.listdir(docs)):
            if name.endswith('.md'):
                copy_normalised(os.path.join(docs, name), os.path.join(OUT, name))
    for name in QUALITY_FILES:
        copy_normalised(os.path.join(V2, name), os.path.join(OUT, 'quality', name))
    for name in ('audit.json', 'sample.json'):
        copy_normalised(os.path.join(V2, 'audit', name), os.path.join(OUT, 'quality', f'audit_{name}'))

    files = {}
    for root, _, names in os.walk(OUT):
        for name in sorted(names):
            path = os.path.join(root, name)
            files[os.path.relpath(path, OUT).replace(os.sep, '/')] = {
                'bytes': os.path.getsize(path), 'sha256': sha256(path)}
    scripts = {}
    for root, _, names in os.walk(os.path.join(BASE, 'scripts')):
        if '__pycache__' in root:
            continue
        for name in sorted(names):
            if name.endswith(('.py', '.html')):
                path = os.path.join(root, name)
                scripts[os.path.relpath(path, BASE).replace(os.sep, '/')] = sha256(path)
    manifest = {
        'name': 'SR-QA', 'version': VERSION,
        'frozen': datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
        'licence': 'CC BY 4.0 (data); every source paper and review is CC BY',
        'counts': {'items': len(items), 'reviews': len(reviews), 'source_papers': len(papers),
                   'tables': len(tables), 'table_maps': n_maps,
                   'items_by_split': {s: len(by_split[s]) for s in SPLITS}},
        'files': files,
        'pipeline_scripts_sha256': scripts,
        'source_xml_sha256': source_hashes,
    }
    with open(os.path.join(OUT, 'MANIFEST.json'), 'w', encoding='utf-8', newline='\n') as f:
        json.dump(manifest, f, indent=1)

    print(f"SR-QA v{VERSION} frozen -> {OUT}")
    print(f"  items {len(items):,} | reviews {len(reviews):,} | source papers {len(papers):,} | "
          f"tables {len(tables):,}")
    for s in SPLITS:
        print(f'  {s:<15} {len(by_split[s]):>7,}')
    total = sum(v['bytes'] for v in files.values())
    print(f'  {len(files)} files, {total / 1e6:.1f} MB; {len(scripts)} scripts and '
          f'{len(source_hashes):,} source XMLs checksummed')


if __name__ == '__main__':
    main()
