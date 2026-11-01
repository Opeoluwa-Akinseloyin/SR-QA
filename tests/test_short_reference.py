"""Tests for the compact bibliography form, short_reference() in scripts/02_list_tables.py."""
import importlib.util
import os
import xml.etree.ElementTree as ET

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location(
    'list_tables', os.path.join(ROOT, 'scripts', '02_list_tables.py'))
list_tables = importlib.util.module_from_spec(spec)
spec.loader.exec_module(list_tables)


def short(xml):
    ref = ET.fromstring(xml)
    return list_tables.short_reference(ref, 'FULL TEXT ' * 30)


def names(*surnames):
    return ''.join(f'<name><surname>{s}</surname><given-names>A</given-names></name>'
                   for s in surnames)


def test_three_authors_and_plain_text_title():
    # <article-title> has no child elements: an `a or b` fallback would have
    # wrongly skipped it for <source> (an element with no children is falsy).
    xml = (f'<ref><element-citation><person-group>{names("Saleh", "Ibrahem", "Mahmoud")}'
           '</person-group><article-title>Surfactant protein D</article-title>'
           '<source>Pediatr Res</source><year>2022</year></element-citation></ref>')
    assert short(xml) == 'Saleh et al., 2022. Surfactant protein D'


def test_two_authors_are_both_named():
    xml = (f'<ref><mixed-citation><person-group>{names("Gwak", "Jang")}</person-group>'
           '<year>2026</year><article-title>Implantation</article-title></mixed-citation></ref>')
    assert short(xml) == 'Gwak and Jang, 2026. Implantation'


def test_group_author_and_source_when_no_article_title():
    xml = ('<ref><element-citation><person-group><collab>WHO</collab></person-group>'
           '<source>Global report</source><year>2021</year></element-citation></ref>')
    assert short(xml) == 'WHO, 2021. Global report'


def test_long_titles_are_cut():
    xml = (f'<ref><element-citation><person-group>{names("Lee")}</person-group>'
           f'<year>2020</year><article-title>{"x" * 150}</article-title></element-citation></ref>')
    assert short(xml) == 'Lee, 2020. ' + 'x' * 100 + '...'


def test_falls_back_to_text_without_author_or_year():
    xml = '<ref><mixed-citation>Anonymous report, undated.</mixed-citation></ref>'
    assert short(xml) == ('FULL TEXT ' * 30)[:150]
