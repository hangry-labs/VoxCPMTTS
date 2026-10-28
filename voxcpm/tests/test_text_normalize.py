# Modified by Hangry Labs in 2026 for the Docker-first inference fork.

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
TEXT_NORMALIZE_PATH = ROOT / "voxcpm" / "utils" / "text_normalize.py"


@pytest.fixture
def text_normalize(monkeypatch):
    monkeypatch.setitem(sys.modules, "regex", types.ModuleType("regex"))

    inflect_stub = types.ModuleType("inflect")
    inflect_stub.engine = lambda: None
    monkeypatch.setitem(sys.modules, "inflect", inflect_stub)

    wetext_stub = types.ModuleType("wetext")
    wetext_stub.Normalizer = object
    monkeypatch.setitem(sys.modules, "wetext", wetext_stub)

    spec = importlib.util.spec_from_file_location("voxcpm.utils.text_normalize_test", TEXT_NORMALIZE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("", ""),
        (" hello", "hello"),
        ("hello ", "hello"),
        (" hello ", "hello"),
        ("hello world", "hello world"),
        ("你 好", "你好"),
        ("你 hello 世界", "你hello世界"),
    ],
)
def test_replace_blank_handles_boundaries_and_preserves_interior_rules(text_normalize, source, expected):
    assert text_normalize.replace_blank(source) == expected


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("if a < b and c > d", "if a < b and c > d"),
        ("if a<b and c>d", "if a<b and c>d"),
        ("if a <b and c> d", "if a <b and c> d"),
        ("1<2>0", "1<2>0"),
        ("We need x<value and y>zero.", "We need x<value and y>zero."),
        ("Use <name> for the speaker.", "Use <name> for the speaker."),
        ("The range is <low to high> inclusive.", "The range is <low to high> inclusive."),
        ("a<b>c", "a<b>c"),
        ("+ 3", "+ 3"),
        ("- 3", "- 3"),
        ("+ x", "+ x"),
        ("+ value", "+ value"),
        ("* variable", "* variable"),
        ("2\n* 3", "2\n* 3"),
        ("2 * 3", "2 * 3"),
        ("2 ** 3 ** 4", "2 ** 3 ** 4"),
        ("2**3**4", "2**3**4"),
        ("a__b__c", "a__b__c"),
        ("Say **hello** and __goodbye__.", "Say hello and goodbye."),
        ("Say **a**.", "Say a."),
        ("* first\n+ second\n- third", "first\nsecond\nthird"),
        ("* **first**\n- __second__", "first\nsecond"),
        ("Read [the guide](https://example.com).", "Read the guide."),
        ("Hello <b>world</b>.", "Hello world."),
        ("hel<strong>lo</strong>", "hello"),
        ('Hello <span\nclass="greeting">world</span>.', "Hello world."),
        ('Hello <span title="a > b">world</span>.', "Hello world."),
        ("<p>one</p><p>two</p>", "one  two"),
        ("hello<br>world", "hello world"),
        ("hello<!-- editor note -->world", "hello world"),
        ("<!DOCTYPE html><p>Hello.</p>", "Hello."),
        ("one &lt; two &gt; zero", "one < two > zero"),
        ("&lt;b&gt;hello&lt;/b&gt;", "<b>hello</b>"),
        ("one&nbsp;two", "one\xa0two"),
        ("Hello <custom>world</custom>.", "Hello <custom>world</custom>."),
        ("Use <![unknown]> literally.", "Use <![unknown]> literally."),
    ],
)
def test_markdown_cleanup_preserves_speech_content(text_normalize, source, expected):
    assert text_normalize.clean_markdown(source) == expected
