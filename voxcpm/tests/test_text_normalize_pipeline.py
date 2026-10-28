# Modified by Hangry Labs in 2026 for the Docker-first inference fork.

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def normalizer_module():
    path = Path(__file__).resolve().parents[1] / "utils" / "text_normalize.py"
    spec = importlib.util.spec_from_file_location("voxcpm_text_normalize_pipeline_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def normalizer(normalizer_module):
    return normalizer_module.TextNormalizer()


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("if a < b and c > d", "if a < b and c > d"),
        ("if a<b and c>d", "if a<b and c>d"),
        ("Keep 1 < 2 > 0.", "Keep one < two > zero."),
        ("Please say **hello** and __goodbye__.", "Please say hello and goodbye."),
        ("* first\n+ second", "first second"),
        ("+ 3", "plus three"),
        ("- 3", "- three"),
        ("-3", "negative three"),
        ("+ value", "plus value"),
        ("2\n* 3", "two times three"),
        ("2 ** 3 ** 4", "two ** three ** four"),
        ('Hello <span\nclass="greeting">world</span>.', "Hello world."),
        ('Hello <span title="a > b">world</span>.', "Hello world."),
        ("Hello <!-- editor note --> world.", "Hello world."),
        ("<p>one</p><p>two</p>", "one two"),
        ("hello<br>world", "hello world"),
        ("one&nbsp;two", "one two"),
        ("Keep 1 &lt; 2 &gt; 0.", "Keep one < two > zero."),
        ("Say `code` now.", "Say now."),
        ("This is a perfectly ordinary sentence.", "This is a perfectly ordinary sentence."),
        ("Go to section3 on the 21st day.", "Go to section three on the twenty first day."),
        ("三 > 二", "三大于二"),
        ("数值 1 < 2 > 0", "数值一小于二大于零"),
        ("请说 **你好**", "请说你好"),
        ("", ""),
        ("<!-- editor note -->", ""),
        ("```code```", ""),
    ],
)
def test_real_normalizer_preserves_meaning(normalizer, source, expected):
    assert normalizer.normalize(source) == expected


def test_clean_text_collapses_whitespace_created_by_removal(normalizer_module):
    assert normalizer_module.clean_text("  Hello\t<!-- note -->\nworld.  ") == "Hello world."
