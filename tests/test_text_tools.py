"""Тесты для text_tools.py."""

import pymorphy2
import pytest

from text_tools import calculate_jaundice_rate, split_by_words


@pytest.fixture(scope="module")
def morph():
    return pymorphy2.MorphAnalyzer()


def test_split_by_words(morph):
    assert split_by_words(morph, "Во-первых, он хочет, чтобы") == [
        "во-первых",
        "хотеть",
        "чтобы",
    ]
    assert split_by_words(morph, "«Удивительно, но это стало началом!»") == [
        "удивительно",
        "это",
        "стать",
        "начало",
    ]


def test_calculate_jaundice_rate():
    assert calculate_jaundice_rate([], []) == 0.0
    rate = calculate_jaundice_rate(["все", "аутсайдер", "побег"], ["аутсайдер", "банкротство"])
    assert 33.0 < rate < 34.0
