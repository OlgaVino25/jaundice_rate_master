"""Инструменты для обработки текста: нормализация, расчёт желтушности."""

import string

MIN_WORD_LENGTH = 3
EXCEPTION_WORDS = {"не"}


def _clean_word(word):
    """Удаляет из слова кавычки-ёлочки, многоточие и пунктуацию."""

    word = word.replace("«", "").replace("»", "").replace("…", "")
    word = word.strip(string.punctuation)
    return word


def split_by_words(morph, text):
    """Учитывает знаки пунктуации, регистр и словоформы, выкидывает предлоги."""
    words = []
    for word in text.split():
        cleaned_word = _clean_word(word)
        if not cleaned_word:
            continue
        normalized_word = morph.parse(cleaned_word)[0].normal_form
        if len(normalized_word) >= MIN_WORD_LENGTH or normalized_word in EXCEPTION_WORDS:
            words.append(normalized_word)
    return words


def calculate_jaundice_rate(article_words, charged_words):
    """Расчитывает желтушность текста, принимает список "заряженных" слов и ищет их внутри article_words."""

    if not article_words:
        return 0.0

    charged_set = set(charged_words)
    found = sum(1 for w in article_words if w in charged_set)
    return round(found / len(article_words) * 100, 2)
