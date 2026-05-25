import asyncio
import glob
import os

import aiohttp
import pymorphy2

from adapters.inosmi_ru import sanitize
from text_tools import calculate_jaundice_rate, split_by_words


async def fetch(session, url):
    async with session.get(url) as response:
        response.raise_for_status()
        return await response.text()


def load_charged_words(directory="charged_dict"):
    """Загружает все слова из .txt файлов в папке directory."""

    charged_words = []

    pattern = os.path.join(directory, "*.txt")

    for filepath in glob.glob(pattern):
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                word = line.strip()
                if word:
                    charged_words.append(word.lower())
    return list(set(charged_words))


async def main():
    url = "https://inosmi.ru/20260514/vizit-278427943.html"
    async with aiohttp.ClientSession() as session:
        html = await fetch(session, url)
        clean_text = sanitize(html, plaintext=True)

        morph = pymorphy2.MorphAnalyzer()
        article_words = split_by_words(morph, clean_text)

        charged_words = load_charged_words()

        rate = calculate_jaundice_rate(article_words, charged_words)

        print(f"Рейтинг: {rate:.2f}")
        print(f"Слов в статье: {len(article_words)}")


asyncio.run(main())
