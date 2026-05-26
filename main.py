import asyncio
import glob
import logging
import os
from enum import Enum

import aiohttp
import pymorphy2
from anyio import create_task_group

from adapters.inosmi_ru import sanitize
from text_tools import calculate_jaundice_rate, split_by_words

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

TEST_ARTICLES = [
    "https://inosmi.ru/20260525/tramp-278594105.html",
    "https://inosmi.ru/20260525/bpla-278594994.html",
    "https://inosmi.ru/20260525/madyar-278593948.html",
    "https://inosmi.ru/20260523/svyaz-278575751.html",
    "https://inosmi.ru/20260525/rossiya-278593248.html",
]


class ProcessingStatus(Enum):
    OK = "OK"
    FETCH_ERROR = "FETCH_ERROR"


def load_charged_words(directory="charged_dict"):
    """Загружает все слова из .txt файлов в папке directory."""

    charged_words = set()

    pattern = os.path.join(directory, "*.txt")

    for filepath in glob.glob(pattern):
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                word = line.strip().lower()
                if word:
                    charged_words.add(word)
    return list(charged_words)


async def fetch(session, url):
    async with session.get(url) as response:
        response.raise_for_status()
        return await response.text()


async def process_article(session, url, charged_words, morph):
    """Возвращает кортеж (url, рейтинг, количество слов)."""

    try:
        html = await fetch(session, url)
        clean_text = sanitize(html, plaintext=True)
        article_words = split_by_words(morph, clean_text)
        rate = calculate_jaundice_rate(article_words, charged_words)
        return url, ProcessingStatus.OK, rate, len(article_words)
    except aiohttp.ClientError as e:
        logger.error(f"Ошибка при обработке {url}: {e}")
        return url, ProcessingStatus.FETCH_ERROR, None, None
    except Exception as e:
        logger.exception(f"Неожиданная ошибка при обработке {url}: {e}")
        return url, ProcessingStatus.FETCH_ERROR, None, None


async def set_result(idx, url, session, charged_words, morph, results):
    res = await process_article(session, url, charged_words, morph)
    results[idx] = res


async def main():
    charged_words = load_charged_words()
    morph = pymorphy2.MorphAnalyzer()

    async with aiohttp.ClientSession() as session:
        results = [None] * len(TEST_ARTICLES)
        async with create_task_group() as tg:
            for idx, url in enumerate(TEST_ARTICLES):
                tg.start_soon(set_result, idx, url, session, charged_words, morph, results)

        print("\nРезультаты анализа:\n")
        for url, status, rate, word_count in results:
            print(f"URL: {url}")
            print(f"Статус: {status.value}")
            if status == ProcessingStatus.OK:
                print(f"Рейтинг: {rate:.2f}")
                print(f"Слов в статье: {word_count}")
            else:
                print("Рейтинг: None")
                print("Слов в статье: None")
            print("-" * 50)


asyncio.run(main())
