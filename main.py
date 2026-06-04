import asyncio
import glob
import logging
import os
import time
from enum import Enum

import aiohttp
import async_timeout
import pymorphy2
from anyio import create_task_group

from adapters.exceptions import ArticleNotFound
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

REQUEST_TIMEOUT = 5
ANALYSIS_TIMEOUT = 3


class ProcessingStatus(Enum):
    OK = "OK"
    FETCH_ERROR = "FETCH_ERROR"
    PARSING_ERROR = "PARSING_ERROR"
    TIMEOUT = "TIMEOUT"


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


def _analyze_text(html, charged_words, morph):
    """Синхронная обработка текста (запускается в отдельном потоке)."""

    clean_text = sanitize(html, plaintext=True)
    article_words = split_by_words(morph, clean_text)
    rate = calculate_jaundice_rate(article_words, charged_words)
    return clean_text, article_words, rate, len(article_words)


async def process_article(session, url, charged_words, morph):
    """Возвращает кортеж (url, статус, рейтинг, количество слов, время_сек)."""

    start_time = time.monotonic()
    try:
        async with async_timeout.timeout(REQUEST_TIMEOUT):
            html = await fetch(session, url)

        try:
            _, _, rate, word_count = await asyncio.wait_for(
                asyncio.to_thread(_analyze_text, html, charged_words, morph), timeout=ANALYSIS_TIMEOUT
            )
        except TimeoutError:
            logger.error(f"Таймаут анализа статьи {url}")
            return url, ProcessingStatus.TIMEOUT, None, None, None

        elapsed_time = time.monotonic() - start_time
        logger.info(f"Анализ {url} завершён за {elapsed_time:.2f} сек.")
        return url, ProcessingStatus.OK, rate, word_count, elapsed_time

    except TimeoutError:
        logger.error(f"Таймаут при скачивании {url}")
        return url, ProcessingStatus.TIMEOUT, None, None, None
    except aiohttp.ClientError as e:
        logger.error(f"Ошибка HTTP при обработке {url}: {e}")
        return url, ProcessingStatus.FETCH_ERROR, None, None, None
    except ArticleNotFound as e:
        logger.error(f"Статья не найдена на {url}: {e}")
        return url, ProcessingStatus.PARSING_ERROR, None, None, None
    except Exception as e:
        logger.exception(f"Неожиданная ошибка при обработке {url}: {e}")
        return url, ProcessingStatus.FETCH_ERROR, None, None, None


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
        for item in results:
            if item is None:
                continue
            url, status, rate, word_count, elapsed_time = item
            print(f"URL: {url}")
            print(f"Статус: {status.value}")
            if status == ProcessingStatus.OK:
                print(f"Рейтинг: {rate:.2f}")
                print(f"Слов в статье: {word_count}")
                print(f"Время анализа: {elapsed_time:.2f} сек.")
            else:
                print("Рейтинг: None")
                print("Слов в статье: None")
            print("-" * 50)


asyncio.run(main())
