import argparse
import asyncio
import glob
import logging
import os
import time
from enum import Enum

import aiohttp
import pymorphy2
from anyio import create_task_group

from adapters.exceptions import ArticleNotFound
from adapters.inosmi_ru import sanitize
from text_tools import calculate_jaundice_rate, split_by_words

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class ProcessingStatus(Enum):
    OK = "OK"
    FETCH_ERROR = "FETCH_ERROR"
    PARSING_ERROR = "PARSING_ERROR"
    TIMEOUT = "TIMEOUT"


def parse_args() -> argparse.Namespace:
    """Разбирает аргументы командной строки."""
    parser = argparse.ArgumentParser(description="Анализ статей Inosmi.ru на желтушность")
    parser.add_argument("urls", nargs="+", help="Список URL статей для анализа")
    parser.add_argument(
        "--fetch-timeout",
        type=int,
        default=5,
        help="Таймаут скачивания страницы (сек)",
    )
    parser.add_argument(
        "--parse-timeout",
        type=int,
        default=3,
        help="Таймаут анализа текста (сек)",
    )
    parser.add_argument(
        "--retry-attempts",
        type=int,
        default=3,
        help="Количество повторных попыток при сетевых ошибках",
    )
    parser.add_argument(
        "--retry-delay",
        type=float,
        default=1.0,
        help="Начальная задержка между попытками (сек)",
    )
    return parser.parse_args()


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


async def fetch_with_retry(session, url, timeout, retries, delay):
    """Скачивает страницу с повторными попытками.

    Args:
        session: Сессия aiohttp.
        url: URL страницы.
        timeout: Таймаут на один запрос.
        retries: Максимум попыток.
        delay: Начальная задержка (увеличивается с каждой попыткой).

    Returns:
        Текст HTML.

    Raises:
        aiohttp.ClientError: Если после всех попыток запрос не удался.
        asyncio.TimeoutError: Если время ожидания истекло.
    """

    for attempt in range(retries):
        try:
            async with asyncio.timeout(timeout):
                async with session.get(url) as response:
                    response.raise_for_status()
                    return await response.text()
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            if attempt == retries - 1:
                raise
            wait = delay * (attempt + 1)
            logger.warning(f"Повтор {attempt + 1} для {url} через {wait:.1f} сек: {e}")
            await asyncio.sleep(wait)
    raise RuntimeError("Unreachable")


def _analyze_text(html, charged_words, morph):
    """Синхронная обработка текста (запускается в отдельном потоке)."""

    clean_text = sanitize(html, plaintext=True)
    article_words = split_by_words(morph, clean_text)
    rate = calculate_jaundice_rate(article_words, charged_words)
    return clean_text, article_words, rate, len(article_words)


async def process_article(session, url, charged_words, morph, fetch_timeout, parse_timeout, retries, retry_delay):
    """Возвращает кортеж (url, статус, рейтинг, количество слов, время_сек)."""

    start_time = time.monotonic()
    try:
        html = await fetch_with_retry(session, url, fetch_timeout, retries, retry_delay)

        try:
            _, _, rate, word_count = await asyncio.wait_for(
                asyncio.to_thread(_analyze_text, html, charged_words, morph), timeout=parse_timeout
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


async def main(urls, fetch_timeout, parse_timeout, retry_attempts, retry_delay):
    charged_words = load_charged_words()
    morph = pymorphy2.MorphAnalyzer()

    async with aiohttp.ClientSession() as session:
        results = [None] * len(urls)

        async with create_task_group() as tg:
            for idx, url in enumerate(urls):

                async def worker(i=idx, u=url):
                    results[i] = await process_article(
                        session,
                        u,
                        charged_words,
                        morph,
                        fetch_timeout,
                        parse_timeout,
                        retry_attempts,
                        retry_delay,
                    )

                tg.start_soon(worker)

        logger.info("\nРезультаты анализа:\n")
        for item in results:
            if item is None:
                continue
            url, status, rate, word_count, elapsed_time = item
            logger.info(f"URL: {url}")
            logger.info(f"Статус: {status.value}")
            if status == ProcessingStatus.OK:
                logger.info(f"Рейтинг: {rate:.2f}")
                logger.info(f"Слов в статье: {word_count}")
                logger.info(f"Время анализа: {elapsed_time:.2f} сек.")
            else:
                logger.info("Рейтинг: None")
                logger.info("Слов в статье: None")
            logger.info("-" * 50)


if __name__ == "__main__":
    args = parse_args()
    try:
        asyncio.run(
            main(
                urls=args.urls,
                fetch_timeout=args.fetch_timeout,
                parse_timeout=args.parse_timeout,
                retry_attempts=args.retry_attempts,
                retry_delay=args.retry_delay,
            )
        )
    except KeyboardInterrupt:
        logger.info("\nВыход по прерыванию")
