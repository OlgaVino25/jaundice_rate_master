import asyncio
import glob
import logging
import os
import time
from enum import Enum
from functools import partial

import aiohttp
import anyio
import async_timeout
import pymorphy2
from aiohttp import web

from adapters.exceptions import ArticleNotFound
from adapters.inosmi_ru import sanitize
from text_tools import calculate_jaundice_rate, split_by_words

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

REQUEST_TIMEOUT = 5
ANALYSIS_TIMEOUT = 3
MAX_URLS = 10


class ProcessingStatus(Enum):
    OK = "OK"
    FETCH_ERROR = "FETCH_ERROR"
    PARSING_ERROR = "PARSING_ERROR"
    TIMEOUT = "TIMEOUT"


def load_charged_words(directory="charged_dict"):
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
    """Скачивает HTML-страницу по URL и возвращает текст."""

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
    """Возвращает словарь с результатами анализа."""

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
            return {
                "url": url,
                "status": ProcessingStatus.TIMEOUT.value,
                "score": None,
                "words_count": None,
                "time": None,
            }

        elapsed_time = time.monotonic() - start_time
        logger.info(f"Анализ {url} завершён за {elapsed_time:.2f} сек.")
        return {
            "url": url,
            "status": ProcessingStatus.OK.value,
            "score": round(rate, 2),
            "words_count": word_count,
            "time": round(elapsed_time, 2),
        }
    except TimeoutError:
        logger.error(f"Таймаут скачивания {url}")
        return {"url": url, "status": ProcessingStatus.TIMEOUT.value, "score": None, "words_count": None, "time": None}
    except aiohttp.ClientError as e:
        logger.error(f"Ошибка HTTP при обработке {url}: {e}")
        return {
            "url": url,
            "status": ProcessingStatus.FETCH_ERROR.value,
            "score": None,
            "words_count": None,
            "time": None,
        }
    except ArticleNotFound as e:
        logger.error(f"Статья не найдена на {url}: {e}")
        return {
            "url": url,
            "status": ProcessingStatus.PARSING_ERROR.value,
            "score": None,
            "words_count": None,
            "time": None,
        }
    except Exception as e:
        logger.exception(f"Неожиданная ошибка при обработке {url}: {e}")
        return {
            "url": url,
            "status": ProcessingStatus.FETCH_ERROR.value,
            "score": None,
            "words_count": None,
            "time": None,
        }


async def handle(request, charged_words, morph):
    urls_param = request.query.get("urls", "")
    if not urls_param:
        urls_list = []
    else:
        urls_list = [url.strip() for url in urls_param.split(",")]

    if len(urls_list) > MAX_URLS:
        return web.json_response({"error": f"too many urls in request, should be {MAX_URLS} or less"}, status=400)

    async with aiohttp.ClientSession() as session:
        tasks = [process_article(session, url, charged_words, morph) for url in urls_list]
        results = await anyio.create_task_group(*tasks)

    return web.json_response(results)


def main():
    charged_words = load_charged_words()
    morph = pymorphy2.MorphAnalyzer()

    app = web.Application()
    handler = partial(handle, charged_words=charged_words, morph=morph)
    app.router.add_get("/", handler)

    logger.info("Сервер запущен на http://127.0.0.1:8080")
    web.run_app(app, host="127.0.0.1", port=8080)


if __name__ == "__main__":
    main()
