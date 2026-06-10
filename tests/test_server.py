"""Тесты для server.py."""

from unittest.mock import AsyncMock, Mock, patch

import aiohttp
import pytest

from adapters.exceptions import ArticleNotFound
from server import ProcessingStatus, process_article


@pytest.mark.asyncio
async def test_process_article_ok():
    charged_words = ["слово"]
    morph = AsyncMock()
    mock_response = AsyncMock()
    mock_response.text = AsyncMock(return_value="<html>")
    mock_response.raise_for_status = Mock()
    mock_cm = AsyncMock()
    mock_cm.__aenter__.return_value = mock_response
    mock_cm.__aexit__ = AsyncMock(return_value=None)
    mock_session = Mock()
    mock_session.get = Mock(return_value=mock_cm)

    with patch("server._analyze_text") as mock_analyze:
        mock_analyze.return_value = ("clean", [], 1.23, 100)
        result = await process_article(mock_session, "http://example.com", charged_words, morph)
        assert result["status"] == ProcessingStatus.OK.value
        assert result["score"] == 1.23
        assert result["words_count"] == 100


@pytest.mark.asyncio
async def test_process_article_fetch_error():
    charged_words = []
    morph = None
    mock_session = Mock()
    mock_session.get = Mock(side_effect=aiohttp.ClientError("Network error"))
    result = await process_article(mock_session, "http://example.com", charged_words, morph)
    assert result["status"] == ProcessingStatus.FETCH_ERROR.value


@pytest.mark.asyncio
async def test_process_article_parsing_error():
    charged_words = []
    morph = None
    mock_response = AsyncMock()
    mock_response.text = AsyncMock(return_value="<html>")
    mock_response.raise_for_status = Mock()
    mock_cm = AsyncMock()
    mock_cm.__aenter__.return_value = mock_response
    mock_cm.__aexit__ = AsyncMock(return_value=None)
    mock_session = Mock()
    mock_session.get = Mock(return_value=mock_cm)

    with patch("server._analyze_text") as mock_analyze:
        mock_analyze.side_effect = ArticleNotFound("No article")
        result = await process_article(mock_session, "http://example.com", charged_words, morph)
        assert result["status"] == ProcessingStatus.PARSING_ERROR.value


@pytest.mark.asyncio
async def test_process_article_timeout():
    charged_words = []
    morph = None
    mock_response = AsyncMock()
    mock_response.text = AsyncMock(return_value="<html>")
    mock_response.raise_for_status = Mock()
    mock_cm = AsyncMock()
    mock_cm.__aenter__.return_value = mock_response
    mock_cm.__aexit__ = AsyncMock(return_value=None)
    mock_session = Mock()
    mock_session.get = Mock(return_value=mock_cm)

    with patch("server._analyze_text") as mock_analyze:
        mock_analyze.side_effect = TimeoutError
        result = await process_article(mock_session, "http://example.com", charged_words, morph)
        assert result["status"] == ProcessingStatus.TIMEOUT.value
