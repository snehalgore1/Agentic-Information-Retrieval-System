import asyncio
import logging
from typing import List

import httpx
from src.schemas.embeddings.jina import JinaEmbeddingRequest, JinaEmbeddingResponse

logger = logging.getLogger(__name__)


class JinaEmbeddingsClient:
    """Client for Jina AI embeddings API.

    Uses Jina embeddings v3 model with 1024 dimensions optimized for retrieval.
    Documentation: https://jina.ai/embeddings
    """

    def __init__(self, api_key: str, base_url: str = "https://api.jina.ai/v1"):
        """Initialize Jina embeddings client.

        :param api_key: Jina API key
        :param base_url: API base URL
        """
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        self.client = httpx.AsyncClient(timeout=30.0)
        logger.info("Jina embeddings client initialized")

    async def _post_embeddings(self, request_data: "JinaEmbeddingRequest", max_retries: int = 5) -> JinaEmbeddingResponse:
        """POST to the embeddings API with exponential backoff on rate limits.

        Jina's free tier throttles bursts with HTTP 429/403; retry with backoff so
        indexing and search survive throttling instead of failing.
        """
        delay = 2.0
        for attempt in range(max_retries):
            try:
                response = await self.client.post(
                    f"{self.base_url}/embeddings", headers=self.headers, json=request_data.model_dump()
                )
                response.raise_for_status()
                return JinaEmbeddingResponse(**response.json())
            except httpx.HTTPStatusError as e:
                if e.response.status_code in (429, 403) and attempt < max_retries - 1:
                    logger.warning(
                        f"Jina throttled ({e.response.status_code}); retrying in {delay:.0f}s "
                        f"(attempt {attempt + 1}/{max_retries})"
                    )
                    await asyncio.sleep(delay)
                    delay *= 2
                    continue
                raise
        raise RuntimeError("Unreachable: embeddings retry loop exhausted")

    async def embed_passages(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        """Embed text passages for indexing.

        :param texts: List of text passages to embed
        :param batch_size: Number of texts to process in each API call
        :returns: List of embedding vectors
        """
        embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]

            request_data = JinaEmbeddingRequest(
                model="jina-embeddings-v3", task="retrieval.passage", dimensions=1024, input=batch
            )

            result = await self._post_embeddings(request_data)
            batch_embeddings = [item["embedding"] for item in result.data]
            embeddings.extend(batch_embeddings)
            logger.debug(f"Embedded batch of {len(batch)} passages")

        logger.info(f"Successfully embedded {len(texts)} passages")
        return embeddings

    async def embed_query(self, query: str) -> List[float]:
        """Embed a search query.

        :param query: Query text to embed
        :returns: Embedding vector for the query
        """
        request_data = JinaEmbeddingRequest(model="jina-embeddings-v3", task="retrieval.query", dimensions=1024, input=[query])

        result = await self._post_embeddings(request_data)
        embedding = result.data[0]["embedding"]
        logger.debug(f"Embedded query: '{query[:50]}...'")
        return embedding

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
