from __future__ import annotations

from typing import Any

import requests


def literature_search(
    query: str, limit: int = 3, api_key: str | None = None
) -> dict[str, Any]:
    headers = {"User-Agent": "ChemCrow-Lite/0.1"}
    if api_key:
        headers["x-api-key"] = api_key

    response = requests.get(
        "https://api.semanticscholar.org/graph/v1/paper/search",
        headers=headers,
        params={
            "query": query,
            "limit": limit,
            "fields": "title,year,authors,url,openAccessPdf,abstract",
        },
        timeout=30,
    )

    if response.status_code == 429:
        return {
            "query": query,
            "status": "rate_limited",
            "message": "Semantic Scholar API returned 429. Configure SEMANTIC_SCHOLAR_API_KEY or retry later.",
            "results": [],
        }

    response.raise_for_status()
    payload = response.json()
    results: list[dict[str, Any]] = []
    for item in payload.get("data", []):
        results.append(
            {
                "title": item.get("title"),
                "year": item.get("year"),
                "url": item.get("url"),
                "open_access_pdf": item.get("openAccessPdf"),
                "authors": [author.get("name") for author in item.get("authors", [])],
                "abstract": item.get("abstract"),
            }
        )
    return {
        "query": query,
        "status": "ok",
        "total": payload.get("total"),
        "results": results,
    }
