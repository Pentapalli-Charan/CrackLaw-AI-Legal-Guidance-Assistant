import logging
from collections import OrderedDict
from typing import Any, Dict, List, Optional

logger = logging.getLogger("CrackLaw.Retrieval.Cache")

class LRUCache:
    """Standard Least-Recently-Used cache leveraging OrderedDict."""

    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache: OrderedDict[Any, Any] = OrderedDict()

    def get(self, key: Any) -> Optional[Any]:
        if key not in self.cache:
            return None
        # Move key to end to mark it as recently used
        self.cache.move_to_end(key)
        return self.cache[key]

    def set(self, key: Any, value: Any) -> None:
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.max_size:
            # Pop the first element (least recently used)
            self.cache.popitem(last=False)

    def clear(self) -> None:
        self.cache.clear()

    def __len__(self) -> int:
        return len(self.cache)


class EmbeddingCache:
    """In-memory cache mapping text string segments to generated dense float vectors."""

    def __init__(self, max_size: int = 5000):
        self._cache = LRUCache(max_size=max_size)

    def get(self, text: str) -> Optional[List[float]]:
        return self._cache.get(text)

    def set(self, text: str, embedding: List[float]) -> None:
        self._cache.set(text, embedding)

    def clear(self) -> None:
        self._cache.clear()


class QueryCache:
    """In-memory cache mapping search parameter fingerprints to unified RetrievalResponses."""

    def __init__(self, max_size: int = 1000):
        self._cache = LRUCache(max_size=max_size)

    def _make_key(self, query: str, filters: Optional[Dict[str, Any]], search_mode: str, top_k: int) -> str:
        # Convert filters dict to stable string key representation
        filter_str = ""
        if filters:
            sorted_filters = sorted(filters.items())
            filter_str = str(sorted_filters)
        return f"{query}||{filter_str}||{search_mode}||{top_k}"

    def get(self, query: str, filters: Optional[Dict[str, Any]], search_mode: str, top_k: int) -> Optional[Any]:
        key = self._make_key(query, filters, search_mode, top_k)
        return self._cache.get(key)

    def set(self, query: str, filters: Optional[Dict[str, Any]], search_mode: str, top_k: int, response: Any) -> None:
        key = self._make_key(query, filters, search_mode, top_k)
        self._cache.set(key, response)

    def clear(self) -> None:
        self._cache.clear()
