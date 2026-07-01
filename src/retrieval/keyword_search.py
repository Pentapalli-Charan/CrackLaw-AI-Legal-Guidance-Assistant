import re
import logging
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi
from src.config import Config
from src.retrieval.exceptions import SearchError
from src.retrieval.vector_store import VectorStore
from src.retrieval.search_result import SearchResultItem

logger = logging.getLogger("CrackLaw.Retrieval.Keyword")

class KeywordSearch:
    """Performs scalable keyword search using BM25Okapi locally, and Full-Text-Search in PostgreSQL."""

    def __init__(self, config: Optional[Config] = None, vector_store: Optional[VectorStore] = None):
        self.config = config or Config()
        self.vector_store = vector_store or VectorStore(self.config)

    def _tokenize(self, text: str) -> List[str]:
        """Simple English tokenizer: lowercases and extracts alphanumeric words."""
        return re.findall(r"\b\w+\b", text.lower())

    def search_local_bm25(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResultItem]:
        """Runs the BM25Okapi scoring algorithm locally over a list of chunk dictionaries."""
        # 1. Apply metadata filters first
        filtered_chunks = []
        for c in chunks:
            if filters:
                match = True
                for fk, fv in filters.items():
                    if fv is None:
                        continue
                    if fk in c:
                        if c[fk] != fv:
                            match = False
                            break
                    else:
                        if c.get("metadata", {}).get(fk) != fv:
                            match = False
                            break
                if not match:
                    continue
            filtered_chunks.append(c)

        if not filtered_chunks:
            return []

        # 2. Tokenize corpus
        corpus = [self._tokenize(c["text"]) for c in filtered_chunks]
        
        # Initialize BM25
        try:
            bm25 = BM25Okapi(corpus)
        except Exception as e:
            logger.error("Failed to initialize BM25Okapi model: %s", str(e))
            return []

        # 3. Score documents
        tokenized_query = self._tokenize(query)
        scores = bm25.get_scores(tokenized_query)

        # 4. Map back to SearchResultItems
        results = []
        for idx, score in enumerate(scores):
            # Skip non-matching documents (zero score)
            if score <= 0.0:
                continue

            item = filtered_chunks[idx]
            citation = ""
            if item.get("act"):
                citation = f"{item['act']}"
                if item.get("section"):
                    citation += f", Section {item['section']}"

            results.append(SearchResultItem(
                chunk_id=item["chunk_id"],
                document_id=item["document_id"],
                text=item["text"],
                metadata=item.get("metadata", {}),
                score=float(score),
                keyword_score=float(score),
                citation=citation
            ))

        # Sort by score descending
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]

    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResultItem]:
        """Executes keyword query search, selecting local BM25 or database FTS accordingly."""
        if not query:
            return []

        # Case 1: In-memory local fallback
        if self.vector_store.use_fallback:
            logger.info("Executing local BM25 search over in-memory index.")
            return self.search_local_bm25(query, self.vector_store._fallback_db, top_k, filters)

        # Case 2: PostgreSQL Full Text Search
        try:
            filter_clause, params = self.vector_store._build_sql_filters(filters or {})
            
            # Use ts_rank_cd for density-weighted ranking
            sql_query = f"""
                SELECT 
                    chunk_id, document_id, text, metadata,
                    act, section,
                    ts_rank_cd(to_tsvector('english', text), query) AS ts_score
                FROM legal_chunks, plainto_tsquery('english', %s) query
                WHERE to_tsvector('english', text) @@ query {filter_clause}
                ORDER BY ts_score DESC
                LIMIT %s;
            """
            
            sql_params = [query] + params + [top_k]
            
            results = []
            from psycopg2.extras import RealDictCursor
            with self.vector_store.conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(sql_query, sql_params)
                rows = cur.fetchall()
                
                for row in rows:
                    citation = ""
                    if row.get("act"):
                        citation = f"{row['act']}"
                        if row.get("section"):
                            citation += f", Section {row['section']}"

                    results.append(SearchResultItem(
                        chunk_id=row["chunk_id"],
                        document_id=row["document_id"],
                        text=row["text"],
                        metadata=row["metadata"] or {},
                        # Convert FTS score to unified float search score
                        score=float(row["ts_score"]),
                        keyword_score=float(row["ts_score"]),
                        citation=citation
                    ))
            return results
            
        except Exception as e:
            logger.error("PostgreSQL Full-Text Search failed: %s. Falling back to local BM25.", str(e))
            # Fallback to loading database chunks in memory and running local BM25
            try:
                all_chunks = []
                with self.vector_store.conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute("SELECT chunk_id, document_id, text, metadata, act, section FROM legal_chunks;")
                    all_chunks = cur.fetchall()
                return self.search_local_bm25(query, all_chunks, top_k, filters)
            except Exception as e2:
                logger.error("Failed to run local BM25 fallback on database records: %s", str(e2))
                raise SearchError(f"Keyword search failed: {e}") from e
