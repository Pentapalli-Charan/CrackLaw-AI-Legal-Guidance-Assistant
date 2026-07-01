import json
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from src.config import Config
from src.retrieval.exceptions import VectorStoreError, DatabaseConnectionError
from src.retrieval.search_result import SearchResultItem

logger = logging.getLogger("CrackLaw.Retrieval.VectorStore")

class VectorStore:
    """Manages legal chunk vector index operations in PostgreSQL using pgvector, with NumPy fallback."""

    def __init__(self, config: Optional[Config] = None, dimension: int = 384):
        self.config = config or Config()
        self.dimension = dimension
        
        ret_settings = self.config.retrieval_settings
        self.db_config = ret_settings.get("database", {})
        
        self.conn = None
        self.use_fallback = False
        
        # In-memory storage mock variables
        self._fallback_db: List[Dict[str, Any]] = []

        # Try connecting to PostgreSQL
        self._connect_db()

    def _connect_db(self) -> None:
        """Establishes database connection or triggers local fallback mode."""
        if not self.db_config:
            logger.warning("No database configuration defined in config.json. Activating local in-memory fallback store.")
            self.use_fallback = True
            return

        try:
            import psycopg2
            from psycopg2.extras import RealDictCursor
            
            self.conn = psycopg2.connect(
                host=self.db_config.get("host", "localhost"),
                port=self.db_config.get("port", 5432),
                database=self.db_config.get("dbname", "cracklaw"),
                user=self.db_config.get("user", "postgres"),
                password=self.db_config.get("password", "password"),
                connect_timeout=3
            )
            self.conn.autocommit = True
            self.cursor_factory = RealDictCursor
            logger.info("Successfully connected to PostgreSQL database.")
            
            # Setup extension and tables
            self.create_schema()
            
        except Exception as e:
            logger.warning(
                "PostgreSQL pgvector connection failed: %s. Activating NumPy local in-memory fallback database.",
                str(e)
            )
            self.use_fallback = True
            self.conn = None

    def create_schema(self) -> None:
        """Executes DDL migrations to register pgvector extension and create tables."""
        if self.use_fallback:
            return
            
        try:
            with self.conn.cursor() as cur:
                # 1. Enable pgvector extension
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                
                # 2. Create chunks table
                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS legal_chunks (
                        chunk_id VARCHAR(100) PRIMARY KEY,
                        document_id VARCHAR(100) NOT NULL,
                        text TEXT NOT NULL,
                        embedding vector({self.dimension}) NOT NULL,
                        metadata JSONB,
                        act VARCHAR(255),
                        chapter VARCHAR(255),
                        section VARCHAR(255),
                        subsection VARCHAR(255),
                        language VARCHAR(50),
                        source VARCHAR(255),
                        version VARCHAR(50),
                        checksum VARCHAR(64)
                    );
                """)
                
                # 3. Create indices
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON legal_chunks(document_id);
                    CREATE INDEX IF NOT EXISTS idx_chunks_metadata ON legal_chunks USING gin (metadata);
                """)
                
                # Create HNSW vector similarity search index
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS legal_chunks_embedding_hnsw_idx 
                    ON legal_chunks USING hnsw (embedding vector_cosine_ops);
                """)
                
            logger.info("PostgreSQL pgvector schema migrations executed successfully.")
        except Exception as e:
            logger.error("Failed to execute PostgreSQL DDL migrations: %s", str(e))
            self.use_fallback = True
            logger.warning("Failing back to NumPy local database fallback due to schema initialization error.")

    def insert_chunks_batch(self, chunks: List[Dict[str, Any]]) -> None:
        """Registers a batch of chunks into the database (or in-memory mock)."""
        if not chunks:
            return

        if self.use_fallback:
            # 1. Fallback implementation
            for chunk in chunks:
                # Validate dimensions
                emb = chunk.get("embedding")
                if not emb or len(emb) != self.dimension:
                    raise VectorStoreError(f"Embedding dimension mismatch: expected {self.dimension}, got {len(emb) if emb else 0}")
                
                # Delete existing duplicate if present
                self._fallback_db = [c for c in self._fallback_db if c["chunk_id"] != chunk["chunk_id"]]
                
                self._fallback_db.append({
                    "chunk_id": chunk["chunk_id"],
                    "document_id": chunk["document_id"],
                    "text": chunk["text"],
                    "embedding": np.array(emb, dtype=np.float32),
                    "metadata": chunk.get("metadata", {}),
                    "act": chunk.get("act"),
                    "chapter": chunk.get("chapter"),
                    "section": chunk.get("section"),
                    "subsection": chunk.get("subsection"),
                    "language": chunk.get("language", "en"),
                    "source": chunk.get("source"),
                    "version": chunk.get("version"),
                    "checksum": chunk.get("checksum")
                })
            logger.info("Indexed %d chunks in local NumPy fallback storage.", len(chunks))
            return

        # 2. PostgreSQL Implementation
        try:
            with self.conn.cursor() as cur:
                for chunk in chunks:
                    emb = chunk.get("embedding")
                    if not emb or len(emb) != self.dimension:
                        raise VectorStoreError(f"Embedding dimension mismatch: expected {self.dimension}")
                        
                    meta_json = json.dumps(chunk.get("metadata", {}))
                    
                    cur.execute(
                        f"""
                        INSERT INTO legal_chunks (
                            chunk_id, document_id, text, embedding, metadata,
                            act, chapter, section, subsection, language, source, version, checksum
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (chunk_id) DO UPDATE SET
                            document_id = EXCLUDED.document_id,
                            text = EXCLUDED.text,
                            embedding = EXCLUDED.embedding,
                            metadata = EXCLUDED.metadata,
                            act = EXCLUDED.act,
                            chapter = EXCLUDED.chapter,
                            section = EXCLUDED.section,
                            subsection = EXCLUDED.subsection,
                            language = EXCLUDED.language,
                            source = EXCLUDED.source,
                            version = EXCLUDED.version,
                            checksum = EXCLUDED.checksum;
                        """,
                        (
                            chunk["chunk_id"],
                            chunk["document_id"],
                            chunk["text"],
                            emb,
                            meta_json,
                            chunk.get("act"),
                            chunk.get("chapter"),
                            chunk.get("section"),
                            chunk.get("subsection"),
                            chunk.get("language", "en"),
                            chunk.get("source"),
                            chunk.get("version"),
                            chunk.get("checksum")
                        )
                    )
            logger.info("Upserted %d chunks into pgvector database.", len(chunks))
        except Exception as e:
            logger.error("Failed to insert chunk batch into PostgreSQL: %s", str(e))
            raise VectorStoreError(f"Database insertion failed: {e}") from e

    def _build_sql_filters(self, filters: Dict[str, Any]) -> Tuple[str, List[Any]]:
        """Utility converting filter dictionary to SQL WHERE fragment."""
        clauses = []
        params = []
        
        for k, v in filters.items():
            if v is None:
                continue
            # Handle standard fields directly
            if k in ["document_id", "act", "chapter", "section", "subsection", "language", "source", "version"]:
                clauses.append(f"{k} = %s")
                params.append(v)
            else:
                # Handle arbitrary JSONB metadata sub-fields
                clauses.append("metadata->>%s = %s")
                params.extend([k, str(v)])
                
        clause_str = " AND ".join(clauses)
        if clause_str:
            clause_str = " AND " + clause_str
        return clause_str, params

    def search_vector(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        min_similarity: float = 0.0,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResultItem]:
        """Runs vector similarity calculations filtering and ranking nearest neighbours."""
        if len(query_embedding) != self.dimension:
            raise VectorStoreError(f"Query embedding dimension mismatch: expected {self.dimension}")

        # 1. Fallback implementation
        if self.use_fallback:
            results = []
            q_vec = np.array(query_embedding, dtype=np.float32)
            q_norm = np.linalg.norm(q_vec)
            
            for item in self._fallback_db:
                # Apply filters
                if filters:
                    match = True
                    for fk, fv in filters.items():
                        if fv is None:
                            continue
                        if fk in item:
                            if item[fk] != fv:
                                match = False
                                break
                        else:
                            if item.get("metadata", {}).get(fk) != fv:
                                match = False
                                break
                    if not match:
                        continue

                # Cosine Similarity Calculation
                i_vec = item["embedding"]
                i_norm = np.linalg.norm(i_vec)
                if q_norm == 0 or i_norm == 0:
                    sim = 0.0
                else:
                    sim = float(np.dot(q_vec, i_vec) / (q_norm * i_norm))

                # Apply threshold
                if sim >= min_similarity:
                    citation = ""
                    if item.get("act"):
                        citation = f"{item['act']}"
                        if item.get("section"):
                            citation += f", Section {item['section']}"
                    
                    results.append(SearchResultItem(
                        chunk_id=item["chunk_id"],
                        document_id=item["document_id"],
                        text=item["text"],
                        metadata=item["metadata"],
                        score=sim,
                        semantic_score=sim,
                        citation=citation
                    ))
            
            # Sort by score descending and take top_k
            results.sort(key=lambda x: x.score, reverse=True)
            return results[:top_k]

        # 2. PostgreSQL pgvector implementation
        try:
            filter_clause, params = self._build_sql_filters(filters or {})
            
            # PostgreSQL query using <=> cosine distance operator. 
            # Cosine Similarity = 1 - Cosine Distance
            query = f"""
                SELECT 
                    chunk_id, document_id, text, metadata,
                    act, section,
                    (1 - (embedding <=> %s::vector)) AS similarity
                FROM legal_chunks
                WHERE (1 - (embedding <=> %s::vector)) >= %s {filter_clause}
                ORDER BY embedding <=> %s::vector ASC
                LIMIT %s;
            """
            
            sql_params = [query_embedding, query_embedding, min_similarity] + params + [query_embedding, top_k]
            
            results = []
            from psycopg2.extras import RealDictCursor
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, sql_params)
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
                        score=float(row["similarity"]),
                        semantic_score=float(row["similarity"]),
                        citation=citation
                    ))
            return results
        except Exception as e:
            logger.error("Vector search query execution failed in database: %s", str(e))
            raise VectorStoreError(f"Vector search failed: {e}") from e

    def delete_document_chunks(self, document_id: str) -> None:
        """Evicts chunks linked to a target document ID (supporting sync overrides)."""
        if self.use_fallback:
            self._fallback_db = [c for c in self._fallback_db if c["document_id"] != document_id]
            logger.info("Evicted chunks for document '%s' from local NumPy fallback database.", document_id)
            return

        try:
            with self.conn.cursor() as cur:
                cur.execute("DELETE FROM legal_chunks WHERE document_id = %s;", (document_id,))
            logger.info("Deleted chunks for document '%s' from PostgreSQL database.", document_id)
        except Exception as e:
            logger.error("Database deletion operation failed for document '%s': %s", document_id, str(e))
            raise VectorStoreError(f"Failed to delete document chunks: {e}") from e

    def close(self) -> None:
        """Closes connections cleanly."""
        if self.conn and not self.use_fallback:
            try:
                self.conn.close()
                logger.info("Database connection closed cleanly.")
            except Exception:
                pass
