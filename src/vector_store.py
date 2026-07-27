"""
Modern Vector Database Index for Medical Document RAG.
Generates dense vector embeddings for document chunks and performs Top-K cosine similarity retrieval.
"""

import logging
import re
import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger("vector_store")


@dataclass
class DocumentChunk:
    """Represents an intelligently segmented semantic document chunk."""
    chunk_id: str
    page_number: int
    text: str
    normalized_text: str
    bbox: List[float]  # Bounding box [x1, y1, x2, y2]
    raw_bbox: List[float]
    lines_data: List[Dict[str, Any]]
    section_heading: str = ""


class VectorDatabase:
    """In-memory Vector Database for storing document chunk embeddings and performing fast Top-K retrieval."""

    def __init__(self):
        self.chunks: List[DocumentChunk] = []
        self.embeddings: Optional[np.ndarray] = None
        self.encoder_model: Optional[Any] = None
        self._init_encoder()

    def _init_encoder(self):
        """Initialize SentenceTransformer model if available."""
        try:
            from sentence_transformers import SentenceTransformer
            # Load fast, lightweight, high-accuracy embedding model
            self.encoder_model = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("Initialized SentenceTransformer ('all-MiniLM-L6-v2') for vector embeddings.")
        except Exception as e:
            logger.info(f"SentenceTransformer not available or failed to load ({e}). Using dense TF-IDF/SVD vector encoder fallback.")
            self.encoder_model = None

    def add_chunks(self, chunks: List[DocumentChunk]) -> None:
        """
        Add document chunks to the database and generate dense vector embeddings.

        Args:
            chunks (List[DocumentChunk]): List of semantic document chunks.
        """
        self.chunks = chunks
        if not self.chunks:
            self.embeddings = None
            return

        texts = [c.text for c in self.chunks]

        if self.encoder_model is not None:
            try:
                embeddings_list = self.encoder_model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
                self.embeddings = np.array(embeddings_list, dtype=np.float32)
                logger.info(f"Generated SentenceTransformer embeddings for {len(self.chunks)} chunks. Shape: {self.embeddings.shape}")
                return
            except Exception as e:
                logger.warning(f"Error encoding with SentenceTransformer: {e}. Falling back to dense TF-IDF vectorizer.")

        # Fallback Dense Embedding using Sublinear TF-IDF + L2 Normalization
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.tfidf_vectorizer = TfidfVectorizer(
            ngram_range=(1, 3),
            sublinear_tf=True,
            token_pattern=r"(?u)\b\w+\b"
        )
        sparse_vecs = self.tfidf_vectorizer.fit_transform([c.normalized_text for c in self.chunks])
        dense_vecs = sparse_vecs.toarray()
        # L2 normalize dense vectors
        norms = np.linalg.norm(dense_vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.embeddings = (dense_vecs / norms).astype(np.float32)
        logger.info(f"Generated dense TF-IDF vector embeddings for {len(self.chunks)} chunks. Shape: {self.embeddings.shape}")

    def query(self, query_text: str, top_k: int = 5) -> List[DocumentChunk]:
        """
        Retrieve Top K most relevant chunks using dense vector cosine similarity.

        Args:
            query_text (str): Natural language user question.
            top_k (int): Number of top chunks to retrieve (default 5).

        Returns:
            List[DocumentChunk]: Top K relevant document chunks.
        """
        if not self.chunks or self.embeddings is None:
            return []

        if len(self.chunks) <= top_k:
            return list(self.chunks)

        if self.encoder_model is not None:
            try:
                q_emb = self.encoder_model.encode([query_text], convert_to_numpy=True, normalize_embeddings=True)[0]
                scores = np.dot(self.embeddings, q_emb)
                ranked_indices = np.argsort(scores)[::-1]
                top_indices = ranked_indices[:top_k]
                return [self.chunks[i] for i in top_indices]
            except Exception as e:
                logger.warning(f"Error querying with SentenceTransformer: {e}")

        # Fallback Dense TF-IDF Cosine Search
        if hasattr(self, "tfidf_vectorizer") and self.tfidf_vectorizer is not None:
            try:
                norm_q = " ".join(query_text.casefold().split())
                q_vec = self.tfidf_vectorizer.transform([norm_q]).toarray()[0]
                q_norm = np.linalg.norm(q_vec)
                if q_norm > 0:
                    q_vec = q_vec / q_norm
                scores = np.dot(self.embeddings, q_vec)
                ranked_indices = np.argsort(scores)[::-1]
                top_indices = ranked_indices[:top_k]
                return [self.chunks[i] for i in top_indices]
            except Exception as e:
                logger.warning(f"Error querying dense TF-IDF vector database: {e}")

        return self.chunks[:top_k]

    def clear(self):
        """Clear all stored chunks and embeddings."""
        self.chunks.clear()
        self.embeddings = None
