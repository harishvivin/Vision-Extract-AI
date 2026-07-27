"""
Block Matcher Engine.
Ranks text blocks using TF-IDF vector similarity & keyword token overlaps,
and extracts target answer values from matched document blocks.
"""

import logging
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any, Set
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from src.document_index import DocumentIndex, TextBlock, STOP_WORDS

logger = logging.getLogger("block_matcher")


@dataclass
class MatchResult:
    block: TextBlock
    answer_value: str
    confidence: float
    score: float


class BlockMatcher:
    """Matches natural language queries against indexed document text blocks."""

    def __init__(self, index: DocumentIndex):
        self.index = index

    def match(self, question: str, keywords: List[str]) -> Optional[MatchResult]:
        """
        Search document index for the best block matching the query and keywords.

        Args:
            question (str): Natural language question.
            keywords (List[str]): Extracted target keywords.

        Returns:
            Optional[MatchResult]: Matched text block, extracted answer string, and confidence score.
        """
        if not self.index.blocks or not self.index.vectorizer or self.index.embeddings is None:
            return None

        normalized_q = " ".join(question.casefold().split())
        query_tokens = set(re.findall(r"[\w']+", normalized_q)) - STOP_WORDS
        target_keywords = set(keywords) | query_tokens

        # Search query string construction
        search_query = " ".join(keywords) if keywords else " ".join(query_tokens)
        if not search_query.strip():
            return None

        # Compute TF-IDF Cosine Similarity
        query_vec = self.index.vectorizer.transform([search_query])
        scores = cosine_similarity(query_vec, self.index.embeddings)[0]

        # Calculate adjusted scoring for domain disambiguation
        is_pure_hb_query = ("hemoglobin" in target_keywords or "hb" in query_tokens) and not ("hba1c" in target_keywords or "a1c" in query_tokens or "glycated" in query_tokens)
        is_hba1c_query = "hba1c" in target_keywords or "a1c" in query_tokens or "glycated" in query_tokens
        is_hospital_query = "hospital" in target_keywords or "clinic" in query_tokens or "facility" in query_tokens or "lab" in query_tokens
        is_patient_query = "patient" in target_keywords or ("name" in target_keywords and not is_hospital_query)

        adjusted_scores = []
        for idx, block in enumerate(self.index.blocks):
            base_score = float(scores[idx])
            block_norm = block.normalized_text

            # Specific disambiguation rules
            if is_pure_hb_query and ("hba1c" in block_norm or "glycated" in block_norm or "a1c" in block_norm):
                base_score *= 0.2  # Penalize HbA1c blocks for plain hemoglobin queries
            elif is_hba1c_query and ("hba1c" in block_norm or "glycated" in block_norm or "a1c" in block_norm):
                base_score *= 1.5  # Boost HbA1c blocks for HbA1c queries

            if is_hospital_query:
                if any(h_word in block_norm for h_word in ["hospital", "clinic", "diagnostics", "center", "centre", "institute", "laboratory", "lab"]):
                    base_score += 0.5
                if "patient" in block_norm:
                    base_score *= 0.3

            if is_patient_query:
                if "patient" in block_norm or "name:" in block_norm or "name of patient" in block_norm:
                    base_score += 0.4
                if "hospital" in block_norm and "patient" not in block_norm:
                    base_score *= 0.3

            # Boost if block text starts with one of the target keywords
            for kw in target_keywords:
                if block_norm.startswith(kw) or f"\n{kw}" in block_norm or f"{kw}:" in block_norm:
                    base_score += 0.3
                    break

            adjusted_scores.append(base_score)

        ranked_indices = sorted(range(len(adjusted_scores)), key=lambda idx: adjusted_scores[idx], reverse=True)

        best_match: Optional[MatchResult] = None
        highest_combined_score = -1.0

        for idx in ranked_indices[:15]:
            score = float(adjusted_scores[idx])
            block = self.index.blocks[idx]
            block_tokens = set(block.tokens)

            overlap_count = len(block_tokens.intersection(target_keywords))
            kw_match_ratio = overlap_count / max(1, len(target_keywords))

            combined_score = (0.5 * score) + (0.5 * kw_match_ratio)

            is_relevant = False
            if overlap_count >= 1 and (score >= 0.08 or kw_match_ratio >= 0.25):
                is_relevant = True
            elif score >= 0.25:
                is_relevant = True

            if is_relevant:
                extracted_val = self.extract_answer_value(block, normalized_q, target_keywords)
                if extracted_val:
                    if combined_score > highest_combined_score:
                        confidence = round(min(0.99, max(0.40, 0.30 + combined_score * 0.70)), 2)
                        highest_combined_score = combined_score
                        best_match = MatchResult(
                            block=block,
                            answer_value=extracted_val,
                            confidence=confidence,
                            score=score
                        )

        if best_match:
            logger.info(
                f"Matched Block [{best_match.block.block_id}] Page {best_match.block.page_number} "
                f"(Score: {best_match.score:.4f}, Confidence: {best_match.confidence}): Answer='{best_match.answer_value}'"
            )
        else:
            logger.info(f"No sufficiently relevant text block found for query: '{question}'")

        return best_match

    def extract_answer_value(self, block: TextBlock, normalized_q: str, target_keywords: Set[str]) -> Optional[str]:
        """Extract the exact value / text segment matching the question from a block."""
        lines = [l.strip() for l in block.text.splitlines() if l.strip()]
        if not lines:
            return None

        # Strategy 1: Explicit colon / tab / equals key-value pair e.g. "Patient Name: John Doe", "HbA1c: 5.7%"
        for line in lines:
            if ":" in line or "\t" in line or "=" in line or "..." in line:
                pair = self._split_label_value_line(line)
                if pair:
                    label, val = pair
                    label_tokens = set(re.findall(r"[\w']+", label.casefold())) - STOP_WORDS
                    if label_tokens.intersection(target_keywords):
                        clean_val = self._clean_value(val)
                        if clean_val:
                            return clean_val

        # Strategy 2: Check for direct line match where line contains target keyword
        for line in lines:
            line_tokens = set(re.findall(r"[\w']+", line.casefold())) - STOP_WORDS
            if line_tokens.intersection(target_keywords):
                pair = self._split_label_value_line(line)
                if pair and ":" in line:
                    val = self._clean_value(pair[1])
                    if val:
                        return val
                cleaned_line = self._clean_value(line)
                if cleaned_line:
                    return cleaned_line

        # Strategy 3: Multi-line match e.g. Line N = "Patient Name", Line N+1 = "John Doe"
        for i in range(len(lines) - 1):
            curr_line_tokens = set(re.findall(r"[\w']+", lines[i].casefold())) - STOP_WORDS
            if curr_line_tokens.intersection(target_keywords):
                next_val = self._clean_value(lines[i + 1])
                if next_val and not self._is_header_or_label(next_val):
                    return next_val

        # Fallback Strategy 4: Return cleanest full block text
        full_clean = self._clean_value(block.text)
        return full_clean if full_clean else None

    @staticmethod
    def _split_label_value_line(line: str) -> Optional[Tuple[str, str]]:
        match = re.match(r"^\s*(.+?)\s*(?:[:\t\-–=]|\.\.\.+|\s{2,})\s*(.+?)\s*$", line)
        if not match:
            return None
        label, value = match.groups()
        label, value = label.strip(), value.strip()
        if not label or not value or len(label) > 100:
            return None
        return label, value

    @staticmethod
    def _clean_value(val: str) -> str:
        cleaned = " ".join(val.replace("\u00a0", " ").split()).strip(" :-–—=")
        return cleaned

    @staticmethod
    def _is_header_or_label(text: str) -> bool:
        return text.endswith(":") or text.casefold() in {"test", "result", "flag", "units", "reference range"}
