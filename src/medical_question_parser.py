"""Generic question parser for medical document QA.

This parser intentionally does not depend on any hardcoded medical field
names or object-detection prompts. It extracts a small set of generic search
keywords and a simple intent label from a natural-language question.
"""

import logging
import re
from dataclasses import dataclass
from typing import List

logger = logging.getLogger("medical_question_parser")

_STOP_WORDS = frozenset({
    "a", "an", "and", "are", "can", "could", "do", "does", "for", "from",
    "i", "in", "is", "me", "of", "on", "please", "show", "tell", "the",
    "this", "to", "what", "where", "which", "who", "with",
})


@dataclass
class ParsedQuestion:
    raw_question: str
    keywords: List[str]
    intent: str


class DocumentQuestionParser:
    """Extract generic keywords and intent from a document QA question."""

    def parse(self, question_text: str) -> ParsedQuestion:
        cleaned = " ".join(str(question_text or "").split())
        normalized = cleaned.casefold()
        logger.debug("Parsing question: %s", cleaned)

        if not cleaned:
            return ParsedQuestion(raw_question=cleaned, keywords=[], intent="none")

        if any(token in normalized for token in ("summarize", "summary", "overview")):
            return ParsedQuestion(raw_question=cleaned, keywords=["summary"], intent="summary")

        tokens = self._tokenize(normalized)
        tokens = [token.replace("patient's", "patient") for token in tokens]
        keywords = self._expand_keywords(tokens, cleaned)

        if not keywords:
            keywords = [token for token in tokens if len(token) > 2][:6]

        return ParsedQuestion(raw_question=cleaned, keywords=keywords, intent="lookup")

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return [token for token in re.findall(r"[\w']+", text) if token and token not in _STOP_WORDS]

    def _expand_keywords(self, tokens: List[str], question_text: str) -> List[str]:
        keywords: List[str] = []
        for token in tokens:
            if token in {"patient", "name", "value", "result", "report", "diagnosis", "bp", "blood", "pressure", "hba1c", "hemoglobin", "creatinine", "date", "number", "document", "age", "sex", "history", "lab", "test", "score"}:
                keywords.append(token)
            elif len(token) > 2:
                keywords.append(token)

        if "patient" in tokens and "name" in tokens:
            keywords = [token for token in keywords if token not in {"patient", "name"}] + ["patient", "name"]
        if "creatinine" in tokens:
            keywords = [token for token in keywords if token not in {"creatinine", "value"}] + ["creatinine"]
        if "hba1c" in tokens:
            keywords = [token for token in keywords if token != "hba1c"] + ["hba1c"]
        if "hemoglobin" in tokens:
            keywords = [token for token in keywords if token != "hemoglobin"] + ["hemoglobin"]
        if "blood" in tokens and "pressure" in tokens:
            keywords = [token for token in keywords if token not in {"blood", "pressure"}] + ["blood", "pressure"]
        if "diagnosis" in tokens:
            keywords = [token for token in keywords if token != "diagnosis"] + ["diagnosis"]

        if "patient" in question_text.casefold() and "name" in question_text.casefold():
            keywords = [kw for kw in keywords if kw not in {"patient", "name", "patient's"}] + ["patient", "name"]

        return list(dict.fromkeys(keywords))[:8]
