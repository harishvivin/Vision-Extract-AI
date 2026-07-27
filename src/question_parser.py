"""
Question Parser Engine.
Extracts search keywords, query intent (summary vs lookup), and question metadata
without object detection or hardcoded image prompts.
"""

import logging
import re
from dataclasses import dataclass
from typing import List, Dict, Any

logger = logging.getLogger("question_parser")

# Generic stop words to filter out of search queries
STOP_WORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "could", "do", "does", "for",
    "from", "give", "has", "have", "i", "in", "is", "it", "me", "of", "on", "or", "please",
    "provide", "show", "tell", "the", "this", "to", "value", "what", "where", "which",
    "who", "with", "you", "your"
})

# Known medical term variations / synonyms mapping
MEDICAL_EXPANSIONS = {
    "patient's": ["patient"],
    "patient": ["patient"],
    "name": ["name"],
    "haemoglobin": ["hemoglobin", "hb"],
    "hb": ["hemoglobin", "hb"],
    "hba1c": ["hba1c", "a1c", "glycated hemoglobin"],
    "bp": ["blood pressure", "bp"],
    "creatinine": ["creatinine"],
    "hospital": ["hospital", "clinic", "facility", "diagnostics", "center", "centre", "institute", "laboratory", "lab"],
    "clinic": ["hospital", "clinic"],
    "facility": ["hospital", "facility"],
    "doctor": ["physician", "doctor", "dr"],
    "physician": ["physician", "doctor", "dr"],
    "diagnosis": ["diagnosis", "impression", "assessment"],
    "impression": ["diagnosis", "impression", "assessment"],
    "assessment": ["diagnosis", "impression", "assessment"],
    "sex": ["gender", "sex"],
    "gender": ["gender", "sex"],
    "age": ["age"],
    "hiv": ["hiv", "hiv 1", "hiv 2", "hiv status"],
    "ecg": ["ecg", "ekg", "electrocardiogram"],
    "ekg": ["ecg", "ekg", "electrocardiogram"],
    "electrocardiogram": ["ecg", "ekg", "electrocardiogram"],
}



@dataclass
class ParsedQuestion:
    """Structured question metadata container."""
    raw_question: str
    keywords: List[str]
    intent: str  # "summary" or "lookup"
    is_summary_request: bool
    is_lookup_request: bool


class QuestionParser:
    """Parses natural language questions into keywords and intent flags."""

    def parse(self, question_text: str) -> ParsedQuestion:
        """
        Parse raw question text into structured metadata.

        Args:
            question_text (str): Input query text.

        Returns:
            ParsedQuestion: Structured metadata with keywords and intent.
        """
        cleaned = " ".join(str(question_text or "").split())
        normalized = cleaned.casefold()

        if not cleaned:
            return ParsedQuestion(
                raw_question="",
                keywords=[],
                intent="lookup",
                is_summary_request=False,
                is_lookup_request=True
            )

        # Check for Summary Intent
        summary_tokens = {"summarize", "summary", "overview", "abstract", "recap", "brief"}
        if any(token in normalized for token in summary_tokens):
            logger.info(f"Question parser detected SUMMARY intent for: '{cleaned}'")
            return ParsedQuestion(
                raw_question=cleaned,
                keywords=["summary", "overview", "report"],
                intent="summary",
                is_summary_request=True,
                is_lookup_request=False
            )

        # Tokenize and normalize keywords
        tokens = [t for t in re.findall(r"[\w']+", normalized) if t and t not in STOP_WORDS]

        # Apply medical term alias expansions
        expanded_keywords: List[str] = []
        for token in tokens:
            token_clean = re.sub(r"'s$", "", token)
            if token_clean in MEDICAL_EXPANSIONS:
                expanded_keywords.extend(MEDICAL_EXPANSIONS[token_clean])
            else:
                expanded_keywords.append(token_clean)

        # Remove duplicate keywords while preserving order
        unique_keywords = list(dict.fromkeys(expanded_keywords))

        logger.info(f"Question parser parsed query '{cleaned}' -> Intent: lookup, Keywords: {unique_keywords}")
        return ParsedQuestion(
            raw_question=cleaned,
            keywords=unique_keywords,
            intent="lookup",
            is_summary_request=False,
            is_lookup_request=True
        )
