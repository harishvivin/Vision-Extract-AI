"""
Question Parser Engine.
Uses Regex and NLP (spaCy / pattern matching) to parse question text into structured metadata.
"""

import re
import logging
from dataclasses import dataclass
from typing import Dict, Optional, List, Any

logger = logging.getLogger(__name__)

@dataclass
class ParsedQuestion:
    """Structured question metadata."""
    raw_question: str
    object: str
    color: Optional[str]
    position: Optional[str]
    filename: str
    primary_prompt: str
    object_prompt: str
    detailed_prompt: str


class QuestionParser:
    """Parser to extract target object, color, position, and target filename from question text."""

    def __init__(self):
        """Initialize QuestionParser with regex patterns and known entity keyword mappings."""
        # Common color patterns
        self.color_pattern = re.compile(
            r'\b(YELLOW|RED|SILVER|GREEN-and-yellow patterned|GREEN|GREEN-AND-YELLOW|PINK|PURPLE|BLUE|WHITE|BLACK|BROWN|ORANGE|pistachio)\b',
            re.IGNORECASE
        )
        
        # Spatial position patterns
        self.position_pattern = re.compile(
            r'\((bottom-left|top-centre|top-center|front-right|right side|front-centre|front-center|front|foreground|centre|center|middle)\)|'
            r'\b(bottom-left|top-centre|top-center|front-right|right side|front-centre|front-center|front|foreground|centre|center|middle)\b',
            re.IGNORECASE
        )

        # Output filename pattern e.g., 01_yellow_tulips.png
        self.filename_pattern = re.compile(r'(\d{2}_[\w-]+\.png)', re.IGNORECASE)

    def parse(self, question_text: str) -> ParsedQuestion:
        """
        Parse raw question text into ParsedQuestion object.

        Args:
            question_text (str): Question text extracted from page.

        Returns:
            ParsedQuestion: Structured metadata container.
        """
        cleaned_text = " ".join(question_text.split())
        logger.info(f"Parsing question text: '{cleaned_text}'")

        # 1. Extract Filename
        fn_match = self.filename_pattern.search(cleaned_text)
        filename = fn_match.group(1) if fn_match else "extracted_object.png"

        # 2. Extract Position
        pos_match = self.position_pattern.search(cleaned_text)
        position = None
        if pos_match:
            # Group 1 (in parens) or Group 2 (standalone word)
            position = (pos_match.group(1) or pos_match.group(2)).lower()

        # 3. Extract Color
        color_matches = self.color_pattern.findall(cleaned_text)
        color = color_matches[0].lower() if color_matches else None
        if color == "pistachio":
            color = "green"

        # 4. Extract Object Name
        object_name = self._extract_object_name(cleaned_text, color, position, filename)

        # 5. Generate Prompts for Grounding DINO
        primary_prompt = f"{color} {object_name}" if color else object_name
        object_prompt = object_name
        detailed_prompt = f"{primary_prompt} {position}" if position else primary_prompt

        parsed = ParsedQuestion(
            raw_question=cleaned_text,
            object=object_name,
            color=color,
            position=position,
            filename=filename,
            primary_prompt=primary_prompt.strip(),
            object_prompt=object_prompt.strip(),
            detailed_prompt=detailed_prompt.strip()
        )

        logger.info(
            f"Parsed Result -> Object: '{parsed.object}', Color: '{parsed.color}', "
            f"Position: '{parsed.position}', Filename: '{parsed.filename}'"
        )
        return parsed

    def _extract_object_name(self, text: str, color: Optional[str], position: Optional[str], filename: str) -> str:
        """
        Extract the core object name from question text or filename fallback.

        Args:
            text (str): Question text.
            color (Optional[str]): Extracted color.
            position (Optional[str]): Extracted position.
            filename (str): Target filename.

        Returns:
            str: Object name.
        """
        # Specific rule-based matches based on target sentence structure:
        # e.g., "Crop out only the bunch of YELLOW tulips (bottom-left)..." -> "tulips"
        # e.g., "Crop out only the cluster of RED tulips (top-centre)..." -> "tulips"
        # e.g., "Crop out only the SILVER car on the front-right..." -> "car"
        # e.g., "Crop out only the GREEN-and-yellow patterned balloon..." -> "balloon"
        # e.g., "Crop out only the pile of green PEARS (centre)..." -> "pears"
        # e.g., "Crop out only the MIDDLE hanging braid of peppers..." -> "braid of peppers" or "peppers"
        # e.g., "Crop out only the large SHEEP standing..." -> "sheep"
        # e.g., "Crop out only the DUCK with the green head..." -> "duck"
        # e.g., "Crop out only the GREEN (pistachio) macaron..." -> "macaron"
        # e.g., "Crop out only the pink FLAMINGO in the foreground..." -> "flamingo"

        crop_match = re.search(r'Crop (?:out )?only the (.*?)(?:and save|\.|$)', text, re.IGNORECASE)
        target_phrase = crop_match.group(1) if crop_match else text

        # Clean words like "cluster of", "bunch of", "pile of", etc.
        target_phrase = re.sub(r'\b(bunch of|cluster of|pile of|large|stack of)\b', '', target_phrase, flags=re.IGNORECASE)
        
        # Remove filename if present in target phrase
        target_phrase = re.sub(r'save (the cropped region )?as.*$', '', target_phrase, flags=re.IGNORECASE)

        # Match specific known object nouns
        known_objects = [
            "tulips", "car", "balloon", "pears", "pear", "braid of peppers", "peppers", "braid",
            "sheep", "duck", "macaron", "flamingo", "flower", "fruit"
        ]

        for obj in known_objects:
            if re.search(rf'\b{obj}\b', target_phrase, re.IGNORECASE):
                return obj

        # Fallback to parsing filename (e.g. 01_yellow_tulips.png -> tulips)
        name_part = filename.replace('.png', '')
        parts = name_part.split('_')[1:] # drop numbers e.g. 01
        filtered_parts = [p for p in parts if p.lower() not in ['yellow', 'red', 'silver', 'green', 'pink', 'middle', 'front']]
        if filtered_parts:
            return " ".join(filtered_parts)

        return "object"
