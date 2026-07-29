"""
Prompt builder module for the experimental Gemini PDF localization pipeline.
Constructs exact system prompt using Python f-strings as specified.
"""

def build_prompt(question: str) -> str:
    """
    Generate prompt for Gemini PDF localization using Python f-strings.

    Args:
        question (str): The natural language query or requested field name.

    Returns:
        str: Fully formatted prompt string for Gemini API.
    """
    return f"""
You are a precise PDF document localization system.

Analyze ONLY the uploaded PDF.

Question:

{question}

Find the exact location in the PDF that answers the question.

Return ONLY JSON.

{{
  "found": true,
  "page": 1,

  "bounding_box": {{
      "x1": 0.1,
      "y1": 0.2,
      "x2": 0.5,
      "y2": 0.4
  }},

  "matched_text": "exact extracted text phrase or answer",

  "confidence": 0.99
}}

If the answer is missing:

{{
 "found": false
}}

Never hallucinate.

Never use outside knowledge.
"""
