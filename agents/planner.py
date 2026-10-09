import json
import os
import re

from dotenv import load_dotenv
from groq import Groq


MODEL = "openai/gpt-oss-20b"


def _clean_json_response(content: str) -> str:
    """
    Clean common LLM JSON formatting issues.

    Handles:
    - Markdown code fences
    - Extra text before/after JSON
    - Leading/trailing whitespace
    """

    content = content.strip()

    # Remove markdown fences.
    if content.startswith("```"):
        content = re.sub(
            r"^```(?:json)?\s*",
            "",
            content,
            flags=re.IGNORECASE,
        )

        content = re.sub(
            r"\s*```$",
            "",
            content,
        )

        content = content.strip()

    # If the model added text before/after the JSON,
    # isolate the outer JSON object.
    start = content.find("{")
    end = content.rfind("}")

    if start != -1 and end != -1 and end > start:
        content = content[start : end + 1]

    return content.strip()


def _parse_plan(content: str) -> dict:
    """
    Parse planner output.

    First attempts strict JSON parsing.
    If that fails, attempts a safe extraction of
    the sub_questions array.
    """

    cleaned = _clean_json_response(content)

    # Normal path.
    try:
        return json.loads(cleaned)

    except json.JSONDecodeError:
        pass

    # Fallback:
    # Extract the sub_questions array even if the model
    # accidentally added malformed text elsewhere.
    match = re.search(
        r'"sub_questions"\s*:\s*\[(.*?)\]',
        cleaned,
        re.DOTALL,
    )

    if not match:
        raise ValueError(
            "Planner returned invalid JSON and "
            "the sub_questions array could not be recovered.\n\n"
            f"Raw response:\n{content}"
        )

    array_content = match.group(1)

    # Extract JSON-style quoted strings.
    questions = re.findall(
        r'"((?:\\.|[^"\\])*)"',
        array_content,
    )

    if not questions:
        raise ValueError(
            "Planner returned an invalid sub_questions array.\n\n"
            f"Raw response:\n{content}"
        )

    questions = [
        bytes(question, "utf-8").decode(
            "unicode_escape"
        )
        for question in questions
    ]

    return {
        "sub_questions": questions
    }


def create_research_plan(
    research_question: str,
    min_questions: int = 3,
    max_questions: int = 5,
) -> list[str]:
    """
    Decompose a research question into focused,
    evidence-oriented sub-questions.

    The planner should:
    - Produce 3–5 questions.
    - Cover different dimensions of the topic.
    - Avoid redundant questions.
    - Prefer questions that can realistically be answered
      using available research literature.
    - Avoid forcing a question merely to reach five items.
    """

    load_dotenv()

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is missing. Check your .env file."
        )

    client = Groq(
        api_key=api_key
    )

    system_prompt = f"""
You are the research planning component of CREDO,
an Agentic AI Research & Evidence Verification Platform.

Your task is to decompose a research question into
{min_questions} to {max_questions} focused research sub-questions.

Rules:

1. Return ONLY valid JSON.
2. The JSON must have exactly this structure:

{{
  "sub_questions": [
    "question 1",
    "question 2",
    "question 3"
  ]
}}

3. Produce between {min_questions} and {max_questions} questions.
4. Questions must be meaningfully different.
5. Cover different dimensions of the original question.
6. Questions must be answerable through credible research literature,
   academic papers, official documentation, or strong technical sources.
7. Do not create questions merely to reach five questions.
8. Avoid vague questions.
9. Avoid asking the same question in different wording.
10. Prefer questions that can later be supported with citations.
11. Do not answer the questions.
12. Do not include Markdown.
13. Do not include explanations outside the JSON.
14. Escape quotation marks correctly inside JSON strings.
"""

    user_prompt = f"""
Research question:

{research_question}

Create the research plan now.
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        temperature=0,
        max_completion_tokens=1200,
        response_format={
            "type": "json_object"
        },
    )

    content = (
        response.choices[0]
        .message
        .content
        .strip()
    )

    plan = _parse_plan(content)

    sub_questions = plan.get(
        "sub_questions"
    )

    if not isinstance(
        sub_questions,
        list,
    ):
        raise ValueError(
            "Planner JSON does not contain a valid "
            "'sub_questions' list."
        )

    # Clean and validate questions.
    cleaned_questions = []

    for question in sub_questions:
        if not isinstance(
            question,
            str,
        ):
            continue

        question = question.strip()

        if question:
            cleaned_questions.append(
                question
            )

    # Remove exact duplicates while preserving order.
    unique_questions = []

    seen = set()

    for question in cleaned_questions:
        key = question.lower()

        if key not in seen:
            seen.add(key)
            unique_questions.append(
                question
            )

    if not (
        min_questions
        <= len(unique_questions)
        <= max_questions
    ):
        raise ValueError(
            "Planner returned an invalid number of "
            f"sub-questions: {len(unique_questions)}. "
            f"Expected {min_questions}–{max_questions}."
        )

    return unique_questions