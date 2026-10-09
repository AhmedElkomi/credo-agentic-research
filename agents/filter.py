import json
import os
import re

from dotenv import load_dotenv
from groq import Groq

from agents.researcher import deduplicate_sources
from tools.search import is_strong_source


MODEL = "openai/gpt-oss-20b"

STRONG_ROLES = {
    "primary_research",
    "primary",
    "official",
}

SECONDARY_ROLES = {
    "survey_review",
    "secondary_academic",
    "technical_repository",
    "technical_blog",
    "vendor",
    "web",
    "medium",
    "secondary",
}


# ============================================================
# JSON HELPERS
# ============================================================

def _clean_json_response(
    content: str,
) -> str:
    """
    Clean common LLM JSON formatting problems.

    Handles:
    - Markdown fences
    - Extra text before/after JSON
    - Leading/trailing whitespace
    - A known malformed duplicate source_id pattern
    """

    if not content:
        return ""

    content = content.strip()

    # Remove Markdown code fences.
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
        flags=re.IGNORECASE,
    )

    content = content.strip()

    # Recover the outer JSON object if the model added text.
    start = content.find("{")
    end = content.rfind("}")

    if start != -1 and end != -1 and end > start:
        content = content[
            start : end + 1
        ]

    # Fix the specific malformed pattern observed from Groq:
    #
    # "source_id":"source_id":"url:..."
    #
    content = re.sub(
        r'"source_id"\s*:\s*"source_id"\s*:\s*',
        '"source_id": ',
        content,
    )

    return content.strip()


def _parse_json_object(
    content: str,
) -> dict | None:
    """
    Parse an LLM JSON object safely.

    Returns None rather than raising when recovery fails.
    """

    cleaned = _clean_json_response(
        content
    )

    if not cleaned:
        return None

    try:
        parsed = json.loads(
            cleaned
        )

        if isinstance(
            parsed,
            dict,
        ):
            return parsed

    except json.JSONDecodeError:
        pass

    # --------------------------------------------------------
    # Second recovery attempt:
    # Extract the evaluations array manually.
    # --------------------------------------------------------

    match = re.search(
        r'"evaluations"\s*:\s*\[(.*)\]',
        cleaned,
        re.DOTALL,
    )

    if not match:
        return None

    array_content = match.group(1)

    # Try to recover individual JSON objects.
    object_matches = re.findall(
        r'\{[^{}]*\}',
        array_content,
        re.DOTALL,
    )

    recovered = []

    for object_text in object_matches:
        object_text = re.sub(
            r'"source_id"\s*:\s*"source_id"\s*:\s*',
            '"source_id": ',
            object_text,
        )

        try:
            item = json.loads(
                object_text
            )

        except json.JSONDecodeError:
            continue

        if isinstance(
            item,
            dict,
        ):
            recovered.append(
                item
            )

    if recovered:
        return {
            "evaluations": recovered
        }

    return None


# ============================================================
# SOURCE HELPERS
# ============================================================

def _source_id(
    source: dict,
) -> str:
    """Return a stable source identifier."""

    paper_id = source.get(
        "paper_id"
    )

    if paper_id:
        return paper_id

    canonical_url = source.get(
        "canonical_url"
    )

    if canonical_url:
        return canonical_url

    return source.get(
        "url",
        "",
    )


def _infer_source_role(
    source: dict,
) -> str:
    """
    Normalize the research role of a source.

    Academic does NOT automatically mean primary research.
    """

    existing_role = source.get(
        "source_role"
    )

    if existing_role:
        if existing_role == "primary":
            return "primary_research"

        if existing_role == "secondary":
            return "secondary"

        return existing_role

    source_type = source.get(
        "source_type",
        "web",
    )

    title = source.get(
        "title",
        "",
    ).lower()

    review_terms = [
        "survey",
        "systematic review",
        "literature review",
        "review of",
        "review:",
        "overview of",
        "state of the art",
        "mapping study",
    ]

    if source_type == "academic":

        if any(
            term in title
            for term in review_terms
        ):
            return "survey_review"

        return "primary_research"

    if source_type == "official":
        return "official"

    if source_type == "technical_repository":
        return "technical_repository"

    if source_type == "technical_blog":
        return "technical_blog"

    if source_type == "vendor":
        return "vendor"

    if source_type == "medium":
        return "medium"

    return "web"


def _source_is_strong(
    source: dict,
) -> bool:
    """
    Determine whether a source qualifies as strong evidence.

    Uses the explicit source role produced by search.py.
    """

    role = source.get(
        "source_role",
        "",
    )

    if role in STRONG_ROLES:
        return True

    return bool(
        source.get(
            "is_strong",
            False,
        )
    )


# ============================================================
# RESEARCH FLATTENING
# ============================================================

def _flatten_research(
    research_results: dict[str, list[dict]],
) -> list[dict]:
    """
    Flatten grouped research results while preserving
    sub-question provenance.
    """

    source_map = {}

    for question, sources in research_results.items():

        for source in sources:

            source_copy = dict(
                source
            )

            source_id = _source_id(
                source_copy
            )

            if not source_id:
                continue

            existing = source_map.get(
                source_id
            )

            if existing is None:

                existing = source_copy

                existing[
                    "supporting_sub_questions"
                ] = []

                source_map[
                    source_id
                ] = existing

            existing_questions = (
                existing.setdefault(
                    "supporting_sub_questions",
                    [],
                )
            )

            if question not in existing_questions:
                existing_questions.append(
                    question
                )

    return list(
        source_map.values()
    )


# ============================================================
# CANDIDATE RANKING
# ============================================================

def _candidate_score(
    source: dict,
) -> float:
    """
    Deterministic pre-ranking.

    Used only to reduce the candidate pool.
    """

    quality = float(
        source.get(
            "quality_score",
            0.0,
        )
        or 0.0
    )

    retrieval = float(
        source.get(
            "retrieval_score",
            0.0,
        )
        or 0.0
    )

    strong_bonus = (
        0.15
        if is_strong_source(source)
        else 0.0
    )

    return (
        quality * 0.55
        + retrieval * 0.30
        + strong_bonus
    )


def _prepare_llm_candidates(
    candidates: list[dict],
) -> list[dict]:
    """
    Create compact evidence records for the LLM.
    """

    prepared = []

    for source in candidates:

        evidence = source.get(
            "evidence_span",
            source.get(
                "snippet",
                "",
            ),
        )

        prepared.append(
            {
                "source_id": _source_id(
                    source
                ),
                "title": source.get(
                    "title",
                    "",
                ),
                "url": source.get(
                    "url",
                    "",
                ),
                "source_type": source.get(
                    "source_type",
                    "",
                ),
                "source_role": source.get(
                    "source_role",
                    "",
                ),
                "published_year": source.get(
                    "published_year"
                ),
                "evidence": str(
                    evidence
                )[:700],
            }
        )

    return prepared


# ============================================================
# SEMANTIC EVIDENCE EVALUATION
# ============================================================

def _validate_evaluations(
    evaluations,
    candidate_ids: set[str],
    max_selected: int,
) -> list[dict]:
    """
    Validate and normalize LLM-generated evaluations.

    Only evaluations referring to actual candidates survive.
    """

    if not isinstance(
        evaluations,
        list,
    ):
        return []

    valid = []

    for item in evaluations:

        if not isinstance(
            item,
            dict,
        ):
            continue

        source_id = str(
            item.get(
                "source_id",
                "",
            )
        ).strip()

        if not source_id:
            continue

        # Never accept a source ID that wasn't provided.
        if source_id not in candidate_ids:
            continue

        try:
            relevance = float(
                item.get(
                    "relevance_score",
                    0.0,
                )
            )

            support = float(
                item.get(
                    "evidence_support_score",
                    0.0,
                )
            )

        except (
            TypeError,
            ValueError,
        ):
            continue

        if not (
            0.0
            <= relevance
            <= 1.0
        ):
            relevance = max(
                0.0,
                min(
                    1.0,
                    relevance,
                ),
            )

        if not (
            0.0
            <= support
            <= 1.0
        ):
            support = max(
                0.0,
                min(
                    1.0,
                    support,
                ),
            )

        reason = str(
            item.get(
                "reason",
                "",
            )
        ).strip()

        valid.append(
            {
                "source_id": source_id,
                "relevance_score": relevance,
                "evidence_support_score": support,
                "reason": reason,
            }
        )

    # Deduplicate source evaluations.
    unique = {}

    for item in valid:
        unique[
            item["source_id"]
        ] = item

    valid = list(
        unique.values()
    )

    valid.sort(
        key=lambda item: (
            item[
                "evidence_support_score"
            ],
            item[
                "relevance_score"
            ],
        ),
        reverse=True,
    )

    return valid[
        :max_selected
    ]


def _semantic_select_for_question(
    research_question: str,
    sub_question: str,
    candidates: list[dict],
    max_selected: int = 4,
) -> list[dict]:
    """
    Evaluate evidence for ONE sub-question.

    The LLM determines:
    - semantic relevance
    - direct evidence support

    It does NOT determine source quality.
    """

    if not candidates:
        return []

    load_dotenv()

    api_key = os.getenv(
        "GROQ_API_KEY"
    )

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is missing. "
            "Check your .env file."
        )

    client = Groq(
        api_key=api_key
    )

    prepared = _prepare_llm_candidates(
        candidates
    )

    candidate_ids = {
        item["source_id"]
        for item in prepared
    }

    prompt = f"""
Original research question:
{research_question}

Specific sub-question:
{sub_question}

Evaluate the research evidence for the specific
sub-question.

For each useful candidate provide:

- source_id
- relevance_score: 0.0 to 1.0
- evidence_support_score: 0.0 to 1.0
- reason

Definitions:

relevance_score:
How relevant the source is to the specific sub-question.

evidence_support_score:
How directly the supplied evidence supports the
specific sub-question.

Rules:

- Judge only the evidence supplied below.
- Do not judge source quality.
- Do not reward fame, citations, recency, or company reputation.
- Do not invent facts.
- Do not infer unsupported claims.
- Generic discussion of RAG is not enough.
- Prefer direct evidence.
- Return at most {max_selected} useful candidates.
- If a candidate does not provide useful evidence,
  omit it.

The source_id MUST exactly match one of the supplied
candidate source IDs.

Return ONLY JSON using this structure:

{{
  "evaluations": [
    {{
      "source_id": "exact candidate source ID",
      "relevance_score": 0.0,
      "evidence_support_score": 0.0,
      "reason": "brief reason"
    }}
  ]
}}

Candidates:

{json.dumps(
    prepared,
    ensure_ascii=False,
    indent=2,
)}
"""

    try:

        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a research evidence "
                        "verification component. "
                        "Return one JSON object only. "
                        "Do not include Markdown."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0.0,
            max_completion_tokens=1600,
            reasoning_effort="low",
            include_reasoning=False,
            response_format={
                "type": "json_object"
            },
        )

        content = (
            response.choices[0]
            .message
            .content
        )

        parsed = _parse_json_object(
            content
        )

        if not parsed:
            raise ValueError(
                "The model returned invalid JSON."
            )

        evaluations = parsed.get(
            "evaluations",
            [],
        )

        valid = _validate_evaluations(
            evaluations,
            candidate_ids,
            max_selected,
        )

        if not valid:
            raise ValueError(
                "The model returned no valid "
                "candidate evaluations."
            )

        return valid

    except Exception as exc:

        print(
            "\nSemantic evidence evaluation "
            "failed."
        )

        print(
            f"Reason: {exc}"
        )

        return []


# ============================================================
# EVALUATE ALL SUB-QUESTIONS
# ============================================================

def _evaluate_sources_by_question(
    research_question: str,
    research_results: dict[str, list[dict]],
    all_sources: list[dict],
) -> dict[str, list[dict]]:
    """
    Evaluate evidence independently for every sub-question.
    """

    source_map = {
        _source_id(source): source
        for source in all_sources
    }

    evaluations = {}

    for sub_question in research_results:

        candidates = [
            source
            for source in all_sources
            if sub_question
            in source.get(
                "supporting_sub_questions",
                [],
            )
        ]

        candidates.sort(
            key=_candidate_score,
            reverse=True,
        )

        # Keep LLM context compact.
        candidates = candidates[
            :6
        ]

        semantic_results = (
            _semantic_select_for_question(
                research_question,
                sub_question,
                candidates,
                max_selected=4,
            )
        )

        evaluated = []

        for item in semantic_results:

            source = source_map.get(
                item[
                    "source_id"
                ]
            )

            if not source:
                continue

            enriched = dict(
                source
            )

            enriched[
                "relevance_score"
            ] = item[
                "relevance_score"
            ]

            enriched[
                "evidence_support_score"
            ] = item[
                "evidence_support_score"
            ]

            enriched[
                "filter_reason"
            ] = item[
                "reason"
            ]

            evaluated.append(
                enriched
            )

        evaluations[
            sub_question
        ] = evaluated

    return evaluations


# ============================================================
# FINAL EVIDENCE SCORE
# ============================================================

def _evidence_score(
    source: dict,
) -> float:
    """
    Final deterministic ranking score.

    Quality: 25%
    Relevance: 25%
    Evidence support: 40%
    Retrieval: 10%
    Strong-source bonus: +0.10
    """

    quality = float(
        source.get(
            "quality_score",
            0.0,
        )
        or 0.0
    )

    relevance = float(
        source.get(
            "relevance_score",
            0.0,
        )
        or 0.0
    )

    support = float(
        source.get(
            "evidence_support_score",
            0.0,
        )
        or 0.0
    )

    retrieval = float(
        source.get(
            "retrieval_score",
            0.0,
        )
        or 0.0
    )

    strong_bonus = (
        0.10
        if _source_is_strong(source)
        else 0.0
    )

    return (
        quality * 0.25
        + relevance * 0.25
        + support * 0.40
        + retrieval * 0.10
        + strong_bonus
    )


# ============================================================
# DETERMINISTIC FALLBACK
# ============================================================

def _build_deterministic_fallback(
    candidates: list[dict],
    max_selected: int = 4,
) -> list[dict]:
    """
    Build useful evaluations without an LLM.

    This is intentionally conservative.

    Strong sources receive higher support because their
    source role establishes that they are research/official
    evidence, while secondary sources receive a moderate
    support score.

    The fallback does NOT pretend to know semantic relevance.
    Retrieval score and evidence availability provide the
    deterministic signal.
    """

    candidates = list(
        candidates
    )

    candidates.sort(
        key=_candidate_score,
        reverse=True,
    )

    fallback = []

    for source in candidates[
        :max_selected
    ]:

        enriched = dict(
            source
        )

        retrieval = float(
            source.get(
                "retrieval_score",
                0.0,
            )
            or 0.0
        )

        # Retrieval score is typically already bounded,
        # but clamp defensively.
        retrieval = max(
            0.0,
            min(
                1.0,
                retrieval,
            ),
        )

        # The fallback cannot truly assess semantic relevance.
        # Use retrieval as a conservative proxy.
        relevance = retrieval

        # Evidence presence is important.
        has_evidence = bool(
            source.get(
                "evidence_span"
            )
            or source.get(
                "snippet"
            )
        )

        if _source_is_strong(
            source
        ):
            support = (
                0.70
                if has_evidence
                else 0.55
            )

        else:
            support = (
                0.60
                if has_evidence
                else 0.45
            )

        enriched[
            "relevance_score"
        ] = relevance

        enriched[
            "evidence_support_score"
        ] = support

        enriched[
            "filter_reason"
        ] = (
            "Deterministic evidence fallback "
            "after semantic evaluation was unavailable."
        )

        fallback.append(
            enriched
        )

    return fallback


# ============================================================
# FINAL SOURCE SELECTION
# ============================================================

def _select_final_sources(
    evaluations: dict[str, list[dict]],
    max_sources: int,
) -> tuple[list[dict], dict[str, dict]]:
    """
    Select final evidence with sub-question coverage.

    Policy:

    1. Determine evidence status.
    2. Prefer strong sources.
    3. Guarantee useful coverage where available.
    4. Fill remaining capacity with high-value evidence.
    5. Assign citation IDs.
    """

    selected = []
    selected_ids = set()

    status = {}

    # --------------------------------------------------------
    # Step 1: Evidence status.
    # --------------------------------------------------------

    for question, sources in evaluations.items():

        strong_sources = [
            source
            for source in sources
            if _source_is_strong(
                source
            )
            and source.get(
                "evidence_support_score",
                0.0,
            ) >= 0.55
        ]

        useful_sources = [
            source
            for source in sources
            if source.get(
                "evidence_support_score",
                0.0,
            ) >= 0.55
        ]

        if len(
            strong_sources
        ) >= 2:

            question_status = (
                "sufficient"
            )

        elif len(
            strong_sources
        ) == 1:

            question_status = (
                "limited_evidence"
            )

        elif len(
            useful_sources
        ) > 0:

            question_status = (
                "limited_evidence"
            )

        else:

            question_status = (
                "insufficient_evidence"
            )

        status[
            question
        ] = {
            "status": question_status,
            "strong_source_count": len(
                strong_sources
            ),
            "useful_source_count": len(
                useful_sources
            ),
        }

    # --------------------------------------------------------
    # Step 2: Strong-source coverage.
    # --------------------------------------------------------

    for question, sources in evaluations.items():

        strong_sources = [
            source
            for source in sources
            if _source_is_strong(
                source
            )
            and source.get(
                "evidence_support_score",
                0.0,
            ) >= 0.55
        ]

        strong_sources.sort(
            key=_evidence_score,
            reverse=True,
        )

        for source in strong_sources[
            :2
        ]:

            source_id = _source_id(
                source
            )

            if source_id in selected_ids:
                continue

            if len(selected) >= max_sources:
                break

            selected.append(
                source
            )

            selected_ids.add(
                source_id
            )

    # --------------------------------------------------------
    # Step 3: Useful coverage for every question.
    # --------------------------------------------------------

    for question, sources in evaluations.items():

        already_covered = any(
            question
            in source.get(
                "supporting_sub_questions",
                [],
            )
            for source in selected
        )

        if already_covered:
            continue

        useful_sources = [
            source
            for source in sources
            if source.get(
                "evidence_support_score",
                0.0,
            ) >= 0.55
        ]

        useful_sources.sort(
            key=_evidence_score,
            reverse=True,
        )

        for source in useful_sources:

            source_id = _source_id(
                source
            )

            if source_id in selected_ids:
                continue

            if len(selected) >= max_sources:
                break

            selected.append(
                source
            )

            selected_ids.add(
                source_id
            )

            break

    # --------------------------------------------------------
    # Step 4: Fill remaining capacity.
    # --------------------------------------------------------

    remaining = []

    for sources in evaluations.values():
        remaining.extend(
            sources
        )

    unique_remaining = {}

    for source in remaining:

        source_id = _source_id(
            source
        )

        if source_id:
            unique_remaining[
                source_id
            ] = source

    remaining = list(
        unique_remaining.values()
    )

    remaining.sort(
        key=_evidence_score,
        reverse=True,
    )

    secondary_cap = max(
        1,
        max_sources // 3,
    )

    secondary_count = sum(
        1
        for source in selected
        if not _source_is_strong(
            source
        )
    )

    for source in remaining:

        if len(selected) >= max_sources:
            break

        source_id = _source_id(
            source
        )

        if source_id in selected_ids:
            continue

        is_secondary = not _source_is_strong(
            source
        )

        if (
            is_secondary
            and secondary_count
            >= secondary_cap
        ):
            continue

        selected.append(
            source
        )

        selected_ids.add(
            source_id
        )

        if is_secondary:
            secondary_count += 1

    # --------------------------------------------------------
    # Step 5: Final ordering.
    # --------------------------------------------------------

    selected.sort(
        key=lambda source: (
            _source_is_strong(
                source
            ),
            _evidence_score(
                source
            ),
        ),
        reverse=True,
    )

    # --------------------------------------------------------
    # Step 6: Citation IDs.
    # --------------------------------------------------------

    for index, source in enumerate(
        selected,
        start=1,
    ):

        source[
            "citation_id"
        ] = index

    return (
        selected,
        status,
    )


# ============================================================
# PUBLIC FILTER FUNCTION
# ============================================================

def filter_sources(
    research_results: dict[str, list[dict]],
    max_sources: int = 10,
    research_question: str = "",
) -> tuple[
    list[dict],
    dict[str, dict],
]:
    """
    Build the final evidence package.

    Pipeline:

    1. Flatten and deduplicate.
    2. Normalize source roles.
    3. Preserve sub-question provenance.
    4. Perform semantic evidence evaluation.
    5. Recover from malformed LLM output.
    6. Use deterministic fallback when needed.
    7. Guarantee strong-source coverage where available.
    8. Preserve evidence-status information.
    9. Fill remaining capacity.
    10. Assign citation IDs.
    """

    if not research_results:
        return [], {}

    all_sources = _flatten_research(
        research_results
    )

    if not all_sources:
        return [], {}

    # --------------------------------------------------------
    # Global metadata normalization.
    # --------------------------------------------------------

    for source in all_sources:

        source[
            "source_role"
        ] = _infer_source_role(
            source
        )

        source[
            "is_strong"
        ] = _source_is_strong(
            source
        )

        source.setdefault(
            "quality_score",
            0.0,
        )

        source.setdefault(
            "retrieval_score",
            0.0,
        )

        source.setdefault(
            "evidence_span",
            source.get(
                "snippet",
                "",
            ),
        )

    # --------------------------------------------------------
    # Semantic evaluation.
    # --------------------------------------------------------

    if research_question:

        evaluations = (
            _evaluate_sources_by_question(
                research_question,
                research_results,
                all_sources,
            )
        )

    else:

        evaluations = {
            question: []
            for question in research_results
        }

    # --------------------------------------------------------
    # Deterministic fallback for each failed question.
    # --------------------------------------------------------

    for question in research_results:

        if evaluations.get(
            question
        ):
            continue

        candidates = [
            source
            for source in all_sources
            if question
            in source.get(
                "supporting_sub_questions",
                [],
            )
        ]

        fallback = (
            _build_deterministic_fallback(
                candidates,
                max_selected=4,
            )
        )

        evaluations[
            question
        ] = fallback

    # --------------------------------------------------------
    # Final selection.
    # --------------------------------------------------------

    final_sources, evidence_status = (
        _select_final_sources(
            evaluations,
            max_sources,
        )
    )

    return (
        final_sources,
        evidence_status,
    )