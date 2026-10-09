from tools.search import (
    build_entity_keys,
    is_strong_source,
    score_source_quality,
    search_web,
)


def count_strong_sources(
    sources: list[dict],
) -> int:
    """Count sources that satisfy CREDO's deterministic strong-source rule."""

    return sum(
        1
        for source in sources
        if is_strong_source(
            source
        )
    )


def _merge_evidence(
    target: dict,
    incoming: dict,
) -> None:
    """Merge useful evidence and metadata without duplicating entries."""

    target_evidence = target.setdefault(
        "evidence",
        [],
    )

    incoming_evidence = incoming.get(
        "evidence",
        [],
    )

    existing_signatures = {
        (
            item.get("url", ""),
            item.get("quote", ""),
        )
        for item in target_evidence
    }

    for item in incoming_evidence:
        signature = (
            item.get("url", ""),
            item.get("quote", ""),
        )

        if signature not in existing_signatures:
            target_evidence.append(
                item
            )
            existing_signatures.add(
                signature
            )

    alternate_urls = target.setdefault(
        "alternate_urls",
        [],
    )

    for url in [
        incoming.get("url", ""),
        incoming.get("canonical_url", ""),
    ]:
        if (
            url
            and url not in alternate_urls
            and url
            != target.get("url", "")
        ):
            alternate_urls.append(
                url
            )

    target["entity_keys"] = list(
        dict.fromkeys(
            target.get(
                "entity_keys",
                [],
            )
            + incoming.get(
                "entity_keys",
                [],
            )
        )
    )


def _source_preference(
    source: dict,
) -> tuple:
    """
    Determine which representation of the same research entity
    should be retained.
    """

    role = source.get(
        "source_role",
        "secondary",
    )

    role_score = {
        "primary": 3,
        "official": 2,
        "secondary_academic": 1,
        "secondary": 0,
    }.get(
        role,
        0,
    )

    quality = float(
        source.get(
            "quality_score",
            score_source_quality(
                source
            ),
        )
    )

    retrieval = float(
        source.get(
            "retrieval_score"
        )
        or 0.0
    )

    evidence_length = len(
        source.get(
            "evidence_span",
            "",
        )
    )

    return (
        role_score,
        quality,
        retrieval,
        evidence_length,
    )


def deduplicate_sources(
    sources: list[dict],
) -> list[dict]:
    """
    Deduplicate research entities using multiple identity keys.

    This can resolve cases such as:
        arXiv PDF
        arXiv HTML
        AlphaXiv mirror
        ACM paper page
        Semantic Scholar representation

    when they refer to the same paper.
    """

    unique_sources: list[dict] = []
    key_to_index: dict[str, int] = {}

    for raw_source in sources:
        source = dict(
            raw_source
        )

        if not source.get(
            "entity_keys"
        ):
            source["entity_keys"] = build_entity_keys(
                source
            )

        source["quality_score"] = score_source_quality(
            source
        )

        source["is_strong"] = is_strong_source(
            source
        )

        matching_indices = set()

        for key in source.get(
            "entity_keys",
            [],
        ):
            if key in key_to_index:
                matching_indices.add(
                    key_to_index[key]
                )

        # No match: new entity
        if not matching_indices:
            index = len(
                unique_sources
            )

            unique_sources.append(
                source
            )

            for key in source.get(
                "entity_keys",
                [],
            ):
                key_to_index[key] = index

            continue

        # Select the first matching cluster as the base.
        base_index = min(
            matching_indices
        )

        base_source = unique_sources[
            base_index
        ]

        # If multiple previous entities now connect,
        # merge those clusters first.
        extra_indices = sorted(
            matching_indices - {
                base_index
            },
            reverse=True,
        )

        for extra_index in extra_indices:
            extra_source = unique_sources[
                extra_index
            ]

            # Keep the stronger representation
            if _source_preference(
                extra_source
            ) > _source_preference(
                base_source
            ):
                preferred = extra_source
                weaker = base_source

                _merge_evidence(
                    preferred,
                    weaker,
                )

                base_source = preferred
                unique_sources[
                    base_index
                ] = preferred

            else:
                _merge_evidence(
                    base_source,
                    extra_source,
                )

            unique_sources.pop(
                extra_index
            )

            # Rebuild key map later. This is small-N research data,
            # so correctness is more important than micro-optimization.
            key_to_index = {}

            for idx, item in enumerate(
                unique_sources
            ):
                for key in item.get(
                    "entity_keys",
                    [],
                ):
                    key_to_index[key] = idx

            # Base index can shift after removing an earlier item.
            for idx, item in enumerate(
                unique_sources
            ):
                if item is base_source:
                    base_index = idx
                    break

        current_source = unique_sources[
            base_index
        ]

        # Prefer the stronger representation
        if _source_preference(
            source
        ) > _source_preference(
            current_source
        ):
            preferred = source
            weaker = current_source

            _merge_evidence(
                preferred,
                weaker,
            )

            preferred.setdefault(
                "alternate_urls",
                [],
            )

            if current_source.get(
                "url"
            ):
                if (
                    current_source["url"]
                    not in preferred[
                        "alternate_urls"
                    ]
                ):
                    preferred[
                        "alternate_urls"
                    ].append(
                        current_source["url"]
                    )

            unique_sources[
                base_index
            ] = preferred

        else:
            _merge_evidence(
                current_source,
                source,
            )

        # Rebuild all key mappings
        key_to_index = {}

        for idx, item in enumerate(
            unique_sources
        ):
            for key in item.get(
                "entity_keys",
                [],
            ):
                key_to_index[key] = idx

    return unique_sources


def build_targeted_queries(
    sub_question: str,
) -> list[str]:
    """
    Build targeted searches based on the actual information need.

    Correlation/downstream queries are checked before generic
    retrieval-metric queries so Q4/Q5 get the intended search.
    """

    question = sub_question.lower()

    queries = []

    # --------------------------------------------------------
    # 1. Retrieval ↔ downstream correlation
    # --------------------------------------------------------

    if (
        "correlation" in question
        or "downstream" in question
        or "trade-off" in question
        or "tradeoff" in question
    ):
        queries.extend(
            [
                (
                    f"{sub_question} "
                    "retrieval quality downstream generation "
                    "performance correlation empirical RAG paper"
                ),
                (
                    f"{sub_question} "
                    "retrieval metrics end-to-end RAG "
                    "human evaluation correlation study"
                ),
            ]
        )

        return queries

    # --------------------------------------------------------
    # 2. Retrieval metrics
    # --------------------------------------------------------

    if (
        "retrieval" in question
        and (
            "metric" in question
            or "metrics" in question
            or "quality" in question
        )
    ):
        queries.extend(
            [
                (
                    f"{sub_question} "
                    "retrieval evaluation metrics "
                    "precision recall MRR NDCG MAP RAG paper"
                ),
                (
                    f"{sub_question} "
                    "RAG retrieval quality evaluation "
                    "empirical study benchmark"
                ),
            ]
        )

        return queries

    # --------------------------------------------------------
    # 3. Generation evaluation / human judgment
    # --------------------------------------------------------

    if (
        "human" in question
        or "qualitative" in question
        or "faithfulness" in question
        or "coherence" in question
    ):
        queries.extend(
            [
                (
                    f"{sub_question} "
                    "RAG generation evaluation human judgment "
                    "faithfulness groundedness empirical paper"
                ),
                (
                    f"{sub_question} "
                    "RAG evaluation framework human alignment "
                    "RAGAS ARES RAGChecker paper"
                ),
            ]
        )

        return queries

    # --------------------------------------------------------
    # 4. Benchmarks / datasets / frameworks
    # --------------------------------------------------------

    if (
        "benchmark" in question
        or "dataset" in question
        or "framework" in question
        or "reproducibility" in question
    ):
        queries.extend(
            [
                (
                    f"{sub_question} "
                    "RAG benchmark dataset evaluation framework "
                    "CRAG RAGBench ARES RAGAS research paper"
                ),
                (
                    f"{sub_question} "
                    "RAG benchmark empirical comparison "
                    "coverage difficulty reproducibility paper"
                ),
            ]
        )

        return queries

    # --------------------------------------------------------
    # 5. Limitations / challenges
    # --------------------------------------------------------

    if (
        "challenge" in question
        or "limitation" in question
        or "future" in question
        or "open" in question
    ):
        queries.extend(
            [
                (
                    f"{sub_question} "
                    "RAG evaluation limitations open challenges "
                    "research paper survey"
                ),
                (
                    f"{sub_question} "
                    "RAG evaluation robustness reliability "
                    "future research benchmark paper"
                ),
            ]
        )

        return queries

    # --------------------------------------------------------
    # 6. Domain-specific research
    # --------------------------------------------------------

    if (
        "medical" in question
        or "healthcare" in question
        or "legal" in question
        or "scientific" in question
        or "domain-specific" in question
    ):
        queries.extend(
            [
                (
                    f"{sub_question} "
                    "domain-specific RAG evaluation benchmark "
                    "medical legal scientific research paper"
                ),
                (
                    f"{sub_question} "
                    "specialized RAG evaluation metrics "
                    "benchmark empirical study"
                ),
            ]
        )

        return queries

    # --------------------------------------------------------
    # 7. Generic research fallback
    # --------------------------------------------------------

    queries.extend(
        [
            (
                f"{sub_question} "
                "RAG evaluation research paper benchmark"
            ),
            (
                f"{sub_question} "
                "RAG evaluation empirical study"
            ),
        ]
    )

    return queries


def _annotate_global_coverage(
    research_results: dict[str, list[dict]],
) -> None:
    """
    Record which sub-questions are supported by each research entity.

    The same paper can support several questions, but the final filter
    will only keep one entity representation.
    """

    entity_questions: dict[str, list[str]] = {}

    for question, sources in research_results.items():
        for source in sources:
            entity_id = source.get(
                "paper_id"
            ) or source.get(
                "canonical_url",
                source.get("url", ""),
            )

            entity_questions.setdefault(
                entity_id,
                [],
            ).append(
                question
            )

    for question, sources in research_results.items():
        for source in sources:
            entity_id = source.get(
                "paper_id"
            ) or source.get(
                "canonical_url",
                source.get("url", ""),
            )

            source[
                "supporting_sub_questions"
            ] = list(
                dict.fromkeys(
                    entity_questions.get(
                        entity_id,
                        [question],
                    )
                )
            )


def research_sub_questions(
    sub_questions: list[str],
    max_results_per_question: int = 5,
    min_strong_sources: int = 2,
) -> dict[str, list[dict]]:
    """
    Research each planned sub-question.

    Behavior:
    1. Initial search
    2. Entity deduplication
    3. Deterministic source-quality scoring
    4. Strong-source coverage check
    5. Targeted second-round search when needed
    6. Final entity deduplication
    7. Global sub-question coverage annotation
    """

    research_results: dict[
        str,
        list[dict],
    ] = {}

    for i, question in enumerate(
        sub_questions,
        start=1,
    ):
        print(
            f"\nResearching sub-question "
            f"{i}/{len(sub_questions)}:"
        )

        print(question)

        # ----------------------------------------------------
        # Initial search
        # ----------------------------------------------------

        sources = search_web(
            question,
            max_results=max_results_per_question,
        )

        sources = deduplicate_sources(
            sources
        )

        strong_count = count_strong_sources(
            sources
        )

        print(
            f"Found {len(sources)} unique sources "
            f"({strong_count} strong)."
        )

        # ----------------------------------------------------
        # Adaptive targeted retrieval
        # ----------------------------------------------------

        if (
            strong_count
            < min_strong_sources
        ):
            print(
                "\nInsufficient strong evidence."
            )

            targeted_queries = build_targeted_queries(
                question
            )

            for targeted_query in targeted_queries:
                print(
                    "\nTargeted search:"
                )
                print(
                    targeted_query
                )

                additional_sources = search_web(
                    targeted_query,
                    max_results=max_results_per_question,
                )

                print(
                    f"Found "
                    f"{len(additional_sources)} "
                    f"additional sources."
                )

                sources.extend(
                    additional_sources
                )

                sources = deduplicate_sources(
                    sources
                )

                strong_count = count_strong_sources(
                    sources
                )

                if (
                    strong_count
                    >= min_strong_sources
                ):
                    print(
                        "Strong-source coverage "
                        "threshold reached."
                    )
                    break

        # ----------------------------------------------------
        # Sort for reproducible downstream behavior
        # ----------------------------------------------------

        sources.sort(
            key=lambda source: (
                float(
                    source.get(
                        "quality_score",
                        0.0,
                    )
                ),
                float(
                    source.get(
                        "retrieval_score"
                    )
                    or 0.0
                ),
            ),
            reverse=True,
        )

        research_results[
            question
        ] = sources

        print(
            f"Final evidence: "
            f"{len(sources)} unique sources "
            f"({count_strong_sources(sources)} strong)."
        )

    # --------------------------------------------------------
    # Cross-question coverage
    # --------------------------------------------------------

    _annotate_global_coverage(
        research_results
    )

    return research_results