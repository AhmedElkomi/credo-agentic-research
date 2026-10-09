from agents.planner import create_research_plan
from agents.researcher import research_sub_questions
from agents.filter import filter_sources


def main():
    query = input(
        "\nResearch question:\n> "
    ).strip()

    if not query:
        raise ValueError(
            "Research question cannot be empty."
        )

    print(
        "\n========== PLANNING ==========\n"
    )

    sub_questions = create_research_plan(
        query
    )

    for i, question in enumerate(
        sub_questions,
        start=1,
    ):
        print(
            f"{i}. {question}"
        )

    print(
        "\n========== RESEARCH ==========\n"
    )

    research_results = research_sub_questions(
        sub_questions,
        max_results_per_question=5,
        min_strong_sources=2,
    )

    print(
        "\n========== SOURCE FILTERING ==========\n"
    )

    filtered_sources, evidence_status = filter_sources(
        research_results,
        max_sources=10,
        research_question=query,
    )

    print(
        "\n========== EVIDENCE STATUS ==========\n"
    )

    for i, question in enumerate(
        sub_questions,
        start=1,
    ):
        status = evidence_status.get(
            question,
            {},
        )

        print(
            f"\nQ{i}: {question}"
        )

        print(
            f"Status: "
            f"{status.get('status', 'unknown')}"
        )

        print(
            f"Strong sources: "
            f"{status.get('strong_source_count', 0)}"
        )

        print(
            f"Useful sources: "
            f"{status.get('useful_source_count', 0)}"
        )

    print(
        "\n========== FINAL EVIDENCE ==========\n"
    )

    print(
        f"Selected "
        f"{len(filtered_sources)} "
        f"final evidence sources.\n"
    )

    for source in filtered_sources:
        print(
            f"[{source.get('citation_id')}] "
            f"{source.get('title', '')}"
        )

        print(
            f"URL: "
            f"{source.get('url', '')}"
        )

        print(
            f"Type: "
            f"{source.get('source_type', '')}"
        )

        print(
            f"Role: "
            f"{source.get('source_role', '')}"
        )

        print(
            f"Quality: "
            f"{source.get('quality_score', 0.0):.3f}"
        )

        print(
            f"Relevance: "
            f"{source.get('relevance_score', 0.0):.3f}"
        )

        print(
            f"Evidence support: "
            f"{source.get('evidence_support_score', 0.0):.3f}"
        )

        print(
            f"Strong: "
            f"{source.get('is_strong', False)}"
        )

        print(
            "Supports:"
        )

        for question in source.get(
            "supporting_sub_questions",
            [],
        ):
            print(
                f"  - {question}"
            )

        print(
            f"Reason: "
            f"{source.get('filter_reason', '')}"
        )

        print(
            f"Paper ID: "
            f"{source.get('paper_id', '')}"
        )

        doi = source.get(
            "doi"
        )

        if doi:
            print(
                f"DOI: {doi}"
            )

        arxiv_id = source.get(
            "arxiv_id"
        )

        if arxiv_id:
            print(
                f"arXiv: {arxiv_id}"
            )

        print(
            "Evidence:"
        )

        evidence = source.get(
            "evidence",
            [],
        )

        if evidence:
            print(
                evidence[0].get(
                    "quote",
                    "",
                )[:700]
            )

        print(
            "-" * 90
        )


if __name__ == "__main__":
    main()