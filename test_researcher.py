from agents.planner import create_research_plan
from agents.researcher import (
    research_sub_questions,
    count_strong_sources,
)


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

    if not sub_questions:
        raise ValueError(
            "Planner returned no sub-questions."
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
        "\n========== EVIDENCE SUMMARY ==========\n"
    )

    total_sources = 0
    total_strong_sources = 0

    for i, (
        question,
        sources,
    ) in enumerate(
        research_results.items(),
        start=1,
    ):
        strong_count = count_strong_sources(
            sources
        )

        total_sources += len(
            sources
        )

        total_strong_sources += (
            strong_count
        )

        print(
            f"\n[{i}] {question}"
        )

        print(
            f"Sources: {len(sources)}"
        )

        print(
            f"Strong sources: "
            f"{strong_count}"
        )

        for j, source in enumerate(
            sources,
            start=1,
        ):
            print(
                f"\n  {j}. "
                f"{source.get('title', '')}"
            )

            print(
                f"     Type: "
                f"{source.get('source_type', '')}"
            )

            print(
                f"     URL: "
                f"{source.get('url', '')}"
            )

            print(
                f"     Paper ID: "
                f"{source.get('paper_id', '')}"
            )

            doi = source.get(
                "doi"
            )

            if doi:
                print(
                    f"     DOI: {doi}"
                )

            arxiv_id = source.get(
                "arxiv_id"
            )

            if arxiv_id:
                print(
                    f"     arXiv ID: "
                    f"{arxiv_id}"
                )

            published_year = source.get(
                "published_year"
            )

            if published_year:
                print(
                    f"     Published: "
                    f"{published_year}"
                )

            citation_count = source.get(
                "citation_count"
            )

            if citation_count is not None:
                print(
                    f"     Citations: "
                    f"{citation_count}"
                )

            retrieval_score = source.get(
                "retrieval_score"
            )

            if retrieval_score is not None:
                print(
                    f"     Retrieval score: "
                    f"{retrieval_score}"
                )

            evidence = source.get(
                "evidence",
                [],
            )

            print(
                f"     Evidence spans: "
                f"{len(evidence)}"
            )

            if evidence:
                quote = evidence[0].get(
                    "quote",
                    "",
                )

                print(
                    f"     Evidence: "
                    f"{quote[:500]}"
                )

    print(
        "\n========================================"
    )

    print(
        f"Total unique sources: "
        f"{total_sources}"
    )

    print(
        f"Total strong sources: "
        f"{total_strong_sources}"
    )


if __name__ == "__main__":
    main()