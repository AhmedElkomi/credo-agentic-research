from tools.search import search_web


def main():
    query = "latest RAG evaluation techniques for LLMs"

    sources = search_web(query, max_results=5)

    print("\n========== SEARCH RESULTS ==========\n")

    for i, source in enumerate(sources, start=1):
        print(f"[{i}] {source['title']}")
        print(source["url"])
        print(source["snippet"][:500])
        print("-" * 80)


if __name__ == "__main__":
    main()