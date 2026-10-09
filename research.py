from tools.search import search_web
from agents.writer import generate_answer


def main():
    query = input("\nWhat would you like NEXUS to research?\n> ")

    print("\nSearching the web...\n")

    sources = search_web(query, max_results=5)

    if not sources:
        print("No sources were found.")
        return

    print(f"Found {len(sources)} sources.")

    print("\nGenerating evidence-based answer...\n")

    answer = generate_answer(query, sources)

    print("\n========== NEXUS REPORT ==========\n")
    print(answer)

    print("\n========== SOURCES ==========\n")

    for i, source in enumerate(sources, start=1):
        print(f"[{i}] {source['title']}")
        print(source["url"])
        print()


if __name__ == "__main__":
    main()