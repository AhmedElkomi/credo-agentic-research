from agents.planner import create_research_plan


def main():
    query = input("\nResearch question:\n> ")

    questions = create_research_plan(query)

    print("\n========== CREDO RESEARCH PLAN ==========\n")

    for i, question in enumerate(questions, start=1):
        print(f"{i}. {question}")


if __name__ == "__main__":
    main()