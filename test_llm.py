import os

from dotenv import load_dotenv
from groq import Groq


def main():
    # Load environment variables from .env
    load_dotenv()

    # Get API key
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is missing. "
            "Check your .env file."
        )

    # Create Groq client
    client = Groq(api_key=api_key)

    # Send request to the LLM
    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are NEXUS, an AI research assistant "
                    "focused on evidence-based research."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Introduce yourself and explain what "
                    "NEXUS is going to become."
                ),
            },
        ],
        temperature=0.2,
        max_tokens=200,
    )

    # Extract response
    answer = response.choices[0].message.content

    print("\n========== NEXUS ==========\n")
    print(answer)
    print("\n====== CONNECTION OK ======\n")


if __name__ == "__main__":
    main()