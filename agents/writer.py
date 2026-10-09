import os

from dotenv import load_dotenv
from groq import Groq


def generate_answer(query: str, sources: list[dict]) -> str:
    """Generate an evidence-based answer using retrieved sources."""

    load_dotenv()

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is missing. Check your .env file."
        )

    client = Groq(api_key=api_key)

    formatted_sources = []

    for i, source in enumerate(sources, start=1):
        formatted_sources.append(
            f"[{i}] {source['title']}\n"
            f"URL: {source['url']}\n"
            f"Content: {source['snippet']}"
        )

    source_text = "\n\n".join(formatted_sources)

    system_prompt = """
You are NEXUS, an evidence-based AI research assistant.

Your job is to answer the user's research question using ONLY
the provided sources.

Citation rules:

1. Every factual claim must be supported by at least one source.
2. Place the citation immediately after the claim it supports.
3. Use the source number exactly as provided: [1], [2], [3], etc.
4. Use multiple citations when multiple sources support a claim.
5. Do not repeatedly cite one source when another provided source
   directly supports the same claim.
6. Do not cite a source for information that does not appear in it.
7. Do not invent facts, statistics, papers, benchmarks, or results.
8. If the sources do not provide enough evidence, explicitly state:
   "The retrieved sources do not provide enough evidence to determine this."
9. Distinguish established findings from recommendations or interpretation.
10. Prefer primary research papers and official documentation when available.

Answer structure:

- Brief introduction
- Main findings
- Comparison or analysis
- Limitations / uncertainty
- Conclusion

Keep the answer concise but sufficiently detailed.
"""

    user_prompt = f"""
Research question:

{query}

Retrieved sources:

{source_text}

Write a rigorous evidence-based research answer.

Use citations throughout the answer.
Do not place all citations at the end.
Each citation should correspond directly to the claim it supports.

Before answering, internally determine which source supports each
important factual claim.
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
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
        temperature=0.2,
        max_tokens=1000,
    )

    return response.choices[0].message.content