"""
llm.py
------
Wraps the local Ollama API for answer generation.
Takes a question and a list of retrieved chunks, formats them into
a prompt, and returns the model's answer.
"""

import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "mistral"

PROMPT_TEMPLATE = """You are a helpful assistant answering questions about endurance training and triathlon.
Use only the context below to answer the question. If the context doesn't contain enough information, say so honestly.

Context:
{context}

Question: {question}

Answer:"""


def build_prompt(question: str, chunks: list[dict]) -> str:
    """Assemble retrieved chunks and the question into a single prompt."""
    context = "\n\n---\n\n".join(
        f"[Source: {chunk['source']}]\n{chunk['text']}" for chunk in chunks
    )
    return PROMPT_TEMPLATE.format(context=context, question=question)


def generate_answer(
    question: str, chunks: list[dict], model: str = DEFAULT_MODEL
) -> str:
    """Send the prompt to Ollama and return the model's answer."""
    prompt = build_prompt(question, chunks)

    response = requests.post(
        OLLAMA_URL,
        json={"model": model, "prompt": prompt, "stream": False},
    )
    response.raise_for_status()

    return response.json()["response"]


if __name__ == "__main__":
    from retrieval import retrieve

    question = "What is zone 2 training?"
    chunks = retrieve(question, top_k=3)
    answer = generate_answer(question, chunks)

    print(f"Question: {question}\n")
    print(f"Answer:\n{answer}\n")