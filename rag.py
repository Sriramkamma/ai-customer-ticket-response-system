
from pathlib import Path
import json
import urllib.error
import urllib.request
from urllib.parse import urlparse

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# Find the project folder and knowledge-base folder
BASE_DIR = Path(__file__).resolve().parent
KB_DIR = BASE_DIR / "knowledge_base"

# Local embedding model
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Search settings
TOP_K = 2
# Normalized embeddings plus inner product produce cosine similarity. A score
# of 0.35 is a conservative floor for this small, general-purpose knowledge base.
MIN_SCORE = 0.35
# When several files have weak overlap, keep only results close to the best
# match. This avoids adding a merely top-k neighbor to the LLM context.
MIN_RELATIVE_SCORE = 0.70


def load_documents():
    """Read all Markdown knowledge-base files."""
    documents = []

    for file_path in sorted(KB_DIR.glob("*.md")):
        text = file_path.read_text(encoding="utf-8").strip()

        if text:
            documents.append({
                "source": file_path.name,
                "text": text,
            })

    return documents


def build_index(documents, model):
    """Convert documents into embeddings and build a FAISS index."""
    texts = [doc["text"] for doc in documents]

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.ndim != 2 or embeddings.shape[0] != len(documents):
        raise ValueError("The embedding model returned an invalid document matrix.")

    # Inner product on normalized vectors gives cosine similarity
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    return index


def score_documents(question, documents, model, index):
    """Return every document ranked by cosine similarity, without filtering."""
    if not isinstance(question, str) or not question.strip() or not documents:
        return []

    query_vector = model.encode(
        [question], convert_to_numpy=True, normalize_embeddings=True
    )
    query_vector = np.asarray(query_vector, dtype=np.float32)
    if query_vector.ndim != 2 or query_vector.shape[1] != index.d:
        raise ValueError("Query and document embedding dimensions do not match.")

    scores, positions = index.search(query_vector, len(documents))
    return [
        {
            "source": documents[int(position)]["source"],
            "text": documents[int(position)]["text"],
            "score": float(score),
        }
        for score, position in zip(scores[0], positions[0])
        if 0 <= int(position) < len(documents)
    ]


def search_knowledge_base(
    question,
    documents,
    model,
    index,
    top_k=TOP_K,
    min_score=MIN_SCORE,
):
    """Return documents whose similarity meets the threshold."""
    if not isinstance(question, str) or not question.strip() or not documents:
        return []
    if index.ntotal != len(documents):
        raise ValueError("FAISS index and document mapping have different sizes.")
    ranked = score_documents(question, documents, model, index)
    limit = min(max(0, int(top_k)), len(ranked))
    candidates = [result for result in ranked[:limit] if result["score"] >= min_score]
    if not candidates:
        return []
    relative_floor = candidates[0]["score"] * MIN_RELATIVE_SCORE
    return [result for result in candidates if result["score"] >= relative_floor]


if __name__ == "__main__":
    print("Loading knowledge-base documents...")

    documents = load_documents()

    if not documents:
        raise SystemExit(
            "No Markdown documents found in the knowledge_base folder."
        )

    print(f"Loaded {len(documents)} documents.")

    print("Loading the embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    print("Building the FAISS index...")
    index = build_index(documents, model)

    question = input("\nEnter a customer question: ")

    results = search_knowledge_base(
        question=question,
        documents=documents,
        model=model,
        index=index,
    )

    print("\nMost relevant knowledge-base results:")

    if not results:
        print(
            "No sufficiently relevant documents found. "
            "Please review the question or contact support."
        )
    else:
        for number, result in enumerate(results, start=1):
            print(f"\nResult {number}")
            print(f"Source: {result['source']}")
            print(f"Similarity score: {result['score']:.3f}")
            print(f"Content: {result['text']}")


def grounded_response(ticket, solution, docs, retrieval_failed=False, *, llm_url=None, llm_model=None):
    """Generate a response grounded in retrieved knowledge."""

    guidance = solution.strip() if isinstance(solution, str) else ""

    relevant_docs = [
        document
        for document in docs
        if isinstance(document, dict)
        and isinstance(document.get("content"), str)
        and document["content"].strip()
    ]

    sources = [
        document.get("source", "knowledge base")
        for document in relevant_docs
    ]

    if not relevant_docs:
        if retrieval_failed:
            return (
                "We're sorry you're experiencing this issue. "
                "I couldn't access the relevant help articles right now. "
                "Please contact our support team for help.",
                "retrieval_unavailable_fallback",
                [],
            )
        return (
            "We're sorry you're experiencing this issue. "
            "I couldn't find enough information in our help articles "
            "to answer safely. Please contact our support team for help.",
            "no_knowledge_fallback",
            [],
        )

    excerpts = "\n\n".join(
        f"Source: {document['source']}\n{document['content']}"
        for document in relevant_docs
    )

    prompt = f"""
You are a customer-support assistant.

Write a concise, polite, helpful response to the customer.

Strict rules:
- Use only the relevant knowledge-base content provided below. The suggested
  solution is a candidate hint; include it only when the content supports it.
- Do not invent policies, actions, outcomes, refund approvals, or dates.
- Do not claim that you checked an order or performed an action.
- Do not add claims about what usually works, likely causes, or expected results.
- Do not promise personal follow-up or actions that the article does not offer.
- Present the listed steps directly; do not say they are common, typical,
  likely to work, or may resolve the problem.
- Do not introduce unrelated topics.
- Follow any conditions in the knowledge base precisely.
- If tracking says delivered, advise checking nearby delivery locations.
  Do not suggest this as the explanation when tracking has not said delivered.
- If information is insufficient, recommend contacting support.
- Do not repeat the knowledge-base text verbatim.
- Return only the customer-facing response.

Ticket subject:
{ticket.get("subject", "")}

Customer description:
{ticket.get("description", "")}

Suggested solution:
{guidance}

Relevant knowledge-base content:
{excerpts}

Customer response:
""".strip()

    try:
        endpoint = urlparse(llm_url or "http://127.0.0.1:11434/api/generate")

        # Enforce local-only Ollama requests.
        if (
            endpoint.scheme != "http"
            or endpoint.hostname not in (
                "127.0.0.1",
                "localhost",
                "::1",
            )
        ):
            raise ValueError(
                "Ollama endpoint must use local loopback HTTP."
            )

        payload = json.dumps({
            "model": llm_model or "llama3.2:3b",
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1},
        }).encode("utf-8")

        request = urllib.request.Request(
            llm_url or "http://127.0.0.1:11434/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urllib.request.urlopen(
            request,
            timeout=180,
        ) as response:
            result = json.loads(
                response.read().decode("utf-8")
            )
        if not isinstance(result, dict):
            raise ValueError("Ollama returned an invalid response object.")

        generated = result.get("response", "")
        generated = generated.strip() if isinstance(generated, str) else ""

        if generated:
            return generated, "ollama", sources

    except (
        OSError,
        ValueError,
        urllib.error.URLError,
        json.JSONDecodeError,
    ) as error:
        print(f"Local Ollama response unavailable: {error}")

    # Do not echo an ML prediction as customer guidance unless the LLM can
    # ground it. The local model failure is explicit and distinct from retrieval.
    return (
        "We're sorry you're experiencing this issue. "
        "I couldn't generate a detailed response right now. "
        "Please contact our support team for help.",
        "ollama_unavailable_fallback",
        sources,
    )

