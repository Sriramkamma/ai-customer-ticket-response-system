"""Print unfiltered semantic similarity scores for all KB documents."""

from rag import (
    MODEL_NAME,
    build_index,
    load_documents,
    score_documents,
)
from sentence_transformers import SentenceTransformer


QUESTIONS = [
    "The application keeps crashing when I open it. What should I do?",
    "I forgot my password and cannot log in.",
    "I was charged twice for my order. Can I get a refund?",
    "My order is delayed and has not been delivered. How can I track it?",
]


def main():
    documents = load_documents()
    if not documents:
        raise SystemExit("No Markdown documents found in knowledge_base.")

    # Load the model once and reuse it for the index and every query.
    model = SentenceTransformer(MODEL_NAME)
    index = build_index(documents, model)

    for question in QUESTIONS:
        print(f"\nQuestion: {question}")
        for result in score_documents(question, documents, model, index):
            print(f"  {result['source']}: {result['score']:.4f}")


if __name__ == "__main__":
    main()
