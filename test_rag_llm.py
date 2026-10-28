"""Interactive smoke check of the same local ticket RAG/LLM path as the app."""

from server import (
    Predictor,
    build_retrieval_query,
    grounded_response,
    retrieve_with_status,
)


def generate_response(question, retrieved_documents, predicted_solution=""):
    """Keep a small helper for manual checks while using the app's LLM path."""
    documents = [
        {
            "source": document.get("source", "knowledge base"),
            "content": document.get("content", document.get("text", "")),
        }
        for document in retrieved_documents
        if isinstance(document, dict)
    ]
    response, _engine, _sources = grounded_response(
        {"subject": question, "description": ""},
        predicted_solution,
        documents,
    )
    return response


def main():
    subject = input("Ticket subject: ").strip()
    description = input("Ticket description: ").strip()
    if not subject and not description:
        raise SystemExit("Enter a ticket subject or description.")

    ticket = {"subject": subject, "description": description}
    ticket_text = f"{subject} {description}".strip()
    solution = Predictor().predict(ticket_text)
    query = build_retrieval_query(ticket, solution)

    print(f"\nPredicted solution: {solution}")
    print("Searching the local knowledge base...")
    documents, retrieval_error = retrieve_with_status(query)

    if documents:
        print("Retrieved documents:")
        for document in documents:
            print(f"- {document['source']} (score: {document['score']:.3f})")
    elif retrieval_error is not None:
        print("Local retrieval failed. See the server message above.")
    else:
        print("No sufficiently relevant document was found.")

    response, engine, sources = grounded_response(
        ticket,
        solution,
        documents,
        retrieval_failed=retrieval_error is not None,
    )
    print(f"\nResponse engine: {engine}")
    print(f"Knowledge sources: {', '.join(sources) if sources else 'none'}")
    print(f"\nCustomer response:\n{response}")


if __name__ == "__main__":
    main()
