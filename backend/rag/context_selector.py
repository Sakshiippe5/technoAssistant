
from backend.rag.hybrid_search import hybrid_search
from backend.rag.reranker import rerank_documents

from backend.rag.relevance_filter import (
    filter_relevant_documents
)


def select_context(query, max_chunks=3):

    candidates = hybrid_search(
        query,
        dense_top_k=10,
        sparse_top_k=10,
        final_top_k=10
    )

    reranked_documents = rerank_documents(
        query,
        candidates
    )

    relevant_documents = filter_relevant_documents(
        query,
        reranked_documents
    )

    selected_documents = relevant_documents[:max_chunks]

    return {
        "query": query,
        "status": (
            "supported"
            if selected_documents
            else "insufficient_context"
        ),
        "retrieved_count": len(candidates),
        "relevant_count": len(relevant_documents),
        "selected_count": len(selected_documents),
        "documents": selected_documents
    }


if __name__ == "__main__":

    query = input("Enter technical issue: ")

    result = select_context(query)

    print("\nStatus:", result["status"])
    print("Retrieved:", result["retrieved_count"])
    print("Relevant:", result["relevant_count"])
    print("Selected:", result["selected_count"])

    for rank, doc in enumerate(
        result["documents"], start=1
    ):

        print(f"\nRank: {rank}")
        print("ID:", doc["chunk_id"])
        print("Title:", doc["metadata"]["title"])
        print("Rerank Score:", doc["rerank_score"])

    if not result["documents"]:

        print(
            "\nInsufficient information in "
            "the knowledge base."
        )
