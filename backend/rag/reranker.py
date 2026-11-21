
from functools import lru_cache
from sentence_transformers import CrossEncoder

from backend.rag.hybrid_search import hybrid_search


MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@lru_cache(maxsize=1)
def get_reranker():
    return CrossEncoder(MODEL_NAME)


def rerank_documents(query, documents):

    if not documents:
        return []

    model = get_reranker()

    pairs = [
        (query, doc["search_text"])
        for doc in documents
    ]

    scores = model.predict(pairs)

    reranked_results = []

    for doc, score in zip(documents, scores):

        result = doc.copy()

        result["rerank_score"] = float(score)

        reranked_results.append(result)

    reranked_results.sort(
        key=lambda doc: doc["rerank_score"],
        reverse=True
    )

    return reranked_results


if __name__ == "__main__":

    query = input("Enter technical issue: ")

    documents = hybrid_search(query)

    results = rerank_documents(query, documents)

    for rank, doc in enumerate(results, start=1):

        print(f"\nRank: {rank}")
        print("ID:", doc["chunk_id"])
        print("Title:", doc["metadata"]["title"])
        print("RRF Score:", doc["rrf_score"])
        print("Rerank Score:", doc["rerank_score"])
