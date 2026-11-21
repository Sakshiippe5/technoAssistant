
from backend.rag.embeddings import generate_query_embedding

from backend.database.vector_store import (
    get_client,
    COLLECTION_NAME
)


def dense_search(query: str, top_k: int = 10):

    client = get_client()

    query_embedding = generate_query_embedding(query)

    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding.tolist(),
        limit=top_k,
        with_payload=True
    )

    results = []

    for point in response.points:

        results.append({
            "chunk_id": point.payload["chunk_id"],
            "search_text": point.payload["search_text"],
            "full_text": point.payload["full_text"],
            "metadata": point.payload["metadata"],
            "dense_score": float(point.score)
        })

    return results


if __name__ == "__main__":

    query = input("Enter technical issue: ")

    results = dense_search(query, top_k=3)

    for rank, result in enumerate(results, start=1):

        print(f"\nRank: {rank}")
        print("ID:", result["chunk_id"])
        print("Title:", result["metadata"]["title"])
        print("Score:", round(result["dense_score"], 4))
