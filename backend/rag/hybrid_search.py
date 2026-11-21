
from backend.rag.dense_retriever import dense_search
from backend.rag.sparse_retriever import sparse_search
from backend.rag.rrf import reciprocal_rank_fusion


def hybrid_search(
    query: str,
    dense_top_k: int = 10,
    sparse_top_k: int = 10,
    final_top_k: int = 10
):

    if not query.strip():
        raise ValueError("Query cannot be empty.")

    dense_results = dense_search(
        query,
        top_k=dense_top_k
    )

    sparse_results = sparse_search(
        query,
        top_k=sparse_top_k
    )

    fused_results = reciprocal_rank_fusion(
        dense_results=dense_results,
        sparse_results=sparse_results,
        k=60,
        top_k=final_top_k
    )

    return fused_results


if __name__ == "__main__":

    query = input("Enter technical issue: ")

    results = hybrid_search(
        query,
        final_top_k=5
    )

    for rank, result in enumerate(results, start=1):

        print(f"\nRank: {rank}")
        print("ID:", result["chunk_id"])
        print("Title:", result["metadata"]["title"])
        print("Dense Score:", result.get("dense_score", "N/A"))
        print("Sparse Score:", result.get("sparse_score", "N/A"))
        print("RRF Score:", round(result["rrf_score"], 6))
