
def reciprocal_rank_fusion(
    dense_results,
    sparse_results,
    k=60,
    top_k=10
):

    fused_scores = {}
    documents = {}

    for results in [dense_results, sparse_results]:

        for rank, document in enumerate(results, start=1):

            chunk_id = document["chunk_id"]

            if chunk_id not in fused_scores:
                fused_scores[chunk_id] = 0.0

            fused_scores[chunk_id] += 1 / (k + rank)

            if chunk_id not in documents:
                documents[chunk_id] = document.copy()
            else:
                documents[chunk_id].update(document)

    sorted_ids = sorted(
        fused_scores,
        key=fused_scores.get,
        reverse=True
    )[:top_k]

    fused_results = []

    for chunk_id in sorted_ids:

        document = documents[chunk_id].copy()

        document["rrf_score"] = fused_scores[chunk_id]

        fused_results.append(document)

    return fused_results
