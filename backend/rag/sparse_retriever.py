
import re
import numpy as np

from functools import lru_cache
from rank_bm25 import BM25Okapi

from backend.rag.data_loader import load_knowledge_base
from backend.rag.chunking import create_chunks


def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())


@lru_cache(maxsize=1)
def initialize_bm25():

    df = load_knowledge_base()
    chunks = create_chunks(df)

    tokenized_corpus = [
        tokenize(chunk["search_text"])
        for chunk in chunks
    ]

    bm25 = BM25Okapi(tokenized_corpus)

    return bm25, chunks


def sparse_search(query: str, top_k: int = 10):

    bm25, chunks = initialize_bm25()

    tokenized_query = tokenize(query)

    scores = bm25.get_scores(tokenized_query)

    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []

    for index in top_indices:

        if scores[index] <= 0:
            continue

        chunk = chunks[index]

        results.append({
            "chunk_id": chunk["chunk_id"],
            "search_text": chunk["search_text"],
            "full_text": chunk["full_text"],
            "metadata": chunk["metadata"],
            "sparse_score": float(scores[index])
        })

    return results


if __name__ == "__main__":

    query = input("Enter technical issue: ")

    results = sparse_search(query, top_k=3)

    for rank, result in enumerate(results, start=1):

        print(f"\nRank: {rank}")
        print("ID:", result["chunk_id"])
        print("Title:", result["metadata"]["title"])
        print("BM25 Score:", round(result["sparse_score"], 4))
