
from uuid import uuid5, NAMESPACE_URL

from qdrant_client.models import PointStruct

from backend.rag.data_loader import load_knowledge_base
from backend.rag.chunking import create_chunks
from backend.rag.embeddings import generate_document_embeddings

from backend.database.vector_store import (
    create_collection,
    COLLECTION_NAME
)


def ingest():

    df = load_knowledge_base()

    chunks = create_chunks(df)

    embeddings = generate_document_embeddings(chunks)

    client = create_collection()

    points = []

    for chunk, embedding in zip(chunks, embeddings):

        point = PointStruct(
            id=str(uuid5(NAMESPACE_URL, chunk["chunk_id"])),

            vector=embedding.tolist(),

            payload={
                "chunk_id": chunk["chunk_id"],
                "search_text": chunk["search_text"],
                "full_text": chunk["full_text"],
                "metadata": chunk["metadata"]
            }
        )

        points.append(point)

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points,
        wait=True
    )

    print(f"\nSuccessfully indexed {len(points)} chunks.")

    count = client.count(
        collection_name=COLLECTION_NAME,
        exact=True
    )

    print("Total stored points:", count.count)


if __name__ == "__main__":
    ingest()
