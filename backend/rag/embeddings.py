
import numpy as np

from sentence_transformers import SentenceTransformer
from backend.rag.data_loader import load_knowledge_base
from backend.rag.chunking import create_chunks


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

model = SentenceTransformer(MODEL_NAME)


def validate_token_lengths(texts):

    max_tokens = model.max_seq_length

    for i, text in enumerate(texts):

        tokens = model.tokenizer.encode(
            text,
            truncation=False
        )

        if len(tokens) > max_tokens:
            raise ValueError(
                f"Chunk {i + 1} exceeds the "
                f"{max_tokens} token limit."
            )


def generate_document_embeddings(chunks):

    texts = [
        chunk["search_text"]
        for chunk in chunks
    ]

    validate_token_lengths(texts)

    embeddings = model.encode(
        texts,
        batch_size=16,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True
    )

    return embeddings


def generate_query_embedding(query):

    if not query.strip():
        raise ValueError("Query cannot be empty.")

    validate_token_lengths([query])

    embedding = model.encode(
        query,
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    return embedding


if __name__ == "__main__":

    df = load_knowledge_base()

    chunks = create_chunks(df)

    embeddings = generate_document_embeddings(chunks)

    print("\nTotal chunks:", len(chunks))

    print("Embeddings shape:", embeddings.shape)

    print("Embedding dimensions:", embeddings.shape[1])

    print("\nFirst 10 values:")
    print(embeddings[0][:10])

    print("\nVector norm:")
    print(np.linalg.norm(embeddings[0]))

    query = "Payment API is returning HTTP 500"

    query_embedding = generate_query_embedding(query)

    print("\nQuery embedding shape:")
    print(query_embedding.shape)

    similarities = np.dot(
        embeddings,
        query_embedding
    )

    top_indices = np.argsort(similarities)[::-1][:3]

    print("\nTop 3 relevant documents:")

    for index in top_indices:

        print(
            chunks[index]["chunk_id"],
            chunks[index]["metadata"]["title"],
            round(float(similarities[index]), 4)
        )
