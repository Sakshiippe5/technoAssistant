
from backend.rag.data_loader import load_knowledge_base


def create_chunks(df):

    chunks = []

    for _, row in df.iterrows():

        search_text = "\n".join([
            f"Title: {row['title']}",
            f"Category: {row['category']}",
            f"Description: {row['description']}",
            f"Symptoms: {row['symptoms']}",
            f"Error Codes: {row['error_codes']}",
            f"Keywords: {row['keywords']}"
        ])

        full_text = "\n".join([
            f"{column.replace('_', ' ').title()}: {row[column]}"
            for column in df.columns
        ])

        chunk = {
            "chunk_id": row["id"],
            "search_text": search_text,
            "full_text": full_text,
            "metadata": row.to_dict()
        }

        chunks.append(chunk)

    return chunks


if __name__ == "__main__":

    df = load_knowledge_base()
    chunks = create_chunks(df)

    print(f"Total records: {len(df)}")
    print(f"Total chunks: {len(chunks)}")

    print("\nSample Chunk ID:")
    print(chunks[0]["chunk_id"])

    print("\nSearch Text:")
    print(chunks[0]["search_text"])

    print("\nComplete Chunk:")
    print(chunks[0]["full_text"])
    assert len(chunks) == len(df)

    assert len(set(c["chunk_id"] for c in chunks)) == len(chunks)

    assert all(c["search_text"].strip() for c in chunks)

    assert all(c["full_text"].strip() for c in chunks)

    print("Chunk validation successful!")