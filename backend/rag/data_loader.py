
import pandas as pd
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[2]
KB_PATH = BASE_DIR / "data" / "knowledge_base.csv"


def load_knowledge_base():
    df = pd.read_csv(
        KB_PATH,
        dtype=str,
        keep_default_na=False
    )

    required_columns = [
        "id",
        "title",
        "category",
        "description",
        "symptoms",
        "error_codes",
        "keywords",
        "possible_causes",
        "diagnostic_checks",
        "known_resolutions",
        "escalation_team",
        "escalation_conditions",
        "required_information"
    ]

    missing_columns = set(required_columns) - set(df.columns)

    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")

    for col in required_columns:
        df[col] = df[col].str.strip()

    mandatory_fields = [
        "id",
        "title",
        "category",
        "symptoms",
        "diagnostic_checks",
        "known_resolutions",
        "escalation_team"
    ]

    for col in mandatory_fields:
        if df[col].eq("").any():
            raise ValueError(f"Empty values found in {col}")

    if df["id"].duplicated().any():
        raise ValueError("Duplicate knowledge base IDs found")

    return df


def prepare_search_text(row):
    return "\n".join([
        f"Title: {row['title']}",
        f"Category: {row['category']}",
        f"Description: {row['description']}",
        f"Symptoms: {row['symptoms']}",
        f"Error Codes: {row['error_codes']}",
        f"Keywords: {row['keywords']}"
    ])


def prepare_documents(df):
    documents = []

    for _, row in df.iterrows():
        document = {
            "id": row["id"],
            "search_text": prepare_search_text(row),
            "metadata": row.to_dict()
        }

        documents.append(document)

    return documents


if __name__ == "__main__":

    df = load_knowledge_base()

    print("Knowledge base loaded successfully!")
    print("Dataset shape:", df.shape)

    print("\nCategory distribution:")
    print(df["category"].value_counts())

    documents = prepare_documents(df)

    print("\nTotal documents:", len(documents))

    print("\nSample document ID:")
    print(documents[0]["id"])

    print("\nSample searchable text:")
    print(documents[0]["search_text"])
