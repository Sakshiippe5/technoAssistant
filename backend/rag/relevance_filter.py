
import re


# Initial threshold: must be calibrated during evaluation
MIN_RERANK_SCORE = 0.0


HTTP_CODE_PATTERN = (
    r"\b(?:http(?:\s+status)?|status(?:\s+code)?|"
    r"error(?:\s+code)?|code)"
    r"\s*[:=-]?\s*([1-5]\d{2})\b"
)

STANDALONE_CODE_PATTERN = r"^\s*([1-5]\d{2})\s*$"


def extract_query_error_codes(query):

    codes = set(
        re.findall(
            HTTP_CODE_PATTERN,
            query,
            flags=re.IGNORECASE
        )
    )

    standalone = re.fullmatch(
        STANDALONE_CODE_PATTERN,
        query
    )

    if standalone:
        codes.add(standalone.group(1))

    return codes


def extract_document_error_codes(document):

    metadata = document["metadata"]

    text = " ".join([
        metadata.get("error_codes", ""),
        metadata.get("title", "")
    ])

    return set(
        re.findall(r"\b[1-5]\d{2}\b", text)
    )


def validate_error_codes(query, document):

    query_codes = extract_query_error_codes(query)

    if not query_codes:
        return True

    document_codes = extract_document_error_codes(
        document
    )

    return query_codes.issubset(document_codes)


def filter_relevant_documents(
    query,
    documents,
    min_score=MIN_RERANK_SCORE
):

    relevant_documents = []

    for doc in documents:

        if not validate_error_codes(query, doc):
            continue

        if doc["rerank_score"] < min_score:
            continue

        relevant_documents.append(doc)

    return relevant_documents
