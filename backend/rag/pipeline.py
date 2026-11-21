
import json
import time
import logging

from groq import APIError

from backend.rag.multi_issue_retriever import (
    retrieve_multiple_issues
)

from backend.rag.generator import generate_answer


logger = logging.getLogger(__name__)


def run_pipeline(query: str):

    start_time = time.perf_counter()

    query = query.strip()

    if not query:
        raise ValueError("Query cannot be empty.")

    retrieval = retrieve_multiple_issues(query)

    if retrieval["status"] == "clarification_required":

        return {
            "query": query,
            "status": "clarification_required",
            "message": retrieval[
                "clarification_question"
            ],
            "total_issues": 0,
            "supported_issues": 0,
            "issues": [],
            "latency_seconds": round(
                time.perf_counter() - start_time, 2
            )
        }

    answers = []

    for item in retrieval["issues"]:

        issue = item["issue"]
        documents = item["documents"]

        # Capture all selected documents for evaluation
        retrieval_context = [
            doc["full_text"]
            for doc in documents
        ]

        retrieved_ids = [
            doc["chunk_id"]
            for doc in documents
        ]

        # Do not call the LLM for unsupported issues
        if not documents:

            answers.append({
                "issue": issue,
                "status": "insufficient_context",
                "message": (
                    "The knowledge base does not "
                    "contain sufficient information "
                    "to troubleshoot this issue."
                ),
                "answer": None,
                "sources": [],
                "retrieval_context": [],
                "retrieved_ids": []
            })

            continue

        try:

            answer = generate_answer(
                issue,
                documents
            )

            cited_ids = set(answer["source_ids"])

            sources = [
                {
                    "id": doc["chunk_id"],
                    "title": doc["metadata"]["title"],
                    "category": doc["metadata"]["category"],
                    "relevance_score": doc["rerank_score"],
                    "content": doc["metadata"]
                }
                for doc in documents
                if doc["chunk_id"] in cited_ids
            ]

            answers.append({
                "issue": issue,
                "status": "supported",
                "answer": answer,
                "sources": sources,

                # DeepEval evaluation fields
                "retrieval_context": retrieval_context,
                "retrieved_ids": retrieved_ids
            })

        except (ValueError, APIError):

            logger.exception("Answer generation failed")

            answers.append({
                "issue": issue,
                "status": "generation_failed",
                "message": (
                    "Unable to generate a validated "
                    "troubleshooting response."
                ),
                "answer": None,
                "sources": [],

                # Preserve retrieval results even
                # when answer generation fails
                "retrieval_context": retrieval_context,
                "retrieved_ids": retrieved_ids
            })

    supported_count = sum(
        item["status"] == "supported"
        for item in answers
    )

    failed_count = sum(
        item["status"] == "generation_failed"
        for item in answers
    )

    if answers and supported_count == len(answers):
        status = "supported"

    elif supported_count > 0:
        status = "partially_supported"

    elif failed_count > 0:
        status = "generation_failed"

    else:
        status = "insufficient_context"

    return {
        "query": query,
        "status": status,
        "total_issues": len(answers),
        "supported_issues": supported_count,
        "issues": answers,
        "latency_seconds": round(
            time.perf_counter() - start_time, 2
        )
    }


if __name__ == "__main__":

    query = input("Enter technical issue: ")

    response = run_pipeline(query)

    print(
        json.dumps(
            response,
            indent=4,
            ensure_ascii=False
        )
    )
