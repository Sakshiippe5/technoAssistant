
from backend.rag.query_decomposer import decompose_query
from backend.rag.context_selector import select_context


def retrieve_multiple_issues(query: str):

    decomposition = decompose_query(query)

    if decomposition["requires_clarification"]:

        return {
            "original_query": query,
            "status": "clarification_required",
            "clarification_question": decomposition[
                "clarification_question"
            ],
            "issues": [],
            "total_unique_chunks": 0
        }

    issue_results = []

    unique_chunk_ids = set()

    for issue in decomposition["issues"]:

        result = select_context(
            issue,
            max_chunks=3
        )

        issue_results.append({
            "issue": issue,
            "status": result["status"],
            "documents": result["documents"],
            "selected_count": result["selected_count"]
        })

        for doc in result["documents"]:

            unique_chunk_ids.add(
                doc["chunk_id"]
            )

    supported_count = sum(
        result["status"] == "supported"
        for result in issue_results
    )

    if supported_count == len(issue_results):
        status = "supported"

    elif supported_count == 0:
        status = "insufficient_context"

    else:
        status = "partially_supported"

    return {
        "original_query": query,
        "status": status,
        "total_issues": len(issue_results),
        "supported_issues": supported_count,
        "unsupported_issues": (
            len(issue_results) - supported_count
        ),
        "total_unique_chunks": len(unique_chunk_ids),
        "issues": issue_results
    }


if __name__ == "__main__":

    query = input("Enter technical issues: ")

    result = retrieve_multiple_issues(query)

    print("\nOverall Status:", result["status"])

    if result["status"] == "clarification_required":

        print(result["clarification_question"])

    else:

        print("Total Issues:", result["total_issues"])

        print(
            "Supported Issues:",
            result["supported_issues"]
        )

        print(
            "Unsupported Issues:",
            result["unsupported_issues"]
        )

        print(
            "Unique Chunks:",
            result["total_unique_chunks"]
        )

        for i, issue in enumerate(
            result["issues"], start=1
        ):

            print(f"\n--- Issue {i} ---")

            print("Query:", issue["issue"])

            print("Status:", issue["status"])

            for doc in issue["documents"]:

                print(
                    doc["chunk_id"],
                    doc["metadata"]["title"],
                    round(doc["rerank_score"], 4)
                )

            if not issue["documents"]:

                print(
                    "Insufficient knowledge base evidence."
                )
