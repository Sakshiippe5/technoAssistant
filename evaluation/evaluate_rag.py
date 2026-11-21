
import os
import json

from pathlib import Path
from statistics import mean

os.environ["DEEPEVAL_TELEMETRY_OPT_OUT"] = "YES"

from deepeval.test_case import LLMTestCase

from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric
)

from backend.rag.pipeline import run_pipeline
from evaluation.judge import GroqJudge


BASE_DIR = Path(__file__).resolve().parent

DATASET_PATH = BASE_DIR / "dataset.json"

RESULTS_PATH = BASE_DIR / "results.json"


def load_dataset():

    with open(
        DATASET_PATH,
        encoding="utf-8"
    ) as file:

        return json.load(file)


def create_metrics(judge):

    settings = {
        "model": judge,
        "include_reason": True,
        "async_mode": False
    }

    return [
        FaithfulnessMetric(
            threshold=0.9,
            **settings
        ),

        AnswerRelevancyMetric(
            threshold=0.8,
            **settings
        ),

        ContextualPrecisionMetric(
            threshold=0.8,
            **settings
        ),

        ContextualRecallMetric(
            threshold=0.8,
            **settings
        ),

        ContextualRelevancyMetric(
            threshold=0.7,
            **settings
        )
    ]


def format_answer(answer):

    lines = [answer["summary"]]

    for section in [
        "possible_causes",
        "diagnostic_steps",
        "required_information"
    ]:

        for item in answer.get(section, []):

            lines.append(item["action"])

    for item in answer.get("resolutions", []):

        lines.append(
            f"If {item['condition']}: "
            f"{item['action']}"
        )

    for item in answer.get("escalation", []):

        lines.append(
            f"Escalation: {item['team']} "
            f"when {item['condition']}"
        )

    return "\n".join(lines)


def evaluate_case(case, judge):

    result = run_pipeline(case["input"])

    actual_status = result["status"]

    items = result.get("issues", [])

    retrieved_ids = list(dict.fromkeys(
        source_id
        for item in items
        for source_id in item.get(
            "retrieved_ids", []
        )
    ))

    expected_ids = set(
        case["expected_source_ids"]
    )

    actual_ids = set(retrieved_ids)

    record = {
        "id": case["id"],
        "input": case["input"],
        "expected_status": case["expected_status"],
        "actual_status": actual_status,
        "status_passed": (
            actual_status == case["expected_status"]
        ),
        "expected_source_ids": sorted(expected_ids),
        "retrieved_ids": retrieved_ids,
        "source_recall": (
            len(expected_ids & actual_ids)
            / len(expected_ids)
            if expected_ids else None
        ),
        "latency_seconds": result["latency_seconds"],
        "metrics": {},
        "pipeline_output": result
    }

    print("\n" + "=" * 60)

    print("Test:", case["id"])
    print("Query:", case["input"])
    print("Expected status:", case["expected_status"])
    print("Actual status:", actual_status)
    print("Retrieved IDs:", retrieved_ids)

    # Edge cases are evaluated separately
    # using expected behavior.

    if case["expected_status"] != "supported":

        record["metric_skip_reason"] = (
            "Behavioral edge case"
        )

        return record

    # Initial semantic evaluation supports
    # single-issue queries.

    if len(items) != 1:

        record["metric_skip_reason"] = (
            "Expected one issue"
        )

        return record

    item = items[0]

    if not item.get("answer"):

        record["metric_skip_reason"] = (
            "No generated answer"
        )

        return record

    retrieval_context = item.get(
        "retrieval_context", []
    )

    if not retrieval_context:

        record["metric_skip_reason"] = (
            "No retrieval context captured"
        )

        return record

    test_case = LLMTestCase(

        input=case["input"],

        actual_output=format_answer(
            item["answer"]
        ),

        expected_output=case["expected_output"],

        retrieval_context=retrieval_context

    )

    for metric in create_metrics(judge):

        metric_name = type(metric).__name__

        try:

            metric.measure(test_case)

            record["metrics"][metric_name] = {
                "score": metric.score,
                "threshold": metric.threshold,
                "passed": metric.is_successful(),
                "reason": metric.reason
            }

            print(
                f"{metric_name}: "
                f"{metric.score:.3f}"
            )

        except Exception as error:

            record["metrics"][metric_name] = {
                "error": str(error)
            }

            print(
                f"{metric_name} failed: {error}"
            )

    return record


def print_summary(records):

    print("\n" + "=" * 60)

    print("EVALUATION SUMMARY")

    print("=" * 60)

    successful_cases = [
        record
        for record in records
        if "error" not in record
    ]

    status_passes = sum(
        record["status_passed"]
        for record in successful_cases
    )

    print(
        f"Status accuracy: "
        f"{status_passes}/{len(records)}"
    )

    metric_names = [
        "FaithfulnessMetric",
        "AnswerRelevancyMetric",
        "ContextualPrecisionMetric",
        "ContextualRecallMetric",
        "ContextualRelevancyMetric"
    ]

    for name in metric_names:

        scores = [
            record["metrics"][name]["score"]

            for record in successful_cases

            if name in record["metrics"]

            and "score" in record["metrics"][name]

            and record["metrics"][name]["score"]
            is not None
        ]

        if scores:

            print(
                f"{name}: {mean(scores):.3f} "
                f"(n={len(scores)})"
            )

        else:

            print(f"{name}: Not evaluated")

    latencies = [
        record["latency_seconds"]
        for record in successful_cases
    ]

    if latencies:

        print(
            f"Average pipeline latency: "
            f"{mean(latencies):.2f}s"
        )


def run_evaluation():

    dataset = load_dataset()

    judge = GroqJudge()

    records = []

    for case in dataset:

        try:

            record = evaluate_case(case, judge)

        except Exception as error:

            record = {
                "id": case["id"],
                "input": case["input"],
                "error": str(error)
            }

            print(
                f"Test {case['id']} failed: {error}"
            )

        records.append(record)

        # Save after every test case

        with open(
            RESULTS_PATH,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                records,
                file,
                indent=4,
                ensure_ascii=False
            )

    print_summary(records)

    print("\nResults saved to:", RESULTS_PATH)


if __name__ == "__main__":

    run_evaluation()
