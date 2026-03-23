"""
Evaluation runner: tests the chatbot against hardcoded questions and logs results.

Usage:
    python -m eval.run_eval

Each question runs in its own fresh conversation (no cross-contamination between
questions). Results are printed to stdout and saved to eval_results/.

This is a minimal eval framework. Production eval systems add:
  - LLM-as-judge scoring (send response + rubric to another LLM for a 1-5 grade)
  - Retrieval metrics (precision@k, recall@k — did the right chunks get retrieved?)
  - Latency tracking (how long did each response take?)
  - Regression tracking (compare scores across code changes)
But for learning, keyword matching + manual review is a good starting point.
"""

import json
import os
import sys
from datetime import datetime

# Add the project root to sys.path so we can import our modules.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from conversation.engine import chat
from conversation.history import create_conversation
from eval.questions import EVAL_QUESTIONS


def check_keywords(response: str, keywords: list[str]) -> dict:
    """
    Check which expected keywords appear in the response.

    Returns a dict with found/missing keywords and whether all were found.
    Case-insensitive matching — we care about concepts, not exact casing.
    """
    response_lower = response.lower()
    found = [kw for kw in keywords if kw.lower() in response_lower]
    missing = [kw for kw in keywords if kw.lower() not in response_lower]
    return {
        "found": found,
        "missing": missing,
        "all_found": len(missing) == 0,
    }


def run_eval():
    """Run all evaluation questions and report results."""
    results = []
    total_pass = 0

    print("=" * 70)
    print("CODEBOT EVALUATION")
    print("=" * 70)

    for i, qa in enumerate(EVAL_QUESTIONS, 1):
        question = qa["question"]
        expected = qa["expected_keywords"]

        print(f"\n--- Question {i}/{len(EVAL_QUESTIONS)} ---")
        print(f"Q: {question}")

        # Each question gets a fresh conversation to avoid context leaking
        # between questions. In a real eval, you might also test multi-turn
        # scenarios (ask a follow-up that depends on the previous answer).
        conv_id = create_conversation()

        try:
            response = chat(conv_id, question)
        except Exception as e:
            response = f"[ERROR] {e}"

        keyword_check = check_keywords(response, expected)
        passed = keyword_check["all_found"]

        if passed:
            total_pass += 1

        print(f"A: {response[:200]}{'...' if len(response) > 200 else ''}")
        print(f"Keywords found: {keyword_check['found']}")
        print(f"Keywords missing: {keyword_check['missing']}")
        print(f"Result: {'PASS' if passed else 'FAIL'}")

        results.append({
            "question": question,
            "response": response,
            "expected_keywords": expected,
            "keywords_found": keyword_check["found"],
            "keywords_missing": keyword_check["missing"],
            "passed": passed,
        })

    # Summary
    print("\n" + "=" * 70)
    print(f"RESULTS: {total_pass}/{len(EVAL_QUESTIONS)} passed")
    print("=" * 70)

    # Save detailed results to a JSON file for later review.
    os.makedirs("eval_results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"eval_results/eval_{timestamp}.json"

    with open(output_path, "w") as f:
        json.dump(
            {
                "timestamp": timestamp,
                "total": len(EVAL_QUESTIONS),
                "passed": total_pass,
                "results": results,
            },
            f,
            indent=2,
        )

    print(f"Detailed results saved to {output_path}")
    return results


if __name__ == "__main__":
    run_eval()
