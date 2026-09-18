"""Optional RAGAS 0.4.3 metrics. No truncation or reference-context fallback."""
import asyncio
import copy
import math
import os


def _build_judge():
    from openai import AsyncOpenAI
    from ragas.llms import llm_factory
    from ragas.embeddings import embedding_factory
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY is required for the optional RAGAS runner.")
    client = AsyncOpenAI(timeout=90, max_retries=2)
    llm = llm_factory(os.getenv("EVAL_JUDGE_MODEL", "gpt-4o-mini"), client=client)
    embeddings = embedding_factory("openai", model="text-embedding-3-small", client=client, interface="modern")
    return llm, embeddings


def _prep_samples(dataset):
    # Keep failed rows in the output, preserve complete answers and real evidence.
    rows = copy.deepcopy(dataset["rag_samples"])
    for row in rows:
        row["actual_contexts"] = row.get("actual_contexts") or []
    return rows


async def run_all_metrics(golden_dataset, status_cb=None, metrics=None):
    if metrics is None:
        from ragas.metrics.collections import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall, AnswerCorrectness
        llm, embeddings = _build_judge()
        metrics = {"faithfulness": Faithfulness(llm=llm),
            "answer_relevancy": AnswerRelevancy(llm=llm, embeddings=embeddings),
            "context_precision": ContextPrecision(llm=llm),
            "context_recall": ContextRecall(llm=llm),
            "answer_correctness": AnswerCorrectness(llm=llm, embeddings=embeddings)}
    rows = _prep_samples(golden_dataset)
    for index, row in enumerate(rows):
        row["ragas_scores"], row["ragas_errors"] = {}, {}
        if row.get("status") == "error" or not row.get("actual_response", "").strip():
            row["ragas_status"] = "request_error"
            continue
        args = {"user_input": row["question"], "response": row["actual_response"],
            "reference": row["reference"], "retrieved_contexts": row["actual_contexts"]}
        fields = {"faithfulness": ("user_input", "response", "retrieved_contexts"),
            "answer_relevancy": ("user_input", "response"),
            "context_precision": ("user_input", "reference", "retrieved_contexts"),
            "context_recall": ("user_input", "reference", "retrieved_contexts"),
            "answer_correctness": ("user_input", "response", "reference")}
        for name, metric in metrics.items():
            if status_cb:
                status_cb(f"Sample {index + 1}/{len(rows)}: {name}")
            try:
                result = await asyncio.wait_for(metric.ascore(**{k: args[k] for k in fields[name]}), timeout=180)
                value = float(result.value)
                if not math.isfinite(value):
                    raise ValueError("non_finite_metric")
                row["ragas_scores"][name] = value
            except Exception as exc:
                row["ragas_scores"][name] = None
                row["ragas_errors"][name] = type(exc).__name__
        called, expected = set(row.get("actual_tools_called", [])), set(row.get("expected_tools", []))
        row["tool_exact_match"] = called == expected
        row["ragas_status"] = "scored_with_errors" if row["ragas_errors"] else "scored"
    summary = {}
    for name in metrics:
        values = [row["ragas_scores"][name] for row in rows if row["ragas_scores"].get(name) is not None]
        summary[name] = {"mean": sum(values) / len(values) if values else None,
            "scored": len(values), "total": len(rows), "unscored": len(rows) - len(values)}
    return {"method": "ragas-0.4.3", "judge_model": os.getenv("EVAL_JUDGE_MODEL", "gpt-4o-mini"),
        "embedding_model": "text-embedding-3-small", "rag_samples": rows, "summary": summary}
