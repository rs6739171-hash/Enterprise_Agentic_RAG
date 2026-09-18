"""Capture complete live outputs; never substitute reference evidence."""
import copy
import json
import os
from pathlib import Path
import time
import uuid
import requests

API_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000").rstrip("/") + "/query"
REQUEST_TIMEOUT = 240

def detect_tool(thought_process):
    steps = " ".join(str(x) for x in (thought_process or [])).lower()
    if "guardrails fired" in steps or "guardrails fail-closed" in steps:
        return "guardrails"
    if any(x in steps for x in ("intent: technical", "search term:", "context retrieved")):
        return "retrieve_documents"
    if "conversational" in steps or "memory" in steps:
        return "direct_answer"
    return "unknown"

def query_sample(question, retrieval_mode="reranked", post=None):
    start = time.perf_counter()
    result = {"actual_response": "", "actual_contexts": [], "actual_tools_called": [],
              "status": "error", "error": None, "retrieval_mode": retrieval_mode,
              "guardrail_status": None}
    try:
        response = (post or requests.post)(API_URL,
            json={"q": question, "thread_id": str(uuid.uuid4()), "retrieval_mode": retrieval_mode},
            timeout=(10, REQUEST_TIMEOUT))
        response.raise_for_status()
        data = response.json()
        answer, sources = data.get("answer"), data.get("sources", [])
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("empty_answer")
        if not isinstance(sources, list) or any(not isinstance(s, str) for s in sources):
            raise ValueError("invalid_sources")
        result.update(actual_response=answer, actual_contexts=sources,
            actual_tools_called=[detect_tool(data.get("thought_process"))],
            guardrail_status=data.get("guardrail_status"), status="success")
    except Exception as exc:
        result["error"] = type(exc).__name__
        status = getattr(getattr(exc, "response", None), "status_code", None)
        if status is not None:
            result["http_status"] = status
    result["latency_seconds"] = round(time.perf_counter() - start, 3)
    return result

def run_pipeline(golden_dataset, progress_callback=None, retrieval_mode="reranked", post=None):
    dataset = copy.deepcopy(golden_dataset)
    samples = dataset["rag_samples"]
    for i, sample in enumerate(samples):
        if progress_callback:
            progress_callback(i, len(samples), sample["question"], "calling")
        sample.update(query_sample(sample["question"], retrieval_mode, post))
        if progress_callback:
            progress_callback(i, len(samples), sample["question"], sample["status"], sample["actual_response"])
    return dataset

def save_results(dataset, path):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(dataset, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(target)

def load_golden_dataset():
    return json.loads(Path(__file__).with_name("golden_dataset.json").read_text(encoding="utf-8"))
