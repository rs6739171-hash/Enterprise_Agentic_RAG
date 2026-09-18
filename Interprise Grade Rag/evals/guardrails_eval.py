"""Report transport failures separately from guardrail classifications."""
import copy
from evals.pipeline import query_sample

def run_guardrails_eval(guardrails_samples, progress_callback=None, post=None):
    samples = copy.deepcopy(guardrails_samples)
    for i, sample in enumerate(samples):
        if progress_callback:
            progress_callback(i, len(samples), sample["input"])
        outcome = query_sample(sample["input"], post=post)
        sample.update(outcome)
        if outcome["status"] != "success":
            sample.update(actual_blocked=None, result="ERROR")
            continue
        blocked = "guardrails" in outcome["actual_tools_called"]
        sample["actual_blocked"] = blocked
        sample["result"] = ("TP" if blocked else "FN") if sample["expected_blocked"] else ("FP" if blocked else "TN")
    return samples

def compute_guardrails_metrics(results):
    counts = {k.lower(): sum(r.get("result") == k for r in results) for k in ("TP", "TN", "FP", "FN", "ERROR")}
    tp, tn, fp, fn = [counts[k] for k in ("tp", "tn", "fp", "fn")]
    completed = tp + tn + fp + fn
    return {**counts, "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "accuracy": (tp + tn) / completed if completed else None,
            "total": len(results), "completed": completed, "correct": tp + tn,
            "coverage": completed / len(results) if results else None}
