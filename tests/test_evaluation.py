import asyncio
import copy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Interprise Grade Rag"))
from evals.pipeline import query_sample, run_pipeline
from evals.guardrails_eval import run_guardrails_eval, compute_guardrails_metrics
from evals.hosted import summarize, run_evaluation
from evals.metrics import _prep_samples, run_all_metrics


def response(answer="answer", sources=None, steps=None):
    result = Mock()
    result.json.return_value = {"answer": answer, "sources": sources or [], "thought_process": steps or ["Context Retrieved"]}
    return result


class EvaluationRegressionTests(unittest.TestCase):
    def test_reranker_bounds_batches_and_preserves_global_order(self):
        import threading
        from helpers import functions_from
        root = Path(__file__).resolve().parents[1] / "Interprise Grade Rag"
        ranker = Mock()
        ranker.rerank.side_effect = lambda request: [{**request.passages[0], "score": float(request.passages[0]["text"])}]
        fn = functions_from(root / "app/services/retrieval/ranking_service.py",
            _ranker=ranker, _ranker_lock=threading.Lock(), RerankRequest=lambda **kwargs: SimpleNamespace(**kwargs),
            logfire=SimpleNamespace(error=lambda *a, **k: None))
        result = fn["rerank_documents"]("q", ["1", "3", "2"], 2)
        self.assertEqual(result, ["3", "2"])
        self.assertTrue(all(len(call.args[0].passages) == 1 for call in ranker.rerank.call_args_list))
        ranker.rerank.side_effect = RuntimeError("model failed")
        with self.assertRaises(RuntimeError):
            fn["rerank_documents"]("q", ["1"])

    def test_guard_internal_error_fails_closed(self):
        import contextlib
        import threading
        from helpers import functions_from
        fn = functions_from(Path(__file__).resolve().parents[1] / "Interprise Grade Rag/app/guardrails/rails.py",
            _rails=SimpleNamespace(generate=lambda **kwargs: {"content": "Internal server error."}),
            _guard_lock=threading.Lock(), RAIL_INDICATORS=[],
            logfire=SimpleNamespace(span=lambda *a, **k: contextlib.nullcontext()))
        with self.assertRaises(RuntimeError):
            fn["guard"]("q")

    def test_complete_outputs_and_isolated_threads(self):
        post = Mock(return_value=response("a" * 1000, ["evidence" * 500] * 5))
        first = query_sample("question", post=post)
        query_sample("question", post=post)
        self.assertEqual(len(first["actual_response"]), 1000)
        self.assertEqual(len(first["actual_contexts"]), 5)
        self.assertEqual(len(first["actual_contexts"][0]), 4000)
        self.assertNotEqual(post.call_args_list[0].kwargs["json"]["thread_id"], post.call_args_list[1].kwargs["json"]["thread_id"])

    def test_no_reference_context_substitution_or_input_mutation(self):
        source = {"rag_samples": [{"question": "question", "reference": "reference", "relevant_contexts": ["secret answer"]}]}
        original = copy.deepcopy(source)
        captured = run_pipeline(source, post=Mock(return_value=response()))
        self.assertEqual(captured["rag_samples"][0]["actual_contexts"], [])
        self.assertEqual(_prep_samples(captured)[0]["actual_contexts"], [])
        self.assertEqual(source, original)

    def test_transport_error_is_not_a_guardrail_pass(self):
        samples = [{"input": "hello", "expected_blocked": False}]
        rows = run_guardrails_eval(samples, post=Mock(side_effect=requests.Timeout()))
        self.assertEqual(rows[0]["result"], "ERROR")
        self.assertIsNone(rows[0]["actual_blocked"])
        metrics = compute_guardrails_metrics(rows)
        self.assertEqual(metrics["error"], 1)
        self.assertEqual(metrics["coverage"], 0)
        self.assertIsNone(metrics["accuracy"])

    def test_judge_failures_remain_in_denominator(self):
        rows = [{"status": "success", "latency_seconds": 1, "scores": None},
                {"status": "error", "latency_seconds": 240, "scores": None}]
        summary = summarize(rows)
        self.assertEqual(summary["requested"], 2)
        self.assertEqual(summary["judge_errors"], 1)
        self.assertEqual(summary["request_errors"], 1)
        self.assertIsNone(summary["mean_rubric_scores"]["faithfulness"])

    def test_baseline_really_bypasses_reranker(self):
        from helpers import functions_from
        import contextlib
        fn = functions_from(Path(__file__).resolve().parents[1] / "Interprise Grade Rag/app/agents/nodes/retriever.py",
            logfire=SimpleNamespace(span=lambda *a, **k: contextlib.nullcontext()))
        search = Mock(return_value=[{"content": str(i)} for i in range(15)])
        rerank = Mock(return_value=["14"])
        fn.update(search_enterprise_knowledge=search, rerank_documents=rerank)
        state = {"current_query": "q", "plan": [], "retrieval_mode": "vector"}
        self.assertEqual(len(fn["retrieve_node"](state)["documents"]), 5)
        rerank.assert_not_called()
        state["retrieval_mode"] = "reranked"
        self.assertEqual(fn["retrieve_node"](state)["documents"], ["CONTENT: 14"])

    def test_ragas_metric_failure_and_request_failure_are_preserved(self):
        class BrokenMetric:
            async def ascore(self, **kwargs):
                raise requests.Timeout()
        rows = [{"question": "q", "reference": "r", "actual_response": "a", "actual_contexts": [], "status": "success"},
                {"question": "q", "reference": "r", "actual_response": "", "status": "error"}]
        result = asyncio.run(run_all_metrics({"rag_samples": rows}, metrics={"faithfulness": BrokenMetric()}))
        self.assertEqual(len(result["rag_samples"]), 2)
        self.assertEqual(result["summary"]["faithfulness"]["unscored"], 2)
        self.assertEqual(result["rag_samples"][0]["ragas_errors"]["faithfulness"], "Timeout")

    def test_report_labels_rubric_and_keeps_partial_failure(self):
        golden = {"rag_samples": [{"question": "q", "reference": "r", "expected_tools": []}], "guardrails_samples": []}
        query = Mock(return_value={"status": "success", "latency_seconds": 1,
            "actual_contexts": [], "actual_response": "a", "actual_tools_called": []})
        with patch("evals.hosted.load_golden_dataset", return_value=golden), patch("builtins.print"):
            report = run_evaluation(1, True, query=query, judge=Mock(side_effect=RuntimeError()))
        self.assertEqual(report["method"], "hosted-rubric-v1")
        self.assertEqual(report["status"], "completed_with_errors")
        self.assertEqual(set(report["runs"]), {"reranked", "vector"})
        self.assertEqual(query.call_args_list[1].kwargs["retrieval_mode"], "vector")
