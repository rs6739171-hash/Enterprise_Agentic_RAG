import ast
import contextlib
from pathlib import Path
from types import SimpleNamespace
import unittest
from helpers import functions_from

APP = Path(__file__).resolve().parents[1] / "Interprise Grade Rag"
LOG = SimpleNamespace(info=lambda *a, **k: None, warning=lambda *a, **k: None,
    error=lambda *a, **k: None, span=lambda *a, **k: contextlib.nullcontext())


class RagRegressions(unittest.TestCase):
    def test_missing_guardrails_fail_closed(self):
        ns = functions_from(APP / "app/guardrails/rails.py", _rails=None, logfire=LOG)
        with self.assertRaises(RuntimeError): ns["guard"]("technical question")

    def test_gemini_outage_does_not_switch_vector_spaces(self):
        ns = functions_from(APP / "app/services/retrieval/embeddings.py", logfire=LOG,
            settings=SimpleNamespace(EMBEDDING_MODEL="model", GEMINI_API_KEY="fake", EMBEDDING_BACKEND="gemini"),
            active_model=None, model_type=None)
        def fail(**kw): raise OSError("provider offline")
        ns["GoogleGenerativeAIEmbeddings"] = fail
        ns["load_fallback"] = lambda: self.fail("Never silently switch embedding models")
        with self.assertRaises(RuntimeError): ns["init"]()
        self.assertIsNone(ns["active_model"])

    def test_retrieval_failure_is_not_an_empty_success(self):
        from functools import lru_cache
        ns = functions_from(APP / "app/services/retrieval/qdrant_service.py", logfire=LOG, lru_cache=lru_cache)
        def fail(query): raise OSError("provider offline")
        ns["embed_query"] = fail
        with self.assertRaises(RuntimeError): ns["search_enterprise_knowledge"]("question")

    def test_no_evidence_skips_generation(self):
        ns = functions_from(APP / "app/agents/nodes/responder.py", logfire=LOG,
            get_portkey_client=lambda: self.fail("No evidence must not generate a technical answer"))
        result = ns["generate_node"]({"current_query": "technical question", "documents": [], "plan": []})
        self.assertIn("could not find supporting", result["final_answer"])

    def test_thread_default_is_unique_and_query_bounded(self):
        import uuid
        from pydantic import BaseModel, Field, ValidationError
        tree = ast.parse((APP / "app/main.py").read_text())
        node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "QueryRequest")
        scope = {"BaseModel": BaseModel, "Field": Field, "uuid": uuid, "__name__": __name__}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), "request_schema", "exec"), scope)
        request = scope["QueryRequest"]
        self.assertNotEqual(request(q="one").thread_id, request(q="two").thread_id)
        for q in ["", "   ", "x" * 8001]:
            with self.assertRaises(ValidationError): request(q=q)


if __name__ == "__main__": unittest.main()
