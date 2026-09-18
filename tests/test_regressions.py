import ast
import contextlib
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
from helpers import functions_from

APP = Path(__file__).resolve().parents[1] / "Interprise Grade Rag"
LOG = SimpleNamespace(info=lambda *a, **k: None, warning=lambda *a, **k: None,
    error=lambda *a, **k: None, span=lambda *a, **k: contextlib.nullcontext())


class RagRegressions(unittest.TestCase):
    def test_seed_preserves_an_existing_collection(self):
        ns = functions_from(APP / "seed_knowledge.py")
        client = Mock()
        client.collection_exists.return_value = True
        client.count.return_value = SimpleNamespace(count=42)
        embedder = Mock(side_effect=AssertionError("Existing data must not be reembedded"))
        result = ns["seed_if_missing"](client, "existing", [], embedder, 2, Mock())
        self.assertEqual(result["point_count"], 42)
        client.create_collection.assert_not_called()
        client.upsert.assert_not_called()

    def test_seed_embedding_mismatch_has_no_database_writes(self):
        ns = functions_from(APP / "seed_knowledge.py")
        client = Mock()
        client.collection_exists.return_value = False
        for vectors in ([], [[0.1]]):
            with self.subTest(vectors=vectors), self.assertRaises(ValueError):
                ns["seed_if_missing"](client, "missing", [{"text": "example"}],
                    lambda text: vectors, 2, Mock())
        client.create_collection.assert_not_called()
        client.upsert.assert_not_called()

    def test_seed_new_collection_keeps_text_and_source_metadata(self):
        ns = functions_from(APP / "seed_knowledge.py")
        client = Mock()
        client.collection_exists.return_value = False
        client.create_collection.return_value = True
        client.count.return_value = SimpleNamespace(count=1)
        models = SimpleNamespace(PointStruct=lambda **kw: kw,
            VectorParams=lambda **kw: kw, Distance=SimpleNamespace(COSINE="Cosine"))
        result = ns["seed_if_missing"](client, "new", [{"id": "stable-id",
            "text": "evidence", "source": "manual.txt", "source_type": "true"}],
            lambda text: [[0.1, 0.2]], 2, models)
        self.assertEqual(result["action"], "created_and_indexed")
        self.assertTrue(client.upsert.call_args.kwargs["wait"])
        point = client.upsert.call_args.kwargs["points"][0]
        self.assertEqual(point["payload"]["text"], "evidence")
        self.assertEqual(point["payload"]["source"], "manual.txt")

    def test_error_diagnostics_keep_status_without_secret_or_prompt(self):
        import re
        import traceback
        ns = functions_from(APP / "app/diagnostics.py", re=re, traceback=traceback)
        exc = RuntimeError("secret-key and private prompt")
        exc.status_code = 401
        exc.body = {"error": {"code": "invalid_api_key", "message": "secret-key"}}
        details = ns["safe_error_details"](exc)
        self.assertEqual(details["provider_status"], 401)
        self.assertEqual(details["provider_code"], "invalid_api_key")
        self.assertNotIn("secret-key", str(details))
        self.assertNotIn("private prompt", str(details))

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
        from typing import Literal
        scope = {"BaseModel": BaseModel, "Field": Field, "uuid": uuid, "Literal": Literal, "__name__": __name__}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), "request_schema", "exec"), scope)
        request = scope["QueryRequest"]
        self.assertNotEqual(request(q="one").thread_id, request(q="two").thread_id)
        for q in ["", "   ", "x" * 8001]:
            with self.assertRaises(ValidationError): request(q=q)


if __name__ == "__main__": unittest.main()
