import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

@unittest.skipUnless(importlib.util.find_spec("langgraph"), "Runtime dependencies are not installed")
class ApiIntegration(unittest.TestCase):
    def test_guardrails_select_remote_embeddings_without_loading_local_model(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Interprise Grade Rag"))
        from types import SimpleNamespace
        from nemoguardrails.embeddings.providers import init_embedding_model
        import app.guardrails.rails as rails
        settings = SimpleNamespace(OPENAI_API_KEY="", PORTKEY_API_KEY="test",
            GEMINI_API_KEY="test", EMBEDDING_MODEL="gemini-embedding-001")
        with patch.object(rails, "settings", settings), \
             patch.object(rails, "get_langchain_llm"), patch.object(rails, "LLMRails") as constructor:
            rails.initialize_rails()
            config = constructor.call_args.args[0]
        model = next(m for m in config.models if m.type == "embeddings")
        with patch("nemoguardrails.embeddings.providers.fastembed.FastEmbedEmbeddingModel.__init__",
                   side_effect=AssertionError("Local model must not load")):
            provider = init_embedding_model(model.model, model.engine, model.parameters)
            self.assertEqual(provider.engine_name, "google")
            with patch.object(provider.client.models, "embed_content",
                              return_value=SimpleNamespace(embeddings=[SimpleNamespace(values=[0.1, 0.2])])):
                self.assertEqual(provider.encode(["hello"]), [[0.1, 0.2]])

    def test_upstream_error_is_503(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Interprise Grade Rag"))
        from fastapi.testclient import TestClient
        import app.main as main
        with patch.object(main, "validate_settings"), patch.object(main, "initialize_rails"), \
             patch.object(main, "guard", return_value=(False, None)), \
             patch.object(main.rag_agent, "invoke", side_effect=RuntimeError("provider failed")):
            with TestClient(main.app) as client:
                self.assertEqual(client.get("/health").status_code, 200)
                response = client.post("/query", json={"q":"technical question"})
                self.assertEqual(response.status_code, 503)
                self.assertNotIn("provider failed", response.text)
