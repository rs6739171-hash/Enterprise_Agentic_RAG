import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

@unittest.skipUnless(importlib.util.find_spec("langgraph"), "Runtime dependencies are not installed")
class ApiIntegration(unittest.TestCase):
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
