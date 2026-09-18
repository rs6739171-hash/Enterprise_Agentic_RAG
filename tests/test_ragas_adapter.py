"""Optional dependency smoke test; no provider calls or real credentials."""
import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("ragas"), "Optional RAGAS environment is not installed")
class RagasAdapterTests(unittest.TestCase):
    def test_pinned_factories_and_metric_constructors(self):
        from openai import AsyncOpenAI
        from ragas.llms import llm_factory
        from ragas.embeddings import embedding_factory
        from ragas.metrics.collections import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall, AnswerCorrectness
        client = AsyncOpenAI(api_key="test-not-a-real-key")
        llm = llm_factory("gpt-4o-mini", client=client)
        embeddings = embedding_factory("openai", model="text-embedding-3-small", client=client, interface="modern")
        for metric in (Faithfulness(llm=llm), AnswerRelevancy(llm=llm, embeddings=embeddings),
                       ContextPrecision(llm=llm), ContextRecall(llm=llm), AnswerCorrectness(llm=llm, embeddings=embeddings)):
            self.assertTrue(callable(metric.ascore))
