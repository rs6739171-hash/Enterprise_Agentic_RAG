"""FlashRank cross-encoder with bounded inference batches."""
import threading
import logfire
from flashrank import Ranker, RerankRequest

_ranker = None
_ranker_lock = threading.Lock()


def _get_ranker():
    global _ranker
    if _ranker is None:
        _ranker = Ranker(model_name="ms-marco-TinyBERT-L-2-v2", cache_dir="/tmp/flashrank", max_length=512)
    return _ranker


def rerank_documents(query, documents, top_n=5):
    if not documents:
        return []
    with _ranker_lock:
        try:
            ranker = _get_ranker()
            results = []
            # A 15 x 512-token batch allocates large ONNX activation buffers.
            # Keep the same per-passage score, with a batch size of one.
            for index, document in enumerate(documents):
                results.extend(ranker.rerank(RerankRequest(query=query,
                    passages=[{"id": index, "text": document}])))
            results.sort(key=lambda item: float(item["score"]), reverse=True)
            return [item["text"] for item in results[:top_n]]
        except Exception as exc:
            logfire.error("Reranking failed: {kind}", kind=type(exc).__name__)
            # A failed reranker is not a valid reranked benchmark result.
            raise RuntimeError("Reranking is temporarily unavailable.") from exc
