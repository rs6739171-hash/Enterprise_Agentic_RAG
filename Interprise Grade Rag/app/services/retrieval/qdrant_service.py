from functools import lru_cache
import logfire
from qdrant_client import QdrantClient
from app.config import settings
from app.services.retrieval.embeddings import embed_query


# Initialize Qdrant Client
@lru_cache(maxsize=1)
def get_client():
    return QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY, timeout=30)


def search_enterprise_knowledge(query: str, limit: int = 8):
    """
    Performs a high-precision search in the enterprise knowledge base.
    Uses the modern query_points interface.
    """
    try:
        query_vector = embed_query(query)

        # Using query_points - the modern standard for Qdrant
        response = get_client().query_points(
            collection_name=settings.QDRANT_COLLECTION,
            query=query_vector,
            limit=limit,
            with_payload=True  # JSON
        )

        results = []
        for res in response.points:
            payload = res.payload or {}
            results.append({
                "content": payload.get("text", ""),
                "source": payload.get("source", "Unknown"),
                "score": res.score
            })
        
        return results
    except Exception as e:
        logfire.error(f"❌ Qdrant Search Failed: {e}")
        raise RuntimeError("Knowledge retrieval is temporarily unavailable.") from e
