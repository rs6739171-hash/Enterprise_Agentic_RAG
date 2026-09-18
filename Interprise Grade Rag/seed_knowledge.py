"""Initialize a missing collection from the repository's prepared documents."""
import json
from pathlib import Path
import uuid


def load_documents(directory: Path) -> list[dict]:
    documents = []
    for path in sorted(directory.rglob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        chunks = data.get("chunks")
        if not isinstance(chunks, list) or not all(isinstance(c, str) for c in chunks):
            raise ValueError("Prepared documents must contain a list of text chunks.")
        for index, chunk in enumerate(chunks):
            if chunk.strip():
                documents.append({
                    "id": str(uuid.uuid5(uuid.NAMESPACE_URL,
                        f"enterprise-rag/{path.relative_to(directory)}/{index}/{chunk}")),
                    "text": chunk,
                    "source": data.get("filename", path.stem),
                    "source_type": data.get("source_type", "general"),
                })
    if not documents:
        raise ValueError("No prepared document chunks were found.")
    return documents


def seed_if_missing(client, collection, documents, embedder, dimension, models):
    # An existing collection belongs to the operator: never clear or overwrite it.
    if client.collection_exists(collection):
        return {"action": "existing_collection_unchanged", "collection": collection,
                "point_count": client.count(collection, exact=True).count}
    vectors = embedder([doc["text"] for doc in documents])
    if len(vectors) != len(documents) or any(len(v) != dimension for v in vectors):
        raise ValueError("Embedding count or dimension mismatch; collection was not created.")
    points = [models.PointStruct(id=doc["id"], vector=vector,
        payload={k: doc[k] for k in ("text", "source", "source_type")})
        for doc, vector in zip(documents, vectors)]
    # Finish embedding before creating anything, so provider failures are harmless.
    created = client.create_collection(collection_name=collection,
        vectors_config=models.VectorParams(size=dimension, distance=models.Distance.COSINE))
    if not created:
        raise RuntimeError("Collection creation was not confirmed; no points were written.")
    client.upsert(collection_name=collection, points=points, wait=True)
    count = client.count(collection, exact=True).count
    if count != len(documents):
        raise RuntimeError("The new collection does not contain all prepared documents.")
    return {"action": "created_and_indexed", "collection": collection,
            "point_count": count, "source_count": len({d["source"] for d in documents})}


def main():
    from qdrant_client import QdrantClient, models
    from app.config import settings
    from app.services.retrieval.embeddings import embed_texts
    from app.diagnostics import safe_error_details
    try:
        client = QdrantClient(url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY, timeout=60)
        inventory = [c.name for c in client.get_collections().collections]
        print("RAG_KNOWLEDGE_COLLECTIONS " + json.dumps(inventory), flush=True)
        documents = load_documents(Path(__file__).parent / "processed_data")
        result = seed_if_missing(client, settings.QDRANT_COLLECTION, documents,
            embed_texts, settings.EMBEDDING_DIM, models)
        print("RAG_KNOWLEDGE_SETUP " + json.dumps(result), flush=True)
    except Exception as exc:
        print("RAG_KNOWLEDGE_SETUP " + json.dumps(safe_error_details(exc)), flush=True)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
