"""Optional, local-only check of the complete deployed question path."""
import json
import os
import urllib.error
import urllib.parse
import urllib.request
import uuid


def inspect_knowledge():
    """Read collection metadata only; never create or upload documents."""
    endpoint = (os.getenv("QDRANT_CLUSTER_ENDPOINT") or os.getenv("QDRANT_URL", "")).rstrip("/")
    collection = os.getenv("QDRANT_COLLECTION", "enterprise_rag")
    headers = {"api-key": os.environ["QDRANT_API_KEY"]}
    request = urllib.request.Request(endpoint + "/collections", headers=headers)
    with urllib.request.urlopen(request, timeout=30) as response:
        names = [item["name"] for item in json.load(response)["result"]["collections"]]
    metadata = {"host": urllib.parse.urlsplit(endpoint).hostname,
                "configured_collection": collection, "collections": names}
    print("RAG_KNOWLEDGE_INVENTORY " + json.dumps(metadata), flush=True)
    return collection in names


def main():
    try:
        if not inspect_knowledge():
            print('RAG_DEPLOYMENT_CHECK {"status": "configured_collection_missing"}', flush=True)
            return
    except Exception as exc:
        print("RAG_DEPLOYMENT_CHECK " + json.dumps({"stage": "knowledge_inventory",
            "error_type": type(exc).__name__}), flush=True)
        return
    request = urllib.request.Request(
        "http://127.0.0.1:8000/query",
        data=json.dumps({"q": "What is a Kubernetes deployment?",
                         "thread_id": str(uuid.uuid4())}).encode(),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(request, timeout=240) as response:
            result = json.load(response)
            print("RAG_DEPLOYMENT_CHECK " + json.dumps({
                "http_status": response.status,
                "answer_present": bool(result.get("answer")),
                "source_count": len(result.get("sources") or []),
                "status": result.get("status"),
            }), flush=True)
    except urllib.error.HTTPError as exc:
        print("RAG_DEPLOYMENT_CHECK " + json.dumps({"http_status": exc.code}), flush=True)
    except Exception as exc:
        print("RAG_DEPLOYMENT_CHECK " + json.dumps({"error_type": type(exc).__name__}), flush=True)


if __name__ == "__main__":
    main()
