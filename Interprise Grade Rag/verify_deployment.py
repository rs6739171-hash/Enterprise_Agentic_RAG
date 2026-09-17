"""Optional, local-only check of the complete deployed question path."""
import json
import urllib.error
import urllib.request
import uuid


def main():
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
