import os
import uuid
import json
import threading
from typing import Literal
from contextlib import asynccontextmanager
import logfire
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field

load_dotenv()
logfire.configure(token=os.getenv("LOGFIRE_TOKEN"), send_to_logfire="if-token-present")
from app.config import validate_settings
from app.agents.graph import rag_agent
from app.guardrails.rails import initialize_rails, guard
from app.diagnostics import safe_error_details
from evals.routes import router as evaluation_router
from evals.hosted import startup_evaluation


@asynccontextmanager
async def lifespan(app):
    validate_settings()
    initialize_rails()
    startup_evaluation()
    yield


app = FastAPI(title="Enterprise Agentic RAG API", lifespan=lifespan)
app.include_router(evaluation_router)
_query_lock = threading.Lock()


class QueryRequest(BaseModel):
    q: str = Field(min_length=1, max_length=8000, pattern=r"\S")
    # Omitted thread IDs must never share another visitor's history.
    thread_id: uuid.UUID = Field(default_factory=uuid.uuid4)
    retrieval_mode: Literal["reranked", "vector"] = "reranked"


@app.get("/")
def home():
    return {"message": "Enterprise LangGraph RAG API is live."}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/graph")
def graph_image():
    try:
        return Response(content=rag_agent.get_graph().draw_mermaid_png(), media_type="image/png")
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Graph image unavailable.") from exc


@app.post("/query")
def query(request: QueryRequest):
    # Bound memory use and serialize NeMo lazy index initialization.
    with _query_lock:
        initial_state = {
            "messages": [{"role": "user", "content": request.q}],
            "current_query": request.q, "documents": [],
            "retrieval_mode": request.retrieval_mode,
            "plan": ["Start"], "status": "Initializing Graph...",
        }
        config = {"configurable": {"thread_id": str(request.thread_id)}}
        try:
            fired, response = guard(request.q)
            if fired:
                return {"question": request.q, "answer": response,
                        "thought_process": ["Intent: Guardrails Fired", "Retrieval: Skipped"],
                        "status": "Handled by guardrails.", "sources": []}
            result = rag_agent.invoke(initial_state, config=config)
            return {"question": request.q, "answer": result.get("final_answer"),
                    "thought_process": result.get("plan"), "status": result.get("status"),
                    "sources": result.get("documents", [])}
        except Exception as exc:
            # Include safe fields in the message so Render's text logs retain them.
            logfire.error("RAG request failed: {details}", details=json.dumps(safe_error_details(exc)))
            raise HTTPException(status_code=503, detail="The AI or knowledge service is temporarily unavailable.") from exc


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000)
