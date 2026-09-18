import logfire
from app.agents.state import AgentState
from app.services.retrieval.qdrant_service import search_enterprise_knowledge
from app.services.retrieval.ranking_service import rerank_documents


def retrieve_node(state: AgentState):
    query = state["current_query"]
    with logfire.span("Knowledge retrieval"):
        raw_results = search_enterprise_knowledge(query, limit=15)
        contents = [doc["content"] for doc in raw_results]
        if state.get("retrieval_mode", "reranked") == "vector":
            selected = contents[:5]
        else:
            selected = rerank_documents(query, contents, top_n=5)
    return {"documents": [f"CONTENT: {doc}" for doc in selected],
            "status": "Found technical context.",
            "plan": state["plan"] + ["Context Retrieved"]}
