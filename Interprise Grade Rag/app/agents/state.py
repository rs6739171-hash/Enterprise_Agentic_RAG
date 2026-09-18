from typing import TypedDict, Annotated, List
import operator


class AgentState(TypedDict):
    messages: Annotated[list[dict], operator.add]
    current_query: str
    documents: List[str]
    plan: List[str]
    status: str
    final_answer: str
    retrieval_mode: str
