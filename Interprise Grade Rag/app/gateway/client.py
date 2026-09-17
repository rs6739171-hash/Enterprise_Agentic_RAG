from functools import lru_cache
from portkey_ai import Portkey, createHeaders, PORTKEY_GATEWAY_URL
from langchain_openai import ChatOpenAI

from app.config import settings


def gateway_options() -> dict:
    """Use a saved config, or inherit the API key's server-side defaults.

    Inline JSON configs can be forbidden by the Portkey workspace. Never send
    one or override its policy with an application-generated routing config.
    """
    if settings.PORTKEY_CONFIG_ID:
        return {"config": settings.PORTKEY_CONFIG_ID}
    return {}


@lru_cache(maxsize=1)
def get_portkey_client():
    return Portkey(api_key=settings.PORTKEY_API_KEY, timeout=60, **gateway_options())



def get_langchain_llm(feature: str = "rishu3") -> ChatOpenAI:
    """
    Returns a Portkey-backed ChatOpenAI — a drop-in for ChatGroq in LangChain nodes.

    Why ChatOpenAI and not ChatGroq:
      Portkey is a proxy. It exposes an OpenAI-compatible endpoint at PORTKEY_GATEWAY_URL.
      ChatGroq is hardwired to Groq's API and does not support routing through a proxy.
      ChatOpenAI supports base_url (points at Portkey) and default_headers (passes Portkey
      auth + metadata). The @slug/model-name format routes requests through the saved integration.
    """
    api_key = settings.PORTKEY_API_KEY
    return ChatOpenAI(
        api_key=api_key,
        timeout=60,
        max_retries=2,
        base_url=PORTKEY_GATEWAY_URL,
        model=f"@{settings.GPT_SLUG}/{settings.OPENAI_MODEL}",
        default_headers=createHeaders(
            api_key=api_key,
            **gateway_options(),
            metadata={
                "feature": feature,
                "_user": "rag-system",
                "environment": "production"
            }
        )
    )

def extract_cache_status(response) -> str:
    """
    Pull x-portkey-cache-status from the Portkey native client response headers.
    Tries multiple attribute paths defensively — returns 'MISS' if not found.
    """
    for attr in ("_raw_response", "_response", "_http_response"):
        raw = getattr(response, attr, None)
        if raw is not None:
            status = getattr(raw, "headers", {}).get("x-portkey-cache-status", "")
            if status:
                return status.upper()
    return "MISS"
