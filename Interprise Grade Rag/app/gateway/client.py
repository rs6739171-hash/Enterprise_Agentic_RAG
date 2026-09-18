from functools import lru_cache
from portkey_ai import Portkey, createHeaders, PORTKEY_GATEWAY_URL
from langchain_openai import ChatOpenAI

from app.config import settings


def gateway_options() -> dict:
    """Use a saved Portkey config, or inherit the API key's server-side defaults.

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
    """Return the deployment LLM used by LangChain nodes.

    Prefer direct OpenAI when OPENAI_API_KEY is configured. The deployed
    guardrails already use and validate this same path, and it avoids a
    workspace-level Portkey ``inline_config_blocked`` policy that can make
    otherwise healthy RAG requests fail during planner/response generation.

    Portkey remains the fallback for environments that do not provide a direct
    OpenAI key. In that mode we reference a saved provider by slug and never
    construct an inline routing config in application code.
    """
    if settings.OPENAI_API_KEY:
        return ChatOpenAI(
            api_key=settings.OPENAI_API_KEY,
            model=settings.OPENAI_MODEL,
            timeout=60,
            max_retries=2,
        )

    api_key = settings.PORTKEY_API_KEY
    if not api_key:
        raise RuntimeError("LLM access requires OPENAI_API_KEY or PORTKEY_API_KEY.")

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
                "environment": "production",
            },
        ),
    )


def extract_cache_status(response) -> str:
    """Pull x-portkey-cache-status from a Portkey native client response."""
    for attr in ("_raw_response", "_response", "_http_response"):
        raw = getattr(response, attr, None)
        if raw is not None:
            status = getattr(raw, "headers", {}).get("x-portkey-cache-status", "")
            if status:
                return status.upper()
    return "MISS"
