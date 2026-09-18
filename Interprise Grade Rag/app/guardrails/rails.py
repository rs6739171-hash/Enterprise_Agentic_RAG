import os
import re
import threading
import logfire
from langchain_openai import ChatOpenAI
from nemoguardrails import RailsConfig, LLMRails
from nemoguardrails.rails.llm.config import Model

from app.config import settings
from app.gateway.client import get_langchain_llm
from app.guardrails.colang_rules import COLANG_CONTENT, YAML_CONTENT, RAIL_INDICATORS


_rails: LLMRails | None = None
_guard_lock = threading.Lock()

_BLOCK_RESPONSE = (
    "I'm an Enterprise IT Assistant focused on Kubernetes, Intel hardware, and networking. "
    "I can't help with requests that try to override safety instructions or enable exploitation. "
    "I can help with defensive security and enterprise infrastructure questions."
)
_PROMPT_INJECTION_PATTERNS = (
    re.compile(r"\bignore\s+(?:all\s+)?(?:previous|prior)\s+instructions?\b", re.I),
    re.compile(r"\byou\s+are\s+now\s+dan\b", re.I),
    re.compile(r"\b(?:jailbreak|bypass)\s+(?:the\s+)?(?:system|guardrails?|safety)\b", re.I),
)
_ATTACK_REQUEST_PATTERN = re.compile(
    r"\b(?:exploit|weaponize|abuse)\b.{0,100}\b"
    r"(?:sql\s+injection|cross[- ]site\s+scripting|xss|vulnerabilit(?:y|ies)|credentials?)\b",
    re.I | re.S,
)


def deterministic_block(message: str) -> str | None:
    """Fast defense-in-depth filter for unambiguous prompt injection or exploit requests.

    This does not replace NeMo Guardrails. It prevents known adversarial patterns from
    reaching the LLM-based safety gate and deliberately avoids matching defensive
    questions such as "How do I prevent SQL injection?".
    """
    if any(pattern.search(message) for pattern in _PROMPT_INJECTION_PATTERNS):
        return _BLOCK_RESPONSE
    if _ATTACK_REQUEST_PATTERN.search(message):
        return _BLOCK_RESPONSE
    return None


def initialize_rails() -> None:
    """
    Build the NeMo LLMRails singleton at app startup.
    Uses the configured LLM and remote Gemini embeddings for intent classification.
    Initialization errors keep the service closed to requests.
    """
    global _rails

    try:
        if settings.OPENAI_API_KEY:
            guard_llm = ChatOpenAI(
                api_key=settings.OPENAI_API_KEY,
                model=settings.OPENAI_MODEL, timeout=60, max_retries=2
            )
        elif settings.PORTKEY_API_KEY:
            guard_llm = get_langchain_llm(feature="guardrails")
        else:
            raise RuntimeError("Guardrails require OPENAI_API_KEY or PORTKEY_API_KEY.")

        config = RailsConfig.from_content(
            colang_content=COLANG_CONTENT,
            yaml_content=YAML_CONTENT
        )
        # NeMo otherwise downloads a local FastEmbed model on the first query.
        # Its ONNX session can exceed the shared 512 MB web-service budget.
        # Keep semantic intent matching and every Colang flow, using the same
        # remote embedding provider that already processes retrieval queries.
        # Keep credentials out of RailsConfig reprs and exception logs.
        if settings.GEMINI_API_KEY:
            os.environ["GOOGLE_API_KEY"] = settings.GEMINI_API_KEY
        config.models.append(Model(
            type="embeddings", engine="google", model=settings.EMBEDDING_MODEL,
            parameters={},
        ))

        _rails = LLMRails(config, llm=guard_llm)
        logfire.info("🛡️ NeMo Guardrails initialised.")
    except Exception as e:
        _rails = None
        raise RuntimeError("Guardrails initialization failed; requests are disabled.") from e


def guard(message: str) -> tuple[bool, str | None]:
    """
    Run a user message through the deterministic pre-filter and NeMo rails gate.

    Returns:
        (True, rail_response) — a safety gate fired; return this response immediately
                               and skip the RAG pipeline entirely.
        (False, None)         — message is clean; proceed to LangGraph.
    """
    preflight = deterministic_block(message)
    if preflight:
        logfire.info("🛡️ Deterministic guardrail fired.")
        return True, preflight

    if _rails is None:
        raise RuntimeError("Guardrails are unavailable; requests are disabled.")

    with _guard_lock, logfire.span("🛡️ Guardrails Check"):
        result = _rails.generate(messages=[{"role": "user", "content": message}])

        # NeMo returns {'role': 'assistant', 'content': '...'} — extract text
        content = result.get("content", "") if isinstance(result, dict) else str(result)

        if not content.strip() or "internal server error" in content.lower() or "internal error" in content.lower():
            raise RuntimeError("Guardrails could not complete the check.")

        fired = any(indicator in content for indicator in RAIL_INDICATORS)

        if fired:
            logfire.info("🛡️ NeMo guardrails fired.")
            return True, content

        logfire.info("✅ Guardrails passed.")
        return False, None
