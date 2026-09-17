"""Useful error context without credentials, prompts, response bodies or locals."""
import re
import traceback


def safe_error_details(exc):
    frames = traceback.extract_tb(exc.__traceback__)
    frame = frames[-1] if frames else None
    status = getattr(exc, "status_code", None)
    code = getattr(exc, "code", None)
    if code is None:
        body = getattr(exc, "body", None)
        if isinstance(body, dict):
            error = body.get("error", body)
            if isinstance(error, dict):
                code = error.get("code") or error.get("type")
    code = str(code or "unknown")
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_.-]{0,63}", code):
        code = "unknown"
    return {
        "error_type": type(exc).__name__,
        "provider_status": status if isinstance(status, int) else None,
        "provider_code": code,
        "function": frame.name if frame else "unknown",
        "line": frame.lineno if frame else None,
        "cause_type": type(exc.__cause__).__name__ if exc.__cause__ else None,
    }
