"""Run Streamlit publicly and one supervised loopback-only API."""
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request

PROJECT = Path(__file__).resolve().parent
UI = "ui/app.py"
API = "app.main:app"
BASE_REQUIRED = ["GEMINI_API_KEY", "QDRANT_CLUSTER_ENDPOINT", "QDRANT_API_KEY"]


def main():
    from dotenv import load_dotenv

    load_dotenv(PROJECT / ".env")
    required = BASE_REQUIRED + (["APP_PASSWORD"] if os.getenv("RENDER") else [])
    missing = [name for name in required if not os.getenv(name)]
    if missing:
        raise SystemExit("Configure these service environment variables: " + ", ".join(missing))
    if not (os.getenv("OPENAI_API_KEY") or os.getenv("PORTKEY_API_KEY") or os.getenv("PORTKEY_API")):
        raise SystemExit("Configure OPENAI_API_KEY or PORTKEY_API_KEY for LLM access.")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONUNBUFFERED"] = "1"
    env["BACKEND_URL"] = "http://127.0.0.1:8000"
    children = []
    probe = None

    def stop(signum=None, frame=None):
        nonlocal probe
        if probe is not None and probe.poll() is None:
            probe.terminate()
            try:
                probe.wait(timeout=5)
            except subprocess.TimeoutExpired:
                probe.kill()
                probe.wait()
        for child in children:
            if child.poll() is None:
                child.terminate()
        for child in children:
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        if signum is not None:
            raise SystemExit(0)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    try:
        if os.getenv("RAG_SEED_IF_MISSING") == "1":
            print("RAG_STARTUP seeding_missing_collection", flush=True)
            subprocess.run(
                [sys.executable, "seed_knowledge.py"],
                cwd=PROJECT,
                env=env,
                check=True,
                timeout=300,
            )

        print("RAG_STARTUP launching_backend", flush=True)
        backend = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", API, "--host", "127.0.0.1", "--port", "8000", "--workers", "1"],
            cwd=PROJECT,
            env=env,
        )
        children.append(backend)

        # Expose the public Render port immediately. The UI already handles a
        # temporarily unavailable backend, while this prevents Render rollout
        # from waiting on all API/guardrail initialization before detecting a port.
        print("RAG_STARTUP launching_streamlit", flush=True)
        ui = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                UI,
                "--server.address",
                "0.0.0.0",
                "--server.port",
                os.getenv("PORT", "8501"),
                "--server.headless",
                "true",
                "--browser.gatherUsageStats",
                "false",
            ],
            cwd=PROJECT,
            env=env,
        )
        children.append(ui)

        # Wait for the loopback API so optional deployment verification only
        # runs after the complete RAG stack is actually ready.
        deadline = time.monotonic() + 180
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        while time.monotonic() < deadline:
            if backend.poll() is not None:
                raise RuntimeError("The API exited during startup; inspect the service logs.")
            if ui.poll() is not None:
                raise RuntimeError("The Streamlit UI exited during startup; inspect the service logs.")
            try:
                with opener.open("http://127.0.0.1:8000/health", timeout=2) as response:
                    if response.status == 200:
                        print("RAG_STARTUP backend_healthy", flush=True)
                        break
            except OSError:
                time.sleep(0.5)
        else:
            raise RuntimeError("The API did not become healthy within 180 seconds.")

        if os.getenv("RAG_VERIFY_ON_START") == "1":
            print("RAG_STARTUP launching_verification", flush=True)
            probe = subprocess.Popen([sys.executable, "verify_deployment.py"], cwd=PROJECT, env=env)

        while all(child.poll() is None for child in children):
            time.sleep(0.5)
        raise RuntimeError("An application process stopped; restarting the service is required.")
    finally:
        stop()


if __name__ == "__main__":
    main()
