"""Run Streamlit publicly and, if needed, one supervised loopback-only API."""
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request

PROJECT = Path(__file__).resolve().parent
UI = 'ui/app.py'
API = 'app.main:app'
REQUIRED = ['PORTKEY_API_KEY', 'GEMINI_API_KEY', 'QDRANT_CLUSTER_ENDPOINT', 'QDRANT_API_KEY']


def main():
    from dotenv import load_dotenv
    load_dotenv(PROJECT / ".env")
    required = REQUIRED + (["APP_PASSWORD"] if os.getenv("RENDER") else [])
    missing = [name for name in required if not os.getenv(name)]
    if missing:
        raise SystemExit("Configure these service environment variables: " + ", ".join(missing))
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT) + os.pathsep + env.get("PYTHONPATH", "")
    env["BACKEND_URL"] = "http://127.0.0.1:8000"
    children = []
    probe = None

    def stop(signum=None, frame=None):
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
            # Finish this one-time job before loading the API and UI into memory.
            subprocess.run([sys.executable, "seed_knowledge.py"], cwd=PROJECT,
                env=env, check=True, timeout=300)
        if API:
            backend = subprocess.Popen([sys.executable, "-m", "uvicorn", API,
                "--host", "127.0.0.1", "--port", "8000", "--workers", "1"], cwd=PROJECT, env=env)
            children.append(backend)
            deadline = time.monotonic() + 180
            # Local health checks must not go through an outbound proxy.
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            while time.monotonic() < deadline:
                if backend.poll() is not None:
                    raise RuntimeError("The API exited during startup; inspect the service logs.")
                try:
                    with opener.open("http://127.0.0.1:8000/health", timeout=2) as response:
                        if response.status == 200:
                            break
                except OSError:
                    time.sleep(0.5)
            else:
                raise RuntimeError("The API did not become healthy within 180 seconds.")
        children.append(subprocess.Popen([sys.executable, "-m", "streamlit", "run", UI,
            "--server.address", "0.0.0.0", "--server.port", os.getenv("PORT", "8501"),
            "--server.headless", "true", "--browser.gatherUsageStats", "false"], cwd=PROJECT, env=env))
        if os.getenv("RAG_VERIFY_ON_START") == "1":
            # This short-lived probe is not a public route and logs no answer text.
            probe = subprocess.Popen([sys.executable, "verify_deployment.py"], cwd=PROJECT, env=env)
        while all(child.poll() is None for child in children):
            time.sleep(0.5)
        raise RuntimeError("An application process stopped; restarting the service is required.")
    finally:
        stop()


if __name__ == "__main__":
    main()
