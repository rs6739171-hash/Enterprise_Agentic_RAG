"""Small hosted dashboard; heavy RAGAS dependencies are only used locally."""
import csv
import io
import json
import os
import requests
import streamlit as st

BASE_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000").rstrip("/")


def render_dashboard():
    st.title("RAG evaluation")
    st.write("Run the Kubernetes question set against the deployed assistant and inspect complete answers, retrieved evidence and failures.")
    st.caption("Hosted scores use an LLM rubric, not RAGAS. The optional local RAGAS runner is documented in the repository.")
    limit = st.selectbox("Questions per retrieval mode", [3, 5, 15], index=0)
    compare = st.checkbox("Compare with vector search without reranking", value=False)
    st.caption("Each run also checks all 6 guardrail cases. Runs use your configured providers and incur API usage.")
    if st.button("Run evaluation", type="primary"):
        try:
            response = requests.post(BASE_URL + "/evaluation/start",
                json={"limit": limit, "compare_baseline": compare}, timeout=10)
            if response.status_code == 409:
                st.warning("An evaluation is already running.")
            else:
                response.raise_for_status()
                st.success("Evaluation started. You can leave this page while it runs.")
        except requests.RequestException:
            st.error("Could not start evaluation. Check that the backend is available.")

    @st.fragment(run_every="5s")
    def results():
        try:
            response = requests.get(BASE_URL + "/evaluation/status", timeout=10)
            response.raise_for_status()
            status = response.json()
            if status["status"] == "running":
                done, total = status.get("completed", 0), status.get("total", 1)
                st.progress(min(done / max(total, 1), 1),
                    text=f"Scored {done}/{total} answers. Guardrail checks finish the run.")
            elif status["status"] == "error":
                st.error("Run failed: " + str(status.get("error")))
            else:
                st.caption("Job: " + status["status"])
            response = requests.get(BASE_URL + "/evaluation/report", timeout=10)
            if response.status_code == 404:
                st.info("No saved report yet.")
                return
            response.raise_for_status()
            report = response.json()
        except requests.RequestException:
            st.warning("The backend is not responding. Try again when it is available.")
            return
        st.subheader("Latest report")
        st.caption(f"Started {report['started_at']} · {report['method']} · {report['status']}")
        st.caption("Temporary hosting storage: download the report before a redeploy.")
        summaries = []
        for mode, item in report.get("summary", {}).items():
            summaries.append({"mode": mode, "responses": item["responses"], "scored": item["scored"],
                "request_errors": item["request_errors"], "judge_errors": item["judge_errors"],
                "p50_seconds": item["latency_p50_seconds"], "p95_seconds": item["latency_p95_seconds"],
                **item["mean_rubric_scores"]})
        if summaries:
            st.dataframe(summaries, hide_index=True, use_container_width=True)
        if report.get("guardrails_summary"):
            st.write("Guardrails", report["guardrails_summary"])
        rows = []
        for mode, samples in report.get("runs", {}).items():
            for sample in samples:
                rows.append({"mode": mode, "question": sample["question"],
                    "request_status": sample["status"], "error": sample.get("error") or sample.get("judge_error"),
                    "latency_seconds": sample["latency_seconds"], **(sample.get("scores") or {})})
                with st.expander(mode + " · " + sample["question"]):
                    st.write(sample.get("actual_response", ""))
                    st.write("Scores", sample.get("scores"))
                    if sample.get("error") or sample.get("judge_error"):
                        st.error(str(sample.get("error") or sample.get("judge_error")))
                    st.write("Retrieved contexts", sample.get("actual_contexts", []))
        st.download_button("Download complete JSON", json.dumps(report, indent=2, ensure_ascii=False),
            file_name="rag-evaluation-" + report["run_id"] + ".json", mime="application/json")
        if rows:
            output = io.StringIO()
            writer = csv.DictWriter(output, fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
            writer.writeheader()
            writer.writerows(rows)
            st.download_button("Download scores CSV", output.getvalue(), file_name="rag-evaluation.csv", mime="text/csv")
        for limitation in report.get("limitations", []):
            st.caption(limitation)
    results()
