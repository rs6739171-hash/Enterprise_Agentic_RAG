"""Hosted evaluation application for the deployed RAG system."""
import csv
import io
import json
import os
from datetime import datetime

import requests
import streamlit as st

BASE_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000").rstrip("/")
REQUEST_TIMEOUT = 10


def _format_score(value):
    if value is None:
        return "—"
    return f"{float(value):.2f}"


def _format_seconds(value):
    if value is None:
        return "—"
    return f"{float(value):.1f}s"


def _backend_health():
    try:
        response = requests.get(BASE_URL + "/health", timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        return response.json().get("status") == "healthy"
    except requests.RequestException:
        return False


def _download_csv(rows):
    output = io.StringIO()
    if not rows:
        return ""
    fields = list(dict.fromkeys(key for row in rows for key in row))
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _render_mode_summary(mode, item):
    label = "Reranked retrieval" if mode == "reranked" else "Vector-only baseline"
    st.markdown(f"### {label}")
    scores = item.get("mean_rubric_scores") or {}
    cols = st.columns(4)
    cols[0].metric("Faithfulness", _format_score(scores.get("faithfulness")))
    cols[1].metric("Answer relevance", _format_score(scores.get("answer_relevance")))
    cols[2].metric("Context relevance", _format_score(scores.get("context_relevance")))
    cols[3].metric("Answer correctness", _format_score(scores.get("answer_correctness")))

    cols = st.columns(5)
    cols[0].metric("Responses", item.get("responses", 0))
    cols[1].metric("Scored", item.get("scored", 0))
    cols[2].metric("Tool exact match", _format_score(item.get("tool_exact_match")))
    cols[3].metric("P50 latency", _format_seconds(item.get("latency_p50_seconds")))
    cols[4].metric("P95 latency", _format_seconds(item.get("latency_p95_seconds")))

    if item.get("request_errors") or item.get("judge_errors"):
        st.warning(
            f"Request errors: {item.get('request_errors', 0)} · "
            f"Judge errors: {item.get('judge_errors', 0)}"
        )


def _render_guardrails(report):
    summary = report.get("guardrails_summary") or {}
    st.markdown("### Safety evaluation")
    if not summary:
        st.info("No safety evaluation data in this report.")
        return

    cols = st.columns(5)
    cols[0].metric("Accuracy", _format_score(summary.get("accuracy")))
    cols[1].metric("Precision", _format_score(summary.get("precision")))
    cols[2].metric("Recall", _format_score(summary.get("recall")))
    cols[3].metric("Coverage", _format_score(summary.get("coverage")))
    cols[4].metric("Cases", summary.get("total", 0))

    if summary.get("error"):
        st.error(f"{summary['error']} safety checks failed to complete.")
    elif summary.get("degraded_fail_closed"):
        st.warning(
            f"{summary['degraded_fail_closed']} case(s) were blocked because "
            "the safety gate was degraded."
        )
    else:
        st.success("All configured safety checks completed without evaluation errors.")

    rows = []
    for item in report.get("guardrails", []):
        rows.append({
            "case": item.get("id"),
            "type": item.get("type"),
            "input": item.get("input"),
            "expected_blocked": item.get("expected_blocked"),
            "actual_blocked": item.get("actual_blocked"),
            "result": item.get("result"),
            "block_source": item.get("block_source") or "—",
            "latency_seconds": item.get("latency_seconds"),
        })
    if rows:
        st.dataframe(rows, hide_index=True, use_container_width=True)


def _render_samples(report):
    st.markdown("### Question-level results")
    table_rows = []
    for mode, samples in report.get("runs", {}).items():
        for sample in samples:
            scores = sample.get("scores") or {}
            table_rows.append({
                "mode": mode,
                "question": sample.get("question"),
                "status": sample.get("status"),
                "faithfulness": scores.get("faithfulness"),
                "answer_relevance": scores.get("answer_relevance"),
                "context_relevance": scores.get("context_relevance"),
                "answer_correctness": scores.get("answer_correctness"),
                "latency_seconds": sample.get("latency_seconds"),
                "error": sample.get("error") or sample.get("judge_error"),
            })
    if table_rows:
        st.dataframe(table_rows, hide_index=True, use_container_width=True)

    for mode, samples in report.get("runs", {}).items():
        for sample in samples:
            title = f"{mode} · {sample.get('question', 'Untitled question')}"
            with st.expander(title):
                if sample.get("error") or sample.get("judge_error"):
                    st.error(str(sample.get("error") or sample.get("judge_error")))
                st.markdown("**Reference answer**")
                st.write(sample.get("reference", ""))
                st.markdown("**Model answer**")
                st.write(sample.get("actual_response", ""))
                st.markdown("**Scores**")
                st.json(sample.get("scores") or {})
                contexts = sample.get("actual_contexts") or []
                st.markdown(f"**Retrieved contexts ({len(contexts)})**")
                for index, context in enumerate(contexts, start=1):
                    st.caption(f"Context {index}")
                    st.code(context, language=None)


def render_dashboard():
    st.title("RAG Evaluation Lab")
    st.write(
        "Run the deployed Enterprise RAG system against its evaluation set, compare "
        "reranked retrieval with a vector-only baseline, and inspect safety behavior, "
        "latency, retrieved evidence, and answer quality."
    )

    healthy = _backend_health()
    top = st.columns([1, 1, 2])
    top[0].metric("Backend", "Online" if healthy else "Offline")
    top[1].metric("Evaluation set", "15 RAG + 6 safety")
    top[2].caption(
        "Hosted quality scores use an LLM rubric. They are intentionally kept separate "
        "from the optional local RAGAS runner."
    )

    st.divider()
    st.subheader("Run configuration")
    controls = st.columns([1, 1, 1.2])
    limit = controls[0].selectbox(
        "RAG questions per retrieval mode",
        [3, 5, 15],
        index=0,
        help="Start with 3 for a smoke test; use 15 for the full developer-authored set.",
    )
    compare = controls[1].checkbox(
        "Compare vector-only baseline",
        value=True,
        help="Runs the same questions again without FlashRank reranking.",
    )
    controls[2].caption(
        "Every run also executes all 6 safety cases. Provider API usage is incurred "
        "for RAG generation and hosted judging."
    )

    if st.button(
        "Run evaluation",
        type="primary",
        use_container_width=True,
        disabled=not healthy,
    ):
        try:
            response = requests.post(
                BASE_URL + "/evaluation/start",
                json={"limit": limit, "compare_baseline": compare},
                timeout=REQUEST_TIMEOUT,
            )
            if response.status_code == 409:
                st.warning("An evaluation is already running.")
            else:
                response.raise_for_status()
                st.success("Evaluation started. This dashboard will update automatically.")
        except requests.RequestException as exc:
            st.error(f"Could not start evaluation: {type(exc).__name__}")

    if not healthy:
        st.error("The RAG backend is not responding, so a new evaluation cannot be started.")

    @st.fragment(run_every="5s")
    def results():
        report = None
        try:
            status_response = requests.get(
                BASE_URL + "/evaluation/status", timeout=REQUEST_TIMEOUT
            )
            status_response.raise_for_status()
            status = status_response.json()

            state = status.get("status", "unknown")
            if state == "running":
                done = status.get("completed", 0)
                total = max(status.get("total", 1), 1)
                st.progress(
                    min(done / total, 1),
                    text=f"Evaluation running · {done}/{total} RAG answers processed. "
                    "Safety checks run after RAG scoring.",
                )
            elif state == "error":
                st.error("Evaluation worker failed: " + str(status.get("error")))
            elif state not in ("idle", "completed", "completed_with_errors"):
                st.caption("Evaluation state: " + str(state))

            report_response = requests.get(
                BASE_URL + "/evaluation/report", timeout=REQUEST_TIMEOUT
            )
            if report_response.status_code != 404:
                report_response.raise_for_status()
                report = report_response.json()
        except requests.RequestException:
            st.warning("The evaluation backend is temporarily unavailable.")
            return

        if report is None:
            st.info("No saved report yet. Start an evaluation above.")
            return

        st.divider()
        st.subheader("Latest evaluation report")
        started = report.get("started_at", "unknown")
        finished = report.get("finished_at")
        try:
            started_display = datetime.fromisoformat(started).strftime("%Y-%m-%d %H:%M UTC")
        except (TypeError, ValueError):
            started_display = started

        metadata = st.columns(4)
        metadata[0].metric("Run status", report.get("status", "unknown"))
        metadata[1].metric("Method", report.get("method", "unknown"))
        metadata[2].metric("Judge", report.get("judge_model", "unknown"))
        metadata[3].metric("Started", started_display)

        st.caption(
            f"Run ID: {report.get('run_id')} · Commit: {report.get('commit')} · "
            f"Dataset SHA-256: {report.get('dataset_sha256')}"
        )
        if finished:
            st.caption("Finished: " + str(finished))

        for mode, item in report.get("summary", {}).items():
            _render_mode_summary(mode, item)
            st.divider()

        _render_guardrails(report)
        st.divider()
        _render_samples(report)

        export_rows = []
        for mode, samples in report.get("runs", {}).items():
            for sample in samples:
                export_rows.append({
                    "mode": mode,
                    "question": sample.get("question"),
                    "status": sample.get("status"),
                    "latency_seconds": sample.get("latency_seconds"),
                    "error": sample.get("error") or sample.get("judge_error"),
                    **(sample.get("scores") or {}),
                })

        st.divider()
        st.subheader("Export")
        downloads = st.columns(2)
        downloads[0].download_button(
            "Download complete JSON",
            json.dumps(report, indent=2, ensure_ascii=False),
            file_name="rag-evaluation-" + str(report.get("run_id", "latest")) + ".json",
            mime="application/json",
            use_container_width=True,
        )
        csv_data = _download_csv(export_rows)
        downloads[1].download_button(
            "Download scores CSV",
            csv_data,
            file_name="rag-evaluation.csv",
            mime="text/csv",
            use_container_width=True,
            disabled=not bool(csv_data),
        )

        with st.expander("Methodology and limitations"):
            st.write(
                "The hosted judge scores faithfulness, answer relevance, context relevance, "
                "and answer correctness from 0 to 1. These are rubric estimates, not RAGAS."
            )
            for limitation in report.get("limitations", []):
                st.write("• " + limitation)

        st.caption(
            "Hosted reports use temporary service storage. Download important reports "
            "before a redeploy or restart."
        )

    results()
