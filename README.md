# Agentic RAG for Kubernetes documentation

Python / FastAPI / Streamlit / LangGraph / Qdrant / Gemini embeddings / FlashRank / NeMo Guardrails

[Live demo](https://enterprise-rag-rishabh.onrender.com/) · [Portfolio and demo access](https://my-portfolio-website-topaz-beta.vercel.app/)

A personal GenAI engineering project by Rishabh Shukla. It answers technical questions using retrieved Kubernetes documentation, shows the retrieved chunks, and includes an evaluation dashboard. The hosted demo is password protected and may need time to wake up.

## What is implemented

- A LangGraph planner routes conversational requests directly to a responder and technical requests through retrieval.
- Qdrant returns 15 vector candidates; FlashRank scores one candidate at a time to bound inference memory and selects 5 chunks for generation. A vector-only top-5 mode provides an evaluation baseline.
- A deterministic defense-in-depth filter blocks obvious prompt-injection and exploit requests before NeMo Guardrails. NeMo still handles the broader input-safety gate. If the safety service degrades, the API fails closed with a safe blocked response and marks the event as `degraded_fail_closed` rather than calling the RAG graph.
- FastAPI runs on loopback; Streamlit is the public, password-protected interface. Each chat and evaluation question gets an isolated thread ID.
- The Evaluation page captures complete live answers and contexts, rubric scores, latency, safety outcomes, block source (`guardrails`, `model_refusal`, or `degraded_fail_closed`), errors and a JSON/CSV export.
- Optional local RAGAS scoring uses the saved outputs and separate dependencies. Hosted rubric results are never labelled RAGAS.

This is a portfolio prototype, not a production certification. The graph is a routed workflow, not a self-correcting loop. Conversation checkpoints and hosted evaluation reports are in process/local storage and do not survive all restarts or deploys. It has not been load tested.

## Run locally

Python 3.12 or 3.13:

```bash
git clone https://github.com/rs6739171-hash/Interprise_Grade_Rag_Application.git
cd Interprise_Grade_Rag_Application
python -m venv .venv
source .venv/bin/activate
pip install -r "Interprise Grade Rag/requirements-runtime.txt"
cd "Interprise Grade Rag"
cp .env.example .env
# Fill in your provider settings in .env.
python serve.py
```

Open the Streamlit URL printed by the launcher. Required settings: `GEMINI_API_KEY`, `QDRANT_CLUSTER_ENDPOINT`, `QDRANT_API_KEY`, and either `OPENAI_API_KEY` or Portkey credentials/config. Set `APP_PASSWORD` for hosted use. Hosted evaluation additionally needs `OPENAI_API_KEY`; it uses `EVAL_JUDGE_MODEL` (default `gpt-4o-mini`). No keys belong in Git.

The existing collection must match `EMBEDDING_MODEL` and `EMBEDDING_DIM`. To create the supplied public-document collection when missing, use `RAG_SEED_IF_MISSING=1`; the seed routine preserves an existing collection. Do not change embedding models against an existing vector space.

## Evaluation

In the live app, enter the demo password, choose **Evaluation**, select 3, 5 or 15 questions and optionally enable the vector-only comparison. One background job runs at a time. Queries are serialized on the memory-constrained host. Each run also executes all 6 safety cases and keeps transport failures or fail-closed degradation separate from ordinary blocked/pass classifications.

The hosted judge produces four 0–1 rubric estimates: faithfulness, answer relevance, context relevance and answer correctness. These are **not RAGAS metrics**. The report includes the sample counts and failures, dataset hash, model names, commit, full answers and retrieved evidence. It does not replace missing retrieval with reference evidence or silently shorten answers.

Latency is API round-trip time, excluding judge time. Judge token usage is recorded when supplied by the provider; total generation cost is not instrumented. The 15 developer-authored questions are a small functional test set, not independent evidence of general reliability. Compare reranking as an experiment; no quality improvement is assumed.

For local RAGAS evaluation, install the separate pinned dependencies and score a downloaded report:

```bash
pip install -r requirements-eval.txt
python -m evals.run --engine ragas --input path/to/downloaded-report.json --output evals/results/ragas.json
```

Or capture a small local run and score it:

```bash
python -m evals.run --engine hosted --limit 3 --compare-baseline --output evals/results/hosted.json
python -m evals.run --engine ragas --input evals/results/hosted.json --output evals/results/ragas.json
```

RAGAS computes faithfulness, answer relevancy, context precision, context recall and answer correctness. Individual failures stay in the report with missing scores and error counts. It uses OpenAI for judging and evaluation embeddings, independently of the application's Gemini retrieval embeddings. Install RAGAS locally, not in the memory-constrained hosted service.

## Verification

From the repository root:

```bash
python -m unittest discover -s tests -v
python -m compileall -q "Interprise Grade Rag"
```

Tests exercise request isolation, full-output capture, no reference-context substitution, baseline routing, deterministic adversarial blocking, safe-refusal attribution, degraded fail-closed reporting, judge failures, provider errors, seeding and guardrail initialization. GitHub Actions also checks dependency compatibility. The optional RAGAS adapter smoke test runs when its dependencies are installed.

## Deployment

`render.yaml` describes a single Python service. `serve.py` supervises the private API and public UI. Automatic deployment is disabled in the blueprint; deploy the selected branch explicitly after validation. Download evaluation reports before a redeploy.

An operator can request a one-time evaluation by setting `EVAL_RUN_COMMIT` to the exact deployed commit, `EVAL_SAMPLE_LIMIT=3` (up to 15), and optionally `EVAL_COMPARE_BASELINE=1`. It is disabled by default. `EVAL_LOG_RESULTS=1` prints full benchmark outputs for this public test corpus; keep it off for private documents.
