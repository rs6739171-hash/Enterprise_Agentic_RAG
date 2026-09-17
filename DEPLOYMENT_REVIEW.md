# Enterprise RAG — review and deployment preparation

Reviewed baseline: `86287a53c756f122a0ee11eb4f202928fe60f5d4`.

## Findings addressed

| Severity | Original location | Finding and correction |
|---|---|---|
| P1 | `app/guardrails/rails.py` | Initialization failures silently disabled the guardrail. Startup and requests now fail closed. This does not turn heuristic rails into a guaranteed security boundary. |
| P1 | `app/services/retrieval/embeddings.py` | A Gemini outage silently switched from 3072-dimensional embeddings to a different 768-dimensional model, incompatible with an existing index. Model changes now require explicit configuration and reindexing. |
| P1 | `app/services/retrieval/qdrant_service.py` and responder | Retrieval exceptions were swallowed as empty results, allowing an unsupported technical answer. Provider failures propagate; empty evidence results in a clear no-evidence response without generation. |
| P1 | `app/main.py:QueryRequest` | All requests omitting thread_id shared `default_user` conversation memory. Requests now get unique UUID sessions and bounded, nonblank input. |
| P2 | `app/config.py` and gateway | Configuration validation was unused and did not require Portkey. Validate actual dependencies, lazily initialize clients, and use a saved Portkey config or the API key's server-side defaults. |
| P2 | `app/main.py` and UI | Backend exceptions returned HTTP 200 and appeared successful. Errors now return 503; UI checks HTTP status. Logfire is optional. |
| P2 | `evals/pipeline.py` and parser | Linux paths used lowercase data despite uppercase DATA; answers were truncated to 300 characters and failed retrievals substituted gold context. Corrected paths, preserved full answers, and removed gold-context substitution. Evaluation callers now use valid unique UUID sessions. |

## Required configuration

`PORTKEY_API_KEY`, `GEMINI_API_KEY`, `QDRANT_CLUSTER_ENDPOINT`, and `QDRANT_API_KEY` are required. Set `PORTKEY_PRIMARY_SLUG` to an existing Portkey provider slug; the old `rag1` is only a default. Optional `PORTKEY_CONFIG_ID` selects a saved config. Otherwise requests inherit any default config attached to the API key. Configure gateway caching and fallback providers in Portkey; the app does not send inline JSON configs, which this workspace rejects with `inline_config_blocked`. `PORTKEY_FALLBACK_SLUG` is no longer used. `OPENAI_API_KEY` optionally uses a direct OpenAI model for guardrails; otherwise guardrails use Portkey. `LOGFIRE_TOKEN` is optional.

Set `OPENAI_MODEL`, `EMBEDDING_MODEL`, `EMBEDDING_DIM`, and `QDRANT_COLLECTION` to the actual deployed services. Existing model defaults are preserved; their availability and your access have not been verified. The Qdrant collection must already exist, contain your documents, and use the same embedding model/dimensions. Changing an embedding model requires rebuilding an appropriate collection; no existing data was deleted or reindexed. For first-time ingestion in a configured environment, install the full requirements and run `python -m app.ingestion.processor DATA` without `--wipe`.

For the prepared JSON chunks already tracked under `processed_data`, `python seed_knowledge.py` creates the configured collection only if it is missing and indexes those chunks with the configured embedding model. An existing collection is never overwritten. Set `RAG_SEED_IF_MISSING=1` for a one-time Render startup setup, then return it to `0`. Setup runs before the web processes to avoid overlapping their memory use. `RAG_VERIFY_ON_START=1` optionally runs one local question after startup and logs only a result summary; return it to `0` after verification.

Render uses root directory `Interprise Grade Rag`, `pip install -r requirements-runtime.txt`, and `python serve.py`. The serving requirements exclude the separate evaluation/ingestion stack. NeMo's and FlashRank's local model dependencies can exceed free-tier memory; the free plan in this blueprint is an initial build target, not a verified capacity recommendation.

## Remaining limitations

Conversation checkpoints are in memory. Retrieved chunks are displayed but document-level citations are not preserved through the current reranker. The current Colang guardrails and response substring matching are heuristic, and output safety is not independently enforced. Benchmark evaluation quality and live answer grounding still require real credentials and data. Guardrail evaluation currently scores provider failures as nonblocked; fix error accounting before using those benchmark scores as evidence of safety.

## Deployment configuration

`render.yaml` defines one Python web service. `serve.py` runs Streamlit on Render's `PORT`; where applicable, FastAPI runs only on `127.0.0.1:8000`. The launcher stops both processes if either exits. The Streamlit UI requires `APP_PASSWORD` on Render; the blueprint generates it. Retrieve that password from the service's Environment page. This is an owner/demo password gate, not enterprise identity or tenant isolation.

Secrets are declared with `sync: false`; enter real values in Render's Environment settings, never in Git or chat. The blueprint explicitly selects the free compute plan and disables automatic deployment. No paid service or database was provisioned by this change. Free instances have limited memory and can suspend; if a real build or runtime exceeds these limits, choose a suitable plan before deployment. See [Render blueprint fields](https://render.com/docs/blueprint-spec).

## Validation and limitations

The local review ran Python syntax checks, parsed the YAML, and ran focused regression tests against actual source functions with external dependencies controlled. Runtime integration tests are also included and run in GitHub Actions after installing the serving dependencies. Local dependency installation was blocked by a timeout downloading packages from files.pythonhosted.org. Therefore, a passing local isolated test does not establish that the complete deployed application starts.

No real provider requests or production data tests were performed: API credentials were not available. Model names remain configurable and preserve the existing defaults; verify access to those models in your provider account. Requirements constrain compatible major versions but are not a fully resolved lockfile. `uv.lock` and `pyproject.toml` are legacy development manifests; Render installs the explicitly named requirements file instead.

The review covered the main application, configuration, dependency, UI, and deployment paths. It is not a penetration test or proof of enterprise readiness. No obvious API-key patterns were found in the downloaded text source; Git history and binary data were not scanned.
