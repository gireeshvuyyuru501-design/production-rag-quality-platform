# Production RAG Quality Platform — MVP

A reproducible **retrieval and evidence-screening baseline** built with Python and FastAPI. Includes BM25-style lexical search, extractive source-grounded answers with citations, abstention, claim lexical screening, a labeled retrieval evaluation dataset, Pytest, GitHub Actions and Docker.

> **Scope and limitations:** This is an MVP, not yet a production deployment. It uses lexical retrieval rather than embeddings, hybrid retrieval, or reranking. The claim checker measures token overlap, **not factual entailment** and not LLM-as-a-Judge. The extractive answer returns source text instead of calling a generative LLM. Do not describe this version as a production-grade hybrid RAG or a verified hallucination detector.

## Quick start (Windows PowerShell)

```powershell
py -3 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m pytest -q
uvicorn app.main:app --reload
```

Visit http://127.0.0.1:8000/docs for interactive Swagger UI.

## Sample request

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/ask -ContentType 'application/json' -Body '{"question":"Do remote employees need a VPN?","top_k":3}'
```

## Endpoints

- `GET /health` — liveness
- `POST /retrieve` — scored source retrieval
- `POST /ask` — extractive grounded answer with source ID
- `POST /check-claims` — heuristic lexical evidence screening
- `POST /evaluate` — labeled Hit@K and MRR@K

## Evaluation

Use `data/golden_dataset.json` as starter cases. These are illustrative synthetic company policies, not private banking records. Metrics are computed on demand by `/evaluate`; no results are claimed without running tests.

## Architecture

`Question -> FastAPI -> lexical retrieval -> evidence-grounded extractive answer + citations`

`Golden dataset -> retrieve -> Hit@K + MRR@K`

`Candidate claims + source IDs -> lexical evidence screening -> review flags`

## Roadmap

1. Add semantic embeddings, BM25+dense hybrid fusion and reranking.
2. Add optional Vertex AI/Gemini grounded generation and explicit citation validation.
3. Add human-labeled claim entailment evaluation and RAGAS/LLM-as-a-Judge.
4. Add authentication/RBAC, PII protection, monitoring and load tests.
5. Compare quality, P95 latency and cost against this baseline.

## License

MIT (add a LICENSE file before publishing if you want to distribute under MIT).
