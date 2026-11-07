import pydantic
from fastapi import FastAPI
from pydantic import Field
from pydantic import BaseModel
from .rag import answer, retrieve, verify_claims, evaluate

app = FastAPI(title='Production RAG Quality Platform — MVP', version='0.1.0')

class Query(BaseModel):
    question: str = pydantic.Field(min_length=1, max_length=2000)
    top_k: int = pydantic.Field(default=3, ge=1, le=10)

class Claims(BaseModel):
    claims: list[str]
    source_ids: list[str]

class EvalCase(BaseModel):
    question: str
    relevant_doc_id: str
    k: int = pydantic.Field(default=3, ge=1, le=10)

@app.get('/health')
def health():
    return {'status': 'ok'}

@app.post('/retrieve')
def search(payload: Query):
    return {'results': retrieve(payload.question, payload.top_k)}

@app.post('/ask')
def ask(payload: Query):
    return answer(payload.question, payload.top_k)

@app.post('/check-claims')
def check_claims(payload: Claims):
    return verify_claims(payload.claims, payload.source_ids)

@app.post('/evaluate')
def evaluate_cases(cases: list[EvalCase]):
    return evaluate([case.model_dump() for case in cases])
