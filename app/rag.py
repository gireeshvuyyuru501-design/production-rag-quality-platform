"""Dependency-light, inspectable retrieval and evidence-grounded answering MVP."""
import json
import math
import re
from collections import Counter
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / 'data' / 'documents.json'

def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())

def load_documents() -> list[dict]:
    return json.loads(DATA.read_text(encoding='utf-8'))

def retrieve(question: str, k: int = 3) -> list[dict]:
    """BM25-like lexical scoring; transparent baseline, not vector/hybrid retrieval."""
    docs = load_documents()
    qterms = set(tokens(question))
    if not qterms:
        return []
    lengths = [len(tokens(d['text'])) for d in docs]
    avg_len = sum(lengths) / max(len(lengths), 1)
    dfs = Counter(term for d in docs for term in set(tokens(d['text'])))
    scored = []
    for doc, length in zip(docs, lengths):
        tf = Counter(tokens(doc['text']))
        score = 0.0
        for term in qterms:
            if not tf[term]:
                continue
            idf = math.log(1 + (len(docs) - dfs[term] + 0.5) / (dfs[term] + 0.5))
            score += idf * (tf[term] * 2.2) / (tf[term] + 1.2 * (0.25 + 0.75 * length / avg_len))
        if score > 0:
            scored.append({**doc, 'score': round(score, 4)})
    return sorted(scored, key=lambda d: d['score'], reverse=True)[:k]

def answer(question: str, k: int = 3) -> dict:
    matches = retrieve(question, k)
    if not matches:
        return {'answer': 'Insufficient evidence in the available documents.', 'citations': [], 'abstained': True, 'retrieved': []}
    # Extractive baseline: quote source material, not an invented LLM response.
    selected = matches[0]
    return {'answer': selected['text'], 'citations': [selected['id']], 'abstained': False, 'retrieved': matches}

def verify_claims(claims: list[str], source_ids: list[str]) -> dict:
    """Lexical evidence flagger, NOT a factual entailment or LLM judge."""
    documents = {d['id']: d['text'] for d in load_documents()}
    corpus = ' '.join(documents[s] for s in source_ids if s in documents)
    evidence = set(tokens(corpus))
    results = []
    for claim in claims:
        meaningful = [t for t in tokens(claim) if len(t) > 3]
        coverage = sum(t in evidence for t in meaningful) / max(len(meaningful), 1)
        results.append({'claim': claim, 'lexical_coverage': round(coverage, 3), 'needs_review': coverage < 0.65})
    return {'checks': results, 'note': 'Lexical screening only; high coverage does not prove factual support.'}

def evaluate(dataset: list[dict]) -> dict:
    hits = 0
    reciprocal_ranks = 0.0
    rows = []
    for item in dataset:
        found = retrieve(item['question'], item.get('k', 3))
        ids = [d['id'] for d in found]
        expected = item['relevant_doc_id']
        hit = expected in ids
        hits += int(hit)
        reciprocal_ranks += 1 / (ids.index(expected) + 1) if hit else 0
        rows.append({'question': item['question'], 'expected': expected, 'retrieved': ids, 'hit': hit})
    n = len(dataset)
    return {'count': n, 'hit_at_k': round(hits/n, 3) if n else 0, 'mrr_at_k': round(reciprocal_ranks/n, 3) if n else 0, 'cases': rows}
