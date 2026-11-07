import json
from pathlib import Path
from app.rag import answer, evaluate, retrieve, verify_claims

def test_retrieval():
    assert retrieve('remote corporate VPN')[0]['id'] == 'doc-003'

def test_abstention():
    assert answer('xyzzy quux nonexistent')["abstained"] is True

def test_citation():
    assert answer('reset password multifactor')['citations'] == ['doc-001']

def test_evaluation():
    cases = json.loads((Path(__file__).resolve().parents[1]/'data'/'golden_dataset.json').read_text())
    result = evaluate(cases)
    assert result['count'] == 5
    assert 0 <= result['hit_at_k'] <= 1

def test_claims_flag_unrelated():
    result = verify_claims(['Bananas are available on Mars'], ['doc-001'])
    assert result['checks'][0]['needs_review'] is True
