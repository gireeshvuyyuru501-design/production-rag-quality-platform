"""Streamlit frontend for the Production RAG Quality Platform FastAPI MVP."""
import os
import requests
import streamlit as st

st.set_page_config(page_title="RAG Quality Control Center", page_icon="🔎", layout="wide")
st.title("🔎 RAG Quality Control Center")
st.caption("Production RAG Quality Platform • Streamlit + FastAPI MVP")

with st.sidebar:
    st.header("Configuration")
    api_url = st.text_input("FastAPI base URL", value=os.getenv("RAG_API_URL", "http://127.0.0.1:8000")).rstrip("/")
    top_k = st.slider("Top K", 1, 10, 3)
    st.caption("Keep FastAPI running in a separate terminal.")


def api(method, path, payload=None):
    try:
        response = requests.request(method, f"{api_url}{path}", json=payload, timeout=20)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as exc:
        st.error(f"API request failed: {exc}")
        return None
    except ValueError:
        st.error("The API did not return valid JSON.")
        return None


tabs = st.tabs(["📊 Overview", "💬 RAG Playground", "🔍 Retrieval", "🛡️ Claim Checker", "📈 Evaluation"])

with tabs[0]:
    health = api("GET", "/health")
    left, middle, right = st.columns(3)
    left.metric("Backend", "Online" if health and health.get("status") == "ok" else "Unavailable")
    middle.metric("API endpoints", "5")
    right.metric("Retrieval method", "Lexical MVP")
    st.info("This is an MVP using lexical retrieval and claim screening, not semantic hallucination verification.")
    st.markdown("**Workflow:** Ask a question → inspect cited evidence → screen claims → evaluate retrieval.")

with tabs[1]:
    st.subheader("Ask the knowledge base")
    question = st.text_input("Question", "How can employees reset their password?", key="ask_question")
    if st.button("Generate answer", type="primary"):
        if not question.strip():
            st.warning("Enter a question.")
        else:
            result = api("POST", "/ask", {"question": question, "top_k": top_k})
            if result is not None:
                st.markdown("#### Answer")
                st.write(result.get("answer", "No answer returned"))
                st.write("**Abstained:**", result.get("abstained"))
                st.write("**Citations:**", ", ".join(result.get("citations", [])) or "None")
                with st.expander("Retrieved evidence", expanded=True):
                    for item in result.get("retrieved", []):
                        st.markdown(f"**{item.get('title', item.get('id', 'Document'))}** · `{item.get('id', '')}` · score `{item.get('score', 'N/A')}`")
                        st.write(item.get("text", ""))
                with st.expander("Raw API response"):
                    st.json(result)

with tabs[2]:
    st.subheader("Document retrieval")
    query = st.text_input("Search query", "password reset procedure")
    if st.button("Retrieve documents"):
        if not query.strip():
            st.warning("Enter a search query.")
        else:
            result = api("POST", "/retrieve", {"question": query, "top_k": top_k})
            if result is not None:
                st.json(result)

with tabs[3]:
    st.subheader("Lexical claim screening")
    st.caption("Enter one claim per line. Claims with low word overlap are flagged for review.")
    claims_text = st.text_area("Claims", value="Employees can reset their password through the company identity portal.\nMultifactor authentication is required to complete the reset.\nEmployees receive a free laptop every month.", height=140)
    source_text = st.text_input("Source document IDs (comma separated)", "doc-001")
    if st.button("Check claims"):
        claims = [line.strip() for line in claims_text.splitlines() if line.strip()]
        ids = [value.strip() for value in source_text.split(",") if value.strip()]
        if not claims or not ids:
            st.warning("Enter at least one claim and one source ID.")
        else:
            result = api("POST", "/check-claims", {"claims": claims, "source_ids": ids})
            if result is not None:
                checks = result.get("checks", [])
                if checks:
                    st.dataframe(checks, use_container_width=True, hide_index=True)
                st.caption(result.get("note", "Lexical screening is not semantic entailment checking."))
                with st.expander("Raw API response"):
                    st.json(result)

with tabs[4]:
    st.subheader("Retrieval evaluation")
    st.caption("Each test case supplies a question and its expected relevant document ID.")
    default_cases = [
        {"question": "How can employees reset their password?", "relevant_doc_id": "doc-001"},
        {"question": "What is the expense reimbursement policy?", "relevant_doc_id": "doc-002"},
        {"question": "What is the remote work policy?", "relevant_doc_id": "doc-003"},
    ]
    cases = st.data_editor(default_cases, num_rows="dynamic", use_container_width=True)
    if st.button("Run evaluation", type="primary"):
        valid_cases = [c for c in cases if str(c.get("question", "")).strip() and str(c.get("relevant_doc_id", "")).strip()]
        if not valid_cases:
            st.warning("Add at least one valid test case.")
        else:
            result = api("POST", "/evaluate", valid_cases)
            if result is not None:
                a, b, c = st.columns(3)
                a.metric("Test cases", result.get("count", "N/A"))
                b.metric("Hit@K", result.get("hit_at_k", "N/A"))
                c.metric("MRR@K", result.get("mrr_at_k", "N/A"))
                if result.get("cases"):
                    st.dataframe(result["cases"], use_container_width=True, hide_index=True)
                with st.expander("Raw API response"):
                    st.json(result)
