import os
import json
import hashlib
from collections import Counter

import streamlit as st
from openai import OpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

CASE_DB_DIR = "chroma_db"
CHUNK_DB_DIR = "chroma_chunk_db"
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
LLM_MODEL = "gpt-5-mini"


SYSTEM_PROMPT = """
You are a senior complaint investigation analyst writing a short internal case-review memo.

Your job is to turn retrieved evidence into decision-support intelligence for a human reviewer.
Do not act like a generic summarizer.

You are given:
1. A new complaint
2. Top similar historical complaint cases
3. Top supporting evidence chunks
4. Company response labels from retrieved historical cases

You must think like an investigator and write like an analyst.

Return valid JSON only with these fields:
- complaint_pattern_summary
- likely_issue_category
- core_conflict_point
- similarity_pivot
- likely_review_paths
- risk_alert
- actionable_checklist
- note

Style requirements:
- Be concise, sharp, and operational
- Sound like an internal analyst memo
- Focus on what matters most for review
- Do not repeat all retrieved cases
- Write as if the reviewer only has 30 seconds to understand the case
- Prefer quick triage logic before deep evidence requests

Field requirements:

1. complaint_pattern_summary
- Maximum 2 sentences
- Sentence 1: state the complaint type
- Sentence 2: state the main review difficulty
- Do not add historical background here

2. likely_issue_category
- short, specific, operationally useful

3. core_conflict_point
- 1 short paragraph
- Use this structure if possible:
  "The case turns on whether ..."

4. similarity_pivot
- Name the closest complaint_id if possible
- State exactly what evidence point makes it the best match
- Do not use numerical overlap percentages

5. likely_review_paths
- Must be written exactly as:
  "Path A: ..."
  "Path B: ..."
- Each path must be 1 sentence only
- Path A should describe the more favorable historically observed path
- Path B should describe the more difficult or adverse historically observed path
- Do not write this as a paragraph

6. risk_alert
- 1 sentence only
- Name the single biggest blocker

7. actionable_checklist
- Must be a JSON array of 3 to 5 strings
- Item 1 and Item 2 must be fast checks that can be verified quickly from the complaint, transaction record, or account history
- Only later items may request merchant-side records, issuer dispute notes, signed receipts, or deeper documentation
- Order items from quick verification first to deeper investigation later
- Every item must be concrete and actionable
- Avoid vague policy language

8. note
- 1 short sentence only
- Say this is a historical investigation reference, not a guaranteed resolution

Hard rules:
- Stay grounded in retrieved evidence only
- Do NOT invent guaranteed outcomes, company actions, or timelines
- Do NOT provide legal advice
- Do NOT overstate confidence
- If evidence is mixed, say so briefly and directly
""".strip()


# -----------------------------
# Session State 初始化
# -----------------------------
DEFAULT_QUERY = "A merchant charged my credit card twice by mistake"

if "selected_query" not in st.session_state:
    st.session_state["selected_query"] = DEFAULT_QUERY

if "analysis_result" not in st.session_state:
    st.session_state["analysis_result"] = None

if "case_results" not in st.session_state:
    st.session_state["case_results"] = None

if "chunk_results" not in st.session_state:
    st.session_state["chunk_results"] = None

if "last_analyzed_query" not in st.session_state:
    st.session_state["last_analyzed_query"] = ""


# -----------------------------
# 資源載入
# -----------------------------
@st.cache_resource
def load_case_vector_db():
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)
    return Chroma(
        persist_directory=CASE_DB_DIR,
        embedding_function=embeddings
    )


@st.cache_resource
def load_chunk_vector_db():
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)
    return Chroma(
        persist_directory=CHUNK_DB_DIR,
        embedding_function=embeddings
    )


def search_cases(query: str, k: int = 3):
    case_db = load_case_vector_db()
    return case_db.similarity_search_with_score(query, k=k)


def search_chunks(query: str, k: int = 5):
    chunk_db = load_chunk_vector_db()
    return chunk_db.similarity_search_with_score(query, k=k)


# -----------------------------
# Prompt 輔助
# -----------------------------
def top_response_labels(case_results, top_n: int = 3):
    labels = []
    for doc, _ in case_results:
        label = doc.metadata.get("company_response_to_consumer")
        if label:
            labels.append(label)
    return [x for x, _ in Counter(labels).most_common(top_n)] if labels else []


def top_issue_labels(case_results, chunk_results, top_n: int = 3):
    labels = []
    for doc, _ in case_results:
        issue = doc.metadata.get("issue")
        sub_issue = doc.metadata.get("sub_issue")
        if issue:
            labels.append(f"{issue} | {sub_issue}")
    for doc, _ in chunk_results:
        issue = doc.metadata.get("issue")
        sub_issue = doc.metadata.get("sub_issue")
        if issue:
            labels.append(f"{issue} | {sub_issue}")
    return [x for x, _ in Counter(labels).most_common(top_n)] if labels else []


def detect_pattern_hints(case_results, chunk_results):
    text_blob = []
    for doc, _ in case_results:
        text_blob.append(doc.page_content.lower())
    for doc, _ in chunk_results:
        text_blob.append(doc.page_content.lower())

    text = "\n".join(text_blob)

    keyword_map = {
        "temporary_credit": ["temporary credit", "provisional credit"],
        "credit_reversed": ["reversed", "reversal", "credit had been reversed"],
        "duplicate_charge": ["charged twice", "duplicate", "rebilled", "same amount", "double charge"],
        "merchant_dispute": ["merchant", "refund", "purchase", "order", "separate bills", "separate orders"],
        "debt_not_mine": ["debt is not yours", "debt is not mine", "someone else's debt", "wrong person"],
        "repeated_contact": ["keep calling", "repeated calls", "calls daily", "harassment"],
        "payment_misapplied": ["not applied", "applied incorrectly", "applied to principal", "wrong account"],
        "credit_report_error": ["credit report", "incorrect information", "wrong alias", "hard inquiries"],
        "claim_denied": ["denied", "nothing they could do", "closed with explanation"],
        "documents_or_verification": ["documentation", "documents", "verify", "verification", "receipts"],
    }

    found = []
    for tag, keywords in keyword_map.items():
        if any(k in text for k in keywords):
            found.append(tag)

    return found


def format_case_results(case_results):
    blocks = []
    for i, (doc, score) in enumerate(case_results, start=1):
        block = [
            f"Case {i}",
            f"score: {score:.4f}",
            f"complaint_id: {doc.metadata.get('complaint_id', '')}",
            f"product: {doc.metadata.get('product', '')}",
            f"sub_product: {doc.metadata.get('sub_product', '')}",
            f"issue: {doc.metadata.get('issue', '')}",
            f"sub_issue: {doc.metadata.get('sub_issue', '')}",
            f"company: {doc.metadata.get('company', '')}",
            f"company_response_to_consumer: {doc.metadata.get('company_response_to_consumer', '')}",
            "content:",
            doc.page_content[:2200],
        ]
        blocks.append("\n".join(block))
    return "\n\n".join(blocks)


def format_chunk_results(chunk_results):
    blocks = []
    for i, (doc, score) in enumerate(chunk_results, start=1):
        block = [
            f"Chunk {i}",
            f"score: {score:.4f}",
            f"chunk_id: {doc.metadata.get('chunk_id', '')}",
            f"complaint_id: {doc.metadata.get('complaint_id', '')}",
            f"chunk_index: {doc.metadata.get('chunk_index', '')}",
            f"product: {doc.metadata.get('product', '')}",
            f"sub_product: {doc.metadata.get('sub_product', '')}",
            f"issue: {doc.metadata.get('issue', '')}",
            f"sub_issue: {doc.metadata.get('sub_issue', '')}",
            f"company: {doc.metadata.get('company', '')}",
            f"company_response_to_consumer: {doc.metadata.get('company_response_to_consumer', '')}",
            "content:",
            doc.page_content[:1600],
        ]
        blocks.append("\n".join(block))
    return "\n\n".join(blocks)


def build_prompt(query: str, case_results, chunk_results) -> str:
    common_labels = top_response_labels(case_results)
    common_issues = top_issue_labels(case_results, chunk_results)
    pattern_hints = detect_pattern_hints(case_results, chunk_results)

    cases_text = format_case_results(case_results)
    chunks_text = format_chunk_results(chunk_results)

    prompt = f"""
New complaint:
{query}

Top similar historical complaint cases:
{cases_text}

Top supporting evidence chunks:
{chunks_text}

Most common company response labels among retrieved cases:
{", ".join(common_labels) if common_labels else "N/A"}

Most common issue patterns among retrieved results:
{", ".join(common_issues) if common_issues else "N/A"}

Weak pattern hints detected from retrieved evidence:
{", ".join(pattern_hints) if pattern_hints else "N/A"}

Instructions:
- Treat this as a live case-review memo, not a generic summary.
- First identify the single most important conflict in the complaint.
- Then identify the retrieved case with the strongest evidence overlap.
- In similarity_pivot, explain the exact fact pattern that makes that case the strongest match.
- In likely_review_paths, write exactly two lines:
  "Path A: ..."
  "Path B: ..."
- Path A and Path B should reflect historically plausible directions suggested by retrieved cases, not guaranteed outcomes.
- In risk_alert, name the single biggest blocker.
- In actionable_checklist, order checks from quick verification first to deeper investigation later.
- The first 2 checklist items must be fast checks that a reviewer can verify quickly.
- Later checklist items may request merchant, issuer, or document evidence if supported by retrieved cases.
- Prefer quick triage logic before deep evidence requests.
- Do not invent certainty, future outcomes, or timelines.
- Do not use numerical overlap percentages.
- Sound like a short internal analyst note, not an encyclopedia entry.

Return valid JSON only in this format:
{{
  "complaint_pattern_summary": "...",
  "likely_issue_category": "...",
  "core_conflict_point": "...",
  "similarity_pivot": "...",
  "likely_review_paths": "...",
  "risk_alert": "...",
  "actionable_checklist": ["...", "..."],
  "note": "..."
}}
""".strip()

    return prompt


def call_llm(prompt: str):
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("找不到 OPENAI_API_KEY，請先設定。")

    client = OpenAI(api_key=api_key)
    response = client.responses.create(
        model=LLM_MODEL,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )

    text = response.output_text.strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM 回傳不是有效 JSON：{text}") from e

    required_keys = [
        "complaint_pattern_summary",
        "likely_issue_category",
        "core_conflict_point",
        "similarity_pivot",
        "likely_review_paths",
        "risk_alert",
        "actionable_checklist",
        "note",
    ]
    missing = [k for k in required_keys if k not in data]
    if missing:
        raise ValueError(f"缺少欄位: {missing}")

    if not isinstance(data["actionable_checklist"], list):
        raise ValueError("actionable_checklist 必須是 list")

    return data


# -----------------------------
# Checklist UI
# -----------------------------
def get_checklist_prefix(query: str) -> str:
    return hashlib.md5(query.encode("utf-8")).hexdigest()[:12]


def render_clickable_checklist(items, query: str):
    prefix = get_checklist_prefix(query)

    st.markdown("**Actionable Checklist**")

    for i, item in enumerate(items):
        state_key = f"{prefix}_check_{i}"
        btn_key = f"{prefix}_btn_{i}"

        if state_key not in st.session_state:
            st.session_state[state_key] = False

        col_btn, col_text = st.columns([0.16, 0.84], vertical_alignment="center")

        with col_btn:
            label = "Done" if st.session_state[state_key] else "To do"
            if st.button(label, key=btn_key, use_container_width=True):
                st.session_state[state_key] = not st.session_state[state_key]

        with col_text:
            if st.session_state[state_key]:
                st.markdown(
                    f"""
                    <div style="
                        padding: 0.8rem 1rem;
                        border-radius: 12px;
                        background: #f3f4f6;
                        color: #6b7280;
                        text-decoration: line-through;
                        border: 1px solid #e5e7eb;
                        margin-bottom: 0.5rem;
                    ">
                        {item}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div style="
                        padding: 0.8rem 1rem;
                        border-radius: 12px;
                        background: #ffffff;
                        color: #111827;
                        border: 1px solid #d1d5db;
                        margin-bottom: 0.5rem;
                    ">
                        {item}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


# -----------------------------
# 頁面設定
# -----------------------------
st.set_page_config(page_title="Complaint Investigation RAG", layout="wide")

st.markdown(
    """
    <style>
    .main > div {
        padding-top: 2rem;
    }
    .stButton>button {
        border-radius: 10px;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Complaint Investigation RAG")
st.caption("Historical case retrieval + investigation-style analysis")


# -----------------------------
# Example 按鈕
# -----------------------------
example_cols = st.columns(3)
examples = [
    "A merchant charged my credit card twice by mistake",
    "There are hard inquiries on my credit report that I did not authorize",
    "The collection account on my file does not belong to me",
]

for idx, ex in enumerate(examples):
    with example_cols[idx]:
        if st.button(f"Use example {idx+1}", key=f"use_ex_{idx}", use_container_width=True):
            st.session_state["selected_query"] = ex
            st.session_state["analysis_result"] = None
            st.session_state["case_results"] = None
            st.session_state["chunk_results"] = None
            st.session_state["last_analyzed_query"] = ""
            st.rerun()


# -----------------------------
# 輸入框
# -----------------------------
query = st.text_area(
    "Enter a new complaint",
    key="selected_query",
    height=140,
    placeholder="Type a complaint here..."
)

run_btn = st.button("Analyze Complaint", use_container_width=True)


# -----------------------------
# 觸發分析
# -----------------------------
if run_btn:
    if not query.strip():
        st.warning("請先輸入 complaint。")
    else:
        with st.spinner("Running retrieval and generating analysis..."):
            try:
                case_results = search_cases(query, k=3)
                chunk_results = search_chunks(query, k=5)
                prompt = build_prompt(query, case_results, chunk_results)
                llm_output = call_llm(prompt)

                st.session_state["case_results"] = case_results
                st.session_state["chunk_results"] = chunk_results
                st.session_state["analysis_result"] = llm_output
                st.session_state["last_analyzed_query"] = query

                st.rerun()

            except Exception as e:
                st.error(f"執行失敗：{e}")


# -----------------------------
# 顯示結果
# -----------------------------
if st.session_state["analysis_result"] is not None:
    llm_output = st.session_state["analysis_result"]
    case_results = st.session_state["case_results"]
    chunk_results = st.session_state["chunk_results"]
    analyzed_query = st.session_state["last_analyzed_query"]

    st.subheader("Investigation Summary")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Likely Issue Category**")
        st.write(llm_output["likely_issue_category"])

        st.markdown("**Core Conflict Point**")
        st.write(llm_output["core_conflict_point"])

        st.markdown("**Risk Alert**")
        st.write(llm_output["risk_alert"])

    with c2:
        st.markdown("**Similarity Pivot**")
        st.write(llm_output["similarity_pivot"])

        st.markdown("**Likely Review Paths**")
        st.write(llm_output["likely_review_paths"])

    render_clickable_checklist(llm_output["actionable_checklist"], analyzed_query)

    st.markdown("**Complaint Pattern Summary**")
    st.write(llm_output["complaint_pattern_summary"])

    st.info(llm_output["note"])

    with st.expander("Top Similar Cases", expanded=False):
        for i, (doc, score) in enumerate(case_results, start=1):
            st.markdown(f"### Case {i}")
            st.write(f"**Score:** {score:.4f}")
            st.write(f"**Complaint ID:** {doc.metadata.get('complaint_id')}")
            st.write(f"**Product:** {doc.metadata.get('product')}")
            st.write(f"**Issue:** {doc.metadata.get('issue')}")
            st.write(f"**Company:** {doc.metadata.get('company')}")
            st.write(f"**Company Response:** {doc.metadata.get('company_response_to_consumer')}")
            st.code(doc.page_content[:1200], language="text")

    with st.expander("Top Evidence Chunks", expanded=False):
        for i, (doc, score) in enumerate(chunk_results, start=1):
            st.markdown(f"### Chunk {i}")
            st.write(f"**Score:** {score:.4f}")
            st.write(f"**Chunk ID:** {doc.metadata.get('chunk_id')}")
            st.write(f"**Complaint ID:** {doc.metadata.get('complaint_id')}")
            st.write(f"**Issue:** {doc.metadata.get('issue')}")
            st.code(doc.page_content[:900], language="text")