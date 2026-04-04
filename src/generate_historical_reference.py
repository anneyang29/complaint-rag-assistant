import os
import json
from collections import Counter

from openai import OpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

CASE_DB_DIR = "chroma_db"
CHUNK_DB_DIR = "chroma_chunk_db"
MODEL_NAME = "BAAI/bge-small-en-v1.5"
LLM_MODEL = "gpt-5-mini"


SYSTEM_PROMPT = """
You are helping generate a grounded complaint retrieval summary.

You are given:
1. A new complaint from a user
2. Top similar historical complaint cases
3. Top supporting evidence chunks
4. Company response labels from retrieved cases

Your task is to produce a grounded JSON output with these fields:
- complaint_pattern_summary
- likely_issue_category
- similarity_explanation
- historical_handling_reference
- note

Rules:
- Only use information supported by the retrieved cases and chunks.
- Do NOT invent guaranteed outcomes or company actions.
- Do NOT provide legal advice.
- "historical_handling_reference" should summarize common handling patterns observed in retrieved cases, such as dispute filing, temporary credit issuance, reversal, merchant verification, repeated contact, payment correction, documentation requests, or response labels.
- "likely_issue_category" should be short and specific.
- "note" should briefly say this is historical reference, not guaranteed resolution.
- Return valid JSON only.
""".strip()


def load_case_vector_db():
    embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)
    vectorstore = Chroma(
        persist_directory=CASE_DB_DIR,
        embedding_function=embeddings
    )
    return vectorstore


def load_chunk_vector_db():
    embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)
    vectorstore = Chroma(
        persist_directory=CHUNK_DB_DIR,
        embedding_function=embeddings
    )
    return vectorstore


def search_cases(query: str, k: int = 3):
    case_db = load_case_vector_db()
    return case_db.similarity_search_with_score(query, k=k)


def search_chunks(query: str, k: int = 5):
    chunk_db = load_chunk_vector_db()
    return chunk_db.similarity_search_with_score(query, k=k)


def summarize_response_labels(case_results):
    labels = []
    for doc, _ in case_results:
        label = doc.metadata.get("company_response_to_consumer")
        if label:
            labels.append(label)

    if not labels:
        return []

    return [x for x, _ in Counter(labels).most_common(3)]


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
            doc.page_content[:2000],
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
            doc.page_content[:1500],
        ]
        blocks.append("\n".join(block))
    return "\n\n".join(blocks)


def build_prompt(query: str, case_results, chunk_results) -> str:
    common_labels = summarize_response_labels(case_results)
    common_labels_text = ", ".join(common_labels) if common_labels else "N/A"

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
{common_labels_text}

Return valid JSON only in this format:
{{
  "complaint_pattern_summary": "...",
  "likely_issue_category": "...",
  "similarity_explanation": "...",
  "historical_handling_reference": "...",
  "note": "..."
}}
""".strip()

    return prompt


def call_llm(prompt: str):
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("找不到 OPENAI_API_KEY，請先在 PowerShell 設定。")

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
        required_keys = [
            "complaint_pattern_summary",
            "likely_issue_category",
            "similarity_explanation",
            "historical_handling_reference",
            "note",
        ]
        missing = [k for k in required_keys if k not in data]
        if missing:
            raise ValueError(f"缺少欄位: {missing}")
        return data
    except Exception as e:
        raise ValueError(f"LLM JSON 解析失敗。\n原始輸出:\n{text}\n錯誤: {e}")


def print_output(query: str, case_results, chunk_results, llm_output: dict):
    print("\n" + "=" * 120)
    print("FINAL RAG OUTPUT")
    print("=" * 120)

    print("\n[User Complaint]")
    print(query)

    print("\n[Likely Issue Category]")
    print(llm_output["likely_issue_category"])

    print("\n[Complaint Pattern Summary]")
    print(llm_output["complaint_pattern_summary"])

    print("\n[Similarity Explanation]")
    print(llm_output["similarity_explanation"])

    print("\n[Historical Handling Reference]")
    print(llm_output["historical_handling_reference"])

    print("\n[Note]")
    print(llm_output["note"])

    print("\n" + "-" * 120)
    print("[Top Similar Cases]")
    for i, (doc, score) in enumerate(case_results, start=1):
        print(f"\nCase {i} | score={score:.4f}")
        print(f"complaint_id: {doc.metadata.get('complaint_id')}")
        print(f"product: {doc.metadata.get('product')}")
        print(f"issue: {doc.metadata.get('issue')}")
        print(f"company: {doc.metadata.get('company')}")
        print(f"company_response_to_consumer: {doc.metadata.get('company_response_to_consumer')}")
        print(doc.page_content[:1000])

    print("\n" + "-" * 120)
    print("[Top Evidence Chunks]")
    for i, (doc, score) in enumerate(chunk_results, start=1):
        print(f"\nChunk {i} | score={score:.4f}")
        print(f"chunk_id: {doc.metadata.get('chunk_id')}")
        print(f"complaint_id: {doc.metadata.get('complaint_id')}")
        print(f"issue: {doc.metadata.get('issue')}")
        print(doc.page_content[:800])


def main():
    print("Generate Historical Handling Reference")
    user_query = input("\n請輸入新的 complaint（若直接按 Enter，使用預設測試 query）: ").strip()

    if not user_query:
        user_query = "A merchant charged my credit card twice by mistake"

    print("\n開始 retrieval...")
    case_results = search_cases(user_query, k=3)
    chunk_results = search_chunks(user_query, k=5)

    print("開始組 prompt...")
    prompt = build_prompt(user_query, case_results, chunk_results)

    print("開始呼叫 LLM 生成結果...")
    llm_output = call_llm(prompt)

    print_output(user_query, case_results, chunk_results, llm_output)


if __name__ == "__main__":
    main()