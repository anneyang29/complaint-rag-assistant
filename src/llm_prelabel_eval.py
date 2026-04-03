import os
import json
import time
import math
import pandas as pd
from openai import OpenAI

INPUT_PATH = r"data/eval/eval_labels_enriched.csv"
OUTPUT_PATH = r"data/eval/eval_labels_llm_scored.csv"

MODEL_NAME = "gpt-5-mini"


SYSTEM_PROMPT = """
You are helping evaluate retrieval relevance for a complaint retrieval system.

Your task is to judge how relevant a retrieved result is to a user query.

Scoring rubric:
- 2 = highly relevant: the result matches the core scenario of the query very closely.
- 1 = partially relevant: the result overlaps with some important concepts but does not fully match the core scenario.
- 0 = not relevant: the result does not meaningfully match the core scenario.

Important rules:
- Focus on the core scenario of the query, not just keyword overlap.
- If the result only matches a peripheral detail, later handling step, or a broader related topic, prefer 1 instead of 2.
- If the result is clearly about a different product, issue, or scenario, assign 0.
- Be consistent and conservative.
- Return valid JSON only.
""".strip()


FEW_SHOT_BLOCK = """
Examples:

Example 1
Query: I found unauthorized charges on my credit card statement
Retrieved result:
Product: Credit card
Issue: Billing disputes
Complaint: An unauthorized charge appeared on my credit card statement and I never authorized the charge.
Label: 2
Reason: Direct match with unauthorized credit card charge on statement.

Example 2
Query: I found unauthorized charges on my credit card statement
Retrieved result:
Product: Credit reporting
Issue: Incorrect information on your report
Complaint: I found an unauthorized charge in delinquent status on my credit report.
Label: 1
Reason: Related to unauthorized charges, but the core issue is credit report accuracy rather than the original card statement dispute.

Example 3
Query: A debt collector keeps calling me every day
Retrieved result:
Product: Debt collection
Issue: Communication tactics
Complaint: They call me 20-30 times a day and keep contacting me after I asked them to stop.
Label: 2
Reason: Direct match with repeated debt collection calls.

Example 4
Query: My mortgage payment was not applied correctly
Retrieved result:
Product: Mortgage
Issue: Trouble during payment process
Complaint: My payment was incorrectly applied to principal, causing later payments to appear late.
Label: 2
Reason: Direct match with mortgage payment being applied incorrectly.

Example 5
Query: My loan servicer applied my payment to the wrong account
Retrieved result:
Product: Student loan
Issue: Trouble with how payments are handled
Complaint: Payments were inconsistently applied to principal and interest.
Label: 1
Reason: Related to payment handling problems, but it does not directly show payment being applied to the wrong account.

Example 6
Query: There is incorrect information on my credit report
Retrieved result:
Product: Credit reporting
Issue: Incorrect information on credit report
Complaint: Wrong alias and addresses are still showing on my credit report.
Label: 2
Reason: Direct match with incorrect information on credit report.
""".strip()


def load_csv_with_fallback(path: str) -> pd.DataFrame:
    encodings_to_try = ["utf-8", "utf-8-sig", "cp950", "big5", "latin1"]
    last_error = None

    for enc in encodings_to_try:
        try:
            print(f"嘗試讀取檔案: {path} | encoding={enc}")
            return pd.read_csv(path, encoding=enc)
        except Exception as e:
            last_error = e

    raise ValueError(f"無法讀取檔案: {path}，最後錯誤: {last_error}")


def normalize_text(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def parse_optional_int(value):
    text = normalize_text(value)
    if text == "":
        return None
    try:
        return int(float(text))
    except Exception:
        return None


def build_result_text(row: pd.Series) -> str:
    retrieval_type = normalize_text(row.get("retrieval_type")).lower()

    parts = [
        f"retrieval_type: {retrieval_type}",
        f"rank: {normalize_text(row.get('rank'))}",
        f"result_id: {normalize_text(row.get('result_id'))}",
    ]

    if retrieval_type == "case":
        field_map = [
            ("Product", "case_product"),
            ("Sub-product", "case_sub_product"),
            ("Issue", "case_issue"),
            ("Sub-issue", "case_sub_issue"),
            ("Company", "case_company"),
            ("Company public response", "case_company_public_response"),
            ("Company response to consumer", "case_company_response_to_consumer"),
            ("Complaint", "case_complaint_text"),
            ("Page content", "case_page_content"),
        ]
    elif retrieval_type == "chunk":
        field_map = [
            ("Product", "chunk_product"),
            ("Sub-product", "chunk_sub_product"),
            ("Issue", "chunk_issue"),
            ("Sub-issue", "chunk_sub_issue"),
            ("Company", "chunk_company"),
            ("Company public response", "chunk_company_public_response"),
            ("Company response to consumer", "chunk_company_response_to_consumer"),
            ("Parent complaint ID", "chunk_parent_complaint_id"),
            ("Chunk text", "chunk_text"),
            ("Page content", "chunk_page_content"),
        ]
    else:
        field_map = []

    for label, col in field_map:
        if col in row:
            text = normalize_text(row.get(col))
            if text:
                parts.append(f"{label}: {text}")

    return "\n".join(parts)


def call_llm(client: OpenAI, query: str, result_text: str) -> tuple[int, str]:
    user_prompt = f"""
{FEW_SHOT_BLOCK}

Now evaluate this item.

Query:
{query}

Retrieved result:
{result_text}

Return valid JSON only in this format:
{{
  "llm_relevance": 0,
  "llm_reason": "short explanation"
}}
""".strip()

    response = client.responses.create(
        model=MODEL_NAME,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )

    text = response.output_text.strip()

    try:
        data = json.loads(text)
        llm_relevance = int(data["llm_relevance"])
        llm_reason = str(data["llm_reason"]).strip()

        if llm_relevance not in [0, 1, 2]:
            raise ValueError("llm_relevance 必須是 0 / 1 / 2")

        return llm_relevance, llm_reason

    except Exception as e:
        raise ValueError(
            f"LLM 回傳格式解析失敗。\n原始輸出:\n{text}\n錯誤: {e}"
        )


def compute_final_relevance(human_rel, llm_rel):
    """
    規則：
    - human 和 llm 都有：平均後向下取整
    - 只有 llm：用 llm
    - 只有 human：用 human
    - 都沒有：None
    """
    if human_rel is not None and llm_rel is not None:
        return math.floor((human_rel + llm_rel) / 2), "human_llm_avg"

    if human_rel is None and llm_rel is not None:
        return llm_rel, "llm_only"

    if human_rel is not None and llm_rel is None:
        return human_rel, "human_only"

    return None, ""


def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("找不到 OPENAI_API_KEY，請先在 PowerShell 設定環境變數。")

    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"找不到檔案: {INPUT_PATH}")

    client = OpenAI(api_key=api_key)
    df = load_csv_with_fallback(INPUT_PATH)

    # 補欄位
    for col in ["llm_relevance", "llm_reason", "final_relevance", "label_source"]:
        if col not in df.columns:
            df[col] = pd.Series(dtype="object")
        else:
            df[col] = df[col].astype("object")

    required_cols = ["query", "retrieval_type", "rank", "result_id"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"缺少必要欄位: {missing_cols}")

    for idx, row in df.iterrows():
        query = normalize_text(row["query"])
        result_text = build_result_text(row)

        if not query or not result_text:
            print(f"第 {idx + 1} 筆缺少 query 或 result_text，跳過。")
            continue

        existing_llm = parse_optional_int(row.get("llm_relevance"))
        llm_reason = normalize_text(row.get("llm_reason"))

        print(
            f"\n處理第 {idx + 1} 筆 | "
            f"query_id={normalize_text(row.get('query_id'))} | "
            f"result_id={normalize_text(row.get('result_id'))}"
        )

        # 如果還沒有 llm_relevance，才呼叫 LLM
        if existing_llm is None:
            try:
                llm_relevance, llm_reason = call_llm(client, query, result_text)
                df.at[idx, "llm_relevance"] = str(llm_relevance)
                df.at[idx, "llm_reason"] = llm_reason
                print(f"  -> llm_relevance = {llm_relevance}")
                print(f"  -> llm_reason = {llm_reason}")
                time.sleep(0.5)
            except Exception as e:
                print(f"  -> LLM 失敗: {e}")
                llm_relevance = None
        else:
            llm_relevance = existing_llm
            print(f"  -> 已有 llm_relevance = {llm_relevance}")

        human_relevance = parse_optional_int(row.get("relevance"))
        final_rel, label_source = compute_final_relevance(human_relevance, llm_relevance)

        if final_rel is not None:
            df.at[idx, "final_relevance"] = str(final_rel)
            df.at[idx, "label_source"] = label_source
            print(f"  -> final_relevance = {final_rel} ({label_source})")
        else:
            print("  -> 無法計算 final_relevance")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 100)
    print("LLM 預標 + final_relevance 完成")
    print("=" * 100)
    print(f"輸出檔案: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()