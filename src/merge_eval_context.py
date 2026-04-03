import os
import pandas as pd

EVAL_PATH = r"data/eval/eval_labels.csv"
CASE_PATH = r"data/processed/complaints_clean.csv"
CHUNK_PATH = r"data/processed/complaints_chunked.csv"
OUTPUT_PATH = r"data/eval/eval_labels_enriched.csv"


def load_csv_with_fallback(path: str) -> pd.DataFrame:
    encodings_to_try = ["utf-8", "utf-8-sig", "cp950", "big5", "latin1"]
    last_error = None

    for enc in encodings_to_try:
        try:
            print(f"嘗試讀取 {path} | encoding={enc}")
            return pd.read_csv(path, encoding=enc)
        except Exception as e:
            last_error = e

    raise ValueError(f"無法讀取檔案: {path}，最後錯誤: {last_error}")


def main():
    if not os.path.exists(EVAL_PATH):
        raise FileNotFoundError(f"找不到檔案: {EVAL_PATH}")
    if not os.path.exists(CASE_PATH):
        raise FileNotFoundError(f"找不到檔案: {CASE_PATH}")
    if not os.path.exists(CHUNK_PATH):
        raise FileNotFoundError(f"找不到檔案: {CHUNK_PATH}")

    eval_df = load_csv_with_fallback(EVAL_PATH)
    case_df = load_csv_with_fallback(CASE_PATH)
    chunk_df = load_csv_with_fallback(CHUNK_PATH)

    # ---------- case-level 可加欄位 ----------
    case_keep_cols = [
        "complaint_id",
        "product",
        "sub_product",
        "issue",
        "sub_issue",
        "company",
        "company_public_response",
        "company_response_to_consumer",
        "complaint_text",
        "page_content"
    ]
    missing_case_cols = [c for c in case_keep_cols if c not in case_df.columns]
    if missing_case_cols:
        raise ValueError(f"complaints_clean.csv 缺少欄位: {missing_case_cols}")

    case_lookup = case_df[case_keep_cols].copy()
    case_lookup = case_lookup.rename(columns={
        "complaint_id": "result_id",
        "product": "case_product",
        "sub_product": "case_sub_product",
        "issue": "case_issue",
        "sub_issue": "case_sub_issue",
        "company": "case_company",
        "company_public_response": "case_company_public_response",
        "company_response_to_consumer": "case_company_response_to_consumer",
        "complaint_text": "case_complaint_text",
        "page_content": "case_page_content",
    })

    # ---------- chunk-level 可加欄位 ----------
    chunk_keep_cols = [
        "chunk_id",
        "complaint_id",
        "product",
        "sub_product",
        "issue",
        "sub_issue",
        "company",
        "company_public_response",
        "company_response_to_consumer",
        "chunk_text",
        "page_content"
    ]
    missing_chunk_cols = [c for c in chunk_keep_cols if c not in chunk_df.columns]
    if missing_chunk_cols:
        raise ValueError(f"complaints_chunked.csv 缺少欄位: {missing_chunk_cols}")

    chunk_lookup = chunk_df[chunk_keep_cols].copy()
    chunk_lookup = chunk_lookup.rename(columns={
        "chunk_id": "result_id",
        "complaint_id": "chunk_parent_complaint_id",
        "product": "chunk_product",
        "sub_product": "chunk_sub_product",
        "issue": "chunk_issue",
        "sub_issue": "chunk_sub_issue",
        "company": "chunk_company",
        "company_public_response": "chunk_company_public_response",
        "company_response_to_consumer": "chunk_company_response_to_consumer",
        "chunk_text": "chunk_text",
        "page_content": "chunk_page_content",
    })

    # result_id 先全部轉字串
    eval_df["result_id"] = eval_df["result_id"].astype(str)
    case_lookup["result_id"] = case_lookup["result_id"].astype(str)
    chunk_lookup["result_id"] = chunk_lookup["result_id"].astype(str)

    # 分開 merge
    eval_case = eval_df[eval_df["retrieval_type"].str.lower() == "case"].copy()
    eval_chunk = eval_df[eval_df["retrieval_type"].str.lower() == "chunk"].copy()

    eval_case = eval_case.merge(case_lookup, on="result_id", how="left")
    eval_chunk = eval_chunk.merge(chunk_lookup, on="result_id", how="left")

    # 合併回來
    enriched_df = pd.concat([eval_case, eval_chunk], ignore_index=True)

    # 依原本順序排一下
    sort_cols = [c for c in ["query_id", "retrieval_type", "rank"] if c in enriched_df.columns]
    if sort_cols:
        enriched_df = enriched_df.sort_values(sort_cols).reset_index(drop=True)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    enriched_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print("=" * 100)
    print("Merge 完成")
    print("=" * 100)
    print(f"輸出檔案: {OUTPUT_PATH}")
    print("\n前 5 筆資料：")
    print(enriched_df.head().T)


if __name__ == "__main__":
    main()