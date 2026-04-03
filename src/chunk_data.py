import os
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter

INPUT_PATH = r"data/processed/complaints_clean.csv"
OUTPUT_PATH = r"data/processed/complaints_chunked.csv"

# Recursive chunking 參數
CHUNK_SIZE = 500
CHUNK_OVERLAP = 80


def load_clean_data(input_path: str) -> pd.DataFrame:
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"找不到檔案: {input_path}")
    return pd.read_csv(input_path)


def validate_columns(df: pd.DataFrame):
    required_cols = [
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
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"complaints_clean.csv 缺少以下欄位: {missing_cols}")


def build_chunker() -> RecursiveCharacterTextSplitter:
    # recursive chunking
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )


def chunk_data(df: pd.DataFrame) -> pd.DataFrame:
    splitter = build_chunker()
    rows = []

    for _, row in df.iterrows():
        complaint_id = str(row["complaint_id"])
        complaint_text = str(row["complaint_text"]).strip()

        if not complaint_text:
            continue

        chunks = splitter.split_text(complaint_text)

        for chunk_idx, chunk_text in enumerate(chunks, start=1):
            chunk_text = chunk_text.strip()
            if not chunk_text:
                continue

            chunk_id = f"{complaint_id}_chunk_{chunk_idx}"

            # chunk-level page_content
            chunk_page_content = "\n".join([
                f"Product: {row['product']}",
                f"Sub-product: {row['sub_product']}",
                f"Issue: {row['issue']}",
                f"Sub-issue: {row['sub_issue']}",
                f"Complaint chunk: {chunk_text}"
            ])

            rows.append({
                "chunk_id": chunk_id,
                "complaint_id": complaint_id,
                "chunk_index": chunk_idx,
                "product": row["product"],
                "sub_product": row["sub_product"],
                "issue": row["issue"],
                "sub_issue": row["sub_issue"],
                "company": row["company"],
                "company_public_response": row["company_public_response"],
                "company_response_to_consumer": row["company_response_to_consumer"],
                "original_text_length": len(complaint_text),
                "chunk_text_length": len(chunk_text),
                "chunk_text": chunk_text,
                "page_content": chunk_page_content
            })

    chunked_df = pd.DataFrame(rows)
    return chunked_df


def main():
    print("開始讀取 clean csv...")
    df = load_clean_data(INPUT_PATH)

    print("檢查欄位...")
    validate_columns(df)

    print("開始 chunking...")
    chunked_df = chunk_data(df)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    chunked_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print("=" * 80)
    print("Chunking 完成")
    print(f"輸入檔案: {INPUT_PATH}")
    print(f"輸出檔案: {OUTPUT_PATH}")
    print(f"原始案例數: {len(df):,}")
    print(f"chunk 總數: {len(chunked_df):,}")

    if len(df) > 0 and len(chunked_df) > 0:
        avg_chunks = len(chunked_df) / len(df)
        print(f"平均每筆案例 chunk 數: {avg_chunks:.2f}")

    print("=" * 80)

    print("\n前 5 筆 chunk 資料：")
    print(chunked_df.head(5).T)


if __name__ == "__main__":
    main()