import os
import shutil
import pandas as pd
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

INPUT_PATH = r"data/processed/complaints_chunked.csv"
PERSIST_DIR = r"chroma_chunk_db"


def load_chunk_data(input_path: str) -> pd.DataFrame:
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"找不到檔案: {input_path}")
    df = pd.read_csv(input_path)
    return df


def validate_columns(df: pd.DataFrame):
    required_cols = [
        "chunk_id",
        "complaint_id",
        "chunk_index",
        "product",
        "sub_product",
        "issue",
        "sub_issue",
        "company",
        "company_public_response",
        "company_response_to_consumer",
        "original_text_length",
        "chunk_text_length",
        "chunk_text",
        "page_content"
    ]

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"complaints_chunked.csv 缺少以下欄位: {missing_cols}")


def build_documents(df: pd.DataFrame) -> list[Document]:
    documents = []

    for _, row in df.iterrows():
        metadata = {
            "chunk_id": str(row["chunk_id"]),
            "complaint_id": str(row["complaint_id"]),
            "chunk_index": int(row["chunk_index"]),
            "product": str(row["product"]),
            "sub_product": str(row["sub_product"]),
            "issue": str(row["issue"]),
            "sub_issue": str(row["sub_issue"]),
            "company": str(row["company"]),
            "company_public_response": str(row["company_public_response"]),
            "company_response_to_consumer": str(row["company_response_to_consumer"]),
            "original_text_length": int(row["original_text_length"]),
            "chunk_text_length": int(row["chunk_text_length"]),
        }

        doc = Document(
            page_content=str(row["page_content"]),
            metadata=metadata
        )
        documents.append(doc)

    return documents


def build_vector_db(documents: list[Document], persist_dir: str):
    # 如果想重建資料庫，就先刪掉舊的
    if os.path.exists(persist_dir):
        shutil.rmtree(persist_dir)

    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5"
    )

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_dir
    )

    return vectorstore


def main():
    print("開始讀取 chunked csv...")
    df = load_chunk_data(INPUT_PATH)

    print("檢查欄位...")
    validate_columns(df)

    print("開始建立 chunk-level documents...")
    documents = build_documents(df)

    print(f"chunk documents 數量: {len(documents):,}")
    print("建立 chunk-level 向量資料庫中...")

    build_vector_db(documents, PERSIST_DIR)

    print("=" * 80)
    print("Chunk-level 向量資料庫建立完成")
    print(f"輸入檔案: {INPUT_PATH}")
    print(f"儲存位置: {PERSIST_DIR}")
    print(f"chunk 數量: {len(documents):,}")
    print("=" * 80)

    if len(documents) > 0:
        print("\n範例 chunk-level document：")
        print(documents[0].page_content[:1000])
        print("\n範例 metadata：")
        print(documents[0].metadata)


if __name__ == "__main__":
    main()