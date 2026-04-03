from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

CASE_DB_DIR = "chroma_db"
CHUNK_DB_DIR = "chroma_chunk_db"
MODEL_NAME = "BAAI/bge-small-en-v1.5"


def load_case_vector_db():
    """
    載入 case-level 向量資料庫
    """
    embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)
    vectorstore = Chroma(
        persist_directory=CASE_DB_DIR,
        embedding_function=embeddings
    )
    return vectorstore


def load_chunk_vector_db():
    """
    載入 chunk-level 向量資料庫
    """
    embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)
    vectorstore = Chroma(
        persist_directory=CHUNK_DB_DIR,
        embedding_function=embeddings
    )
    return vectorstore


def search_cases(query: str, k: int = 3):
    """
    搜尋最相似的完整 complaint cases
    """
    case_db = load_case_vector_db()
    results = case_db.similarity_search_with_score(query, k=k)
    return results


def search_chunks(query: str, k: int = 5):
    """
    搜尋最相似的 complaint chunks
    """
    chunk_db = load_chunk_vector_db()
    results = chunk_db.similarity_search_with_score(query, k=k)
    return results


def print_case_results(results):
    print("\n" + "=" * 120)
    print("Case-level Retrieval Results（相似完整案例）")
    print("=" * 120)

    for i, (doc, score) in enumerate(results, start=1):
        print(f"\n結果 {i}")
        print(f"retrieval score: {score:.4f}")
        print(f"complaint_id: {doc.metadata.get('complaint_id')}")
        print(f"product: {doc.metadata.get('product')}")
        print(f"sub_product: {doc.metadata.get('sub_product')}")
        print(f"issue: {doc.metadata.get('issue')}")
        print(f"sub_issue: {doc.metadata.get('sub_issue')}")
        print(f"company: {doc.metadata.get('company')}")
        print(f"company_response_to_consumer: {doc.metadata.get('company_response_to_consumer')}")
        print("\n內容：")
        print(doc.page_content[:1500])
        print("-" * 120)


def print_chunk_results(results):
    print("\n" + "=" * 120)
    print("Chunk-level Retrieval Results（支持證據片段）")
    print("=" * 120)

    for i, (doc, score) in enumerate(results, start=1):
        print(f"\n結果 {i}")
        print(f"retrieval score: {score:.4f}")
        print(f"chunk_id: {doc.metadata.get('chunk_id')}")
        print(f"complaint_id: {doc.metadata.get('complaint_id')}")
        print(f"chunk_index: {doc.metadata.get('chunk_index')}")
        print(f"product: {doc.metadata.get('product')}")
        print(f"sub_product: {doc.metadata.get('sub_product')}")
        print(f"issue: {doc.metadata.get('issue')}")
        print(f"sub_issue: {doc.metadata.get('sub_issue')}")
        print(f"company: {doc.metadata.get('company')}")
        print(f"company_response_to_consumer: {doc.metadata.get('company_response_to_consumer')}")
        print(f"original_text_length: {doc.metadata.get('original_text_length')}")
        print(f"chunk_text_length: {doc.metadata.get('chunk_text_length')}")
        print("\n內容：")
        print(doc.page_content[:1500])
        print("-" * 120)


def main():
    print("Hybrid Retrieval Demo")

    user_query = input("\n請輸入新的 complaint（若直接按 Enter，使用預設測試 query）: ").strip()

    if not user_query:
        user_query = "I found unauthorized charges on my credit card statement"

    print("\n" + "=" * 120)
    print(f"查詢內容：{user_query}")
    print("=" * 120)

    # Case-level retrieval
    case_results = search_cases(user_query, k=3)
    print_case_results(case_results)

    # Chunk-level retrieval
    chunk_results = search_chunks(user_query, k=5)
    print_chunk_results(chunk_results)

    print("\n" + "=" * 120)
    print("Demo 完成")


if __name__ == "__main__":
    main()