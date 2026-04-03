from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

PERSIST_DIR = "chroma_chunk_db"


def load_chunk_vector_db():
    """
    載入已建立好的 chunk-level Chroma 向量資料庫
    """
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5"
    )

    vectorstore = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings
    )
    return vectorstore


def semantic_search_chunks(query: str, k: int = 5):
    """
    使用 query 到 chunk-level vector DB 搜尋最相關的 chunks
    """
    vectorstore = load_chunk_vector_db()
    results = vectorstore.similarity_search_with_score(query, k=k)

    print(f"\n查詢內容：{query}")
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


if __name__ == "__main__":
    user_query = input("請輸入新的 complaint（若直接按 Enter，使用預設測試 query）: ").strip()

    if not user_query:
        user_query = "I found unauthorized charges on my credit card statement"

    semantic_search_chunks(user_query, k=5)