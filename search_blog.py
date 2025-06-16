from langchain_community.vectorstores import Qdrant
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient

def main():
    # Qdrant 서버에 연결
    qdrant = QdrantClient(host="localhost", port=6333)
    collection_name = "naver_blog"

    # 임베딩 모델
    embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Qdrant 벡터스토어 객체 생성
    vectorstore = Qdrant(
        client=qdrant,
        collection_name=collection_name,
        embedding_function=embedding.embed_query  # 🔥 핵심: 함수만 넘기기
    )

    # 사용자 검색어 입력
    query = input("🔍 검색어를 입력하세요: ")

    # 검색 실행
    results = vectorstore.similarity_search(query, k=3)

    # 결과 출력
    print("\n📌 검색 결과:")
    for i, doc in enumerate(results, 1):
        print(f"\n--- 결과 {i} ---\n{doc.page_content[:500]}...\n")

if __name__ == "__main__":
    main()
