from qdrant_client import QdrantClient

qdrant = QdrantClient(host="localhost", port=6333)

# 저장된 포인트 스크롤 방식으로 모두 가져오기
scroll_result, _ = qdrant.scroll(
    collection_name="naver_blog",
    limit=100,  # 한 번에 가져올 개수
    with_vectors=False,  # True로 하면 벡터도 함께 옴
    with_payload=True
)

for point in scroll_result:
    print(f"ID: {point.id}")
    print(f"메타데이터: {point.payload}")
    print("-" * 40)
