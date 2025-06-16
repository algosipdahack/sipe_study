import requests
import re
import time
from bs4 import BeautifulSoup
from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from tqdm import tqdm
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ---- 설정 ----
NAVER_BLOG_ID = "sophia5460"
QDRANT_COLLECTION = "naver_blog"
EMBED_MODEL = "all-MiniLM-L6-v2"
MAX_POSTS = 10  # 최신글 몇 개 수집할지

# ---- 초기화 ----
embedding = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
qdrant = QdrantClient(host="localhost", port=6333)

# 컬렉션 없으면 생성
if QDRANT_COLLECTION not in [c.name for c in qdrant.get_collections().collections]:
    qdrant.create_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    )

vectorstore = Qdrant(
    client=qdrant,
    collection_name=QDRANT_COLLECTION,
    embeddings=embedding
)

# ---- 유틸 함수 ----


# 최신 logNo 자동 수집
def get_latest_post_ids_from_rss(blog_id, max_posts=10):
    url = f"https://rss.blog.naver.com/{blog_id}.xml"
    headers = {"User-Agent": "Mozilla/5.0"}
    res = requests.get(url, headers=headers)

    soup = BeautifulSoup(res.content, "xml") # 크롤링
    items = soup.find_all("item")[:max_posts]

    post_ids = []
    for item in items:
        link = item.find("link").text
        match = re.search(r"/(\d+)", link)
        if match:
            post_ids.append(match.group(1))

    return post_ids

# 본문 긁기
def crawl_blog_text(blog_id, log_no):
    url = f"https://m.blog.naver.com/{blog_id}/{log_no}"
    headers = {"User-Agent": "Mozilla/5.0"}
    res = requests.get(url, headers=headers)
    soup = BeautifulSoup(res.text, "html.parser")
    content = soup.find("div", class_="se-main-container") or soup.find("div", class_="post_ct")
    if content:
        text = content.get_text(separator="\n")
        text = re.sub(r"\n{2,}", "\n", text)
        return text.strip()
    return None

# Qdrant 중복 체크
def is_already_inserted(log_no):
    points = qdrant.scroll(collection_name=QDRANT_COLLECTION, scroll_filter={
        "must": [{"key": "log_no", "match": {"value": log_no}}]
    }, limit=1)
    return len(points[0]) > 0

# ---- 업데이트 실행 ----
def update_qdrant_with_latest():
    post_ids = get_latest_post_ids_from_rss(NAVER_BLOG_ID, MAX_POSTS)
    texts, metadatas = [], []

    # 청크 분할기 설정 (청크당 500자, 50자 겹침)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    texts, metadatas = [], []

    for post_id in tqdm(post_ids, desc="블로그 수집 중"):
        if is_already_inserted(post_id):
            continue

        content = crawl_blog_text(NAVER_BLOG_ID, post_id)
        if content:
            chunks = text_splitter.split_text(content)  # ✅ 청킹
            for chunk in chunks:
                texts.append(chunk)
                metadatas.append({
                    "source": f"https://blog.naver.com/{NAVER_BLOG_ID}/{post_id}",
                    "log_no": post_id
                })

    if texts:
        vectorstore.add_texts(texts=texts, metadatas=metadatas)
        print(f"✅ {len(texts)}개 블로그 청크를 Qdrant에 저장 완료.")
    else:
        print("⚠️ 새로 추가할 글이 없습니다.")

# ---- 주기적 실행 ----
while True:
    update_qdrant_with_latest()
    print("⏱️ 1시간 후 재실행")
    time.sleep(3600)
