import requests
from bs4 import BeautifulSoup
import re
import time

from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Qdrant
from qdrant_client import QdrantClient
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings

# 공백 정리 함수
def clean_text(text):
    text = re.sub(r"\n{2,}", "\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()

# 블로그 글 번호 수집
def get_blog_post_ids(blog_id, max_pages=3):
    post_ids = []
    headers = {"User-Agent": "Mozilla/5.0"}
    for page in range(1, max_pages + 1):
        url = f"https://blog.naver.com/PostList.naver?blogId={blog_id}&currentPage={page}"
        res = requests.get(url, headers=headers)
        soup = BeautifulSoup(res.text, "html.parser")
        scripts = soup.find_all("script")
        for script in scripts:
            if "logNo=" in script.text:
                found = re.findall(r"logNo=(\d+)", script.text)
                post_ids.extend(found)
        time.sleep(1)
    return list(set(post_ids))

# 본문 크롤링 (모바일 블로그 기준)
def crawl_mobile_blog(blog_id, post_id):
    url = f"https://m.blog.naver.com/{blog_id}/{post_id}"
    headers = {"User-Agent": "Mozilla/5.0"}
    res = requests.get(url, headers=headers)
    if res.status_code != 200:
        return None
    soup = BeautifulSoup(res.text, 'html.parser')
    content = soup.find("div", class_="se-main-container") or soup.find("div", class_="post_ct")
    if not content:
        return None
    return clean_text(content.get_text(separator="\n"))

# 실행 부분
def main():
    blog_id = "sophia5460"  # ← 너의 블로그 ID
    collection_name = "naver_blog"
    max_pages = 3

    # 1. 글 목록 수집
    post_ids = get_blog_post_ids(blog_id, max_pages=max_pages)
    print(f"[총 {len(post_ids)}개 게시물 발견]")

    # 2. 본문 크롤링 + Document 변환
    docs = []
    for post_id in post_ids:
        try:
            text = crawl_mobile_blog(blog_id, post_id)
            if text:
                docs.append(Document(page_content=text))
                print(f"✓ {post_id} 저장 완료")
            else:
                print(f"✗ {post_id} 본문 없음")
        except Exception as e:
            print(f"⚠️ {post_id} 에러: {e}")

    # 3. Qdrant 연결 및 저장
    if docs:
        embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = Qdrant.from_documents(
            documents=docs,
            embedding=embedding,
            url="http://localhost:6333",
            collection_name=collection_name
        )
        print("✅ 모든 게시글 Qdrant에 저장 완료!")

    else:
        print("❌ 저장할 문서가 없습니다.")

if __name__ == "__main__":
    main()
