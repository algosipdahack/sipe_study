# blog_rag_slackbot.py

"""
Step 1 ~ 7 통합 구성
- 네이버 블로그 크롤링
- 말투/스타일 벡터화 (SentenceTransformer)
- Qdrant 벡터 DB 저장
- FastAPI 슬랙 슬래시 명령어 수신
- 프롬프트 생성
- 스타일 반영된 블로그 포스트 생성
- 슬랙 응답
"""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from qdrant_client import QdrantClient
from langchain_qdrant import Qdrant
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from slack_sdk.webhook import WebhookClient
from bs4 import BeautifulSoup
import requests
import threading
import re
import os
import logging

# ---- 전역 로깅 설정 ----
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="blog_slackbot.log",   # 로그를 이 파일로 저장
    filemode="a"                    # 기존 파일에 append (덮어쓰려면 "w")
)
logger = logging.getLogger(__name__)

# ---- 설정 ----
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL")
QDRANT_COLLECTION = "naver_blog"
NAVER_BLOG_ID = "sophia5460"

# ---- FastAPI 앱 및 벡터 DB 초기화 ----
app = FastAPI()
qdrant_client = QdrantClient(host="localhost", port=6333)
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Qdrant(
    client=qdrant_client,
    collection_name=QDRANT_COLLECTION,
    embeddings=embedding
)
retriever = vectorstore.as_retriever()
llm = OpenAI(temperature=0.7, max_tokens=2048)
slack = WebhookClient(SLACK_WEBHOOK_URL)

# ---- 텍스트 정리 ----
def clean_text(text):
    text = re.sub(r"\n{2,}", "\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()

# ---- 네이버 검색 크롤링 ----
def crawl_naver_place_summary(query: str) -> str:
    search_url = f"https://search.naver.com/search.naver?query={query}+맛집"
    headers = {"User-Agent": "Mozilla/5.0"}
    res = requests.get(search_url, headers=headers)
    soup = BeautifulSoup(res.text, "html.parser")

    summaries = []
    for li in soup.select("ul.lst_total._list li")[:3]:
        title = li.select_one("a.api_txt_lines")
        desc = li.select_one("div.total_dsc")
        if title and desc:
            summaries.append(f"{title.text.strip()} - {desc.text.strip()}")

    return "\n".join(summaries) if summaries else "관련 식당 정보를 찾지 못했습니다."

# ---- 후기 생성 ----
def generate_post(request_text: str, past_style_docs: list[str]) -> str:
    style_snippets = "\n\n".join(past_style_docs[:2])
    prompt_template = PromptTemplate.from_template("""
        너는 블로거야. 아래는 너의 과거 블로그 글 스타일이야:
        ---
        {style}
        ---

        이제 다음 요청에 맞춰 같은 스타일로 후기를 작성해줘:
        '{request}'
    """)
    prompt = prompt_template.format(style=style_snippets, request=request_text)

    full_output = ""
    remaining_prompt = prompt

    for _ in range(3):
        response = llm(remaining_prompt)
        full_output += response

        if len(response.strip()) < 1800:
            break
        remaining_prompt = "계속 이어서 작성해줘.\n"

    return full_output.strip()

# ---- 사용자 입력 파싱 ----
def parse_user_text(user_text: str) -> tuple[str, str]:
    if "/" in user_text:
        place, tone = user_text.split("/", 1)
        return place.strip(), tone.strip()
    return user_text.strip(), ""

# ---- 슬래시 명령어 처리 (/blog 식당이름 / 말투)
@app.post("/slack/slash")
async def slack_slash_command(request: Request):
    form = await request.form()
    user_text = form.get("text")
    response_url = form.get("response_url")
    place, tone = parse_user_text(user_text)

    ack_text = {
        "response_type": "ephemeral",
        "text": "후기를 생성 중입니다. 잠시만 기다려 주세요 ⏳"
    }

    def generate_and_send():
        try:
            # 블로그 스타일 불러오기
            docs = retriever.get_relevant_documents("블로그 스타일")
            past_texts = [doc.page_content for doc in docs]

            # 실시간 식당 정보 크롤링
            crawled_info = crawl_naver_place_summary(place)
            combined_input = f"'{place}'라는 키워드로 검색된 식당 정보는 다음과 같습니다:\n{crawled_info}\n\n이를 참고해서 블로그 후기를 작성해줘. {tone}"
            logger.info("▶ 블로그 작성 프롬프트:\n%s", combined_input)

            # 후기 생성
            result = generate_post(combined_input, past_texts)

            # 슬랙 메시지 분할 전송
            MAX_LEN = 3900
            chunks = [result[i:i + MAX_LEN] for i in range(0, len(result), MAX_LEN)]

            for idx, chunk in enumerate(chunks):
                prefix = "📝 블로그 후기 초안:\n\n" if idx == 0 else ""
                requests.post(response_url, json={
                    "response_type": "in_channel",
                    "text": prefix + chunk
                })

        except Exception as e:
            logger.error("❌ 오류 발생: %s", e)
            requests.post(response_url, json={
                "response_type": "ephemeral",
                "text": f"후기 생성 중 오류가 발생했습니다: {str(e)}"
            })

    threading.Thread(target=generate_and_send).start()
    return JSONResponse(content=ack_text)
