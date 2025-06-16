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

from fastapi import FastAPI, Request, Form
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from qdrant_client import QdrantClient
from langchain_qdrant import Qdrant
from langchain_community.embeddings import HuggingFaceEmbeddings  # ✅ 최신 방식
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from slack_sdk.webhook import WebhookClient
from fastapi.responses import JSONResponse
import requests
from bs4 import BeautifulSoup
import re
import os

# ---- Config ----
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL")
QDRANT_COLLECTION = "naver_blog"
NAVER_BLOG_ID = "sophia5460"

# ---- Init ----
app = FastAPI()
qdrant_client = QdrantClient(host="localhost", port=6333)
# 텍스트를 벡터로 변환해주는 도구
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Qdrant(
    client=qdrant_client,
    collection_name=QDRANT_COLLECTION,
    embeddings=embedding
)
retriever = vectorstore.as_retriever()
llm = OpenAI(temperature=0.7, max_tokens=2048) # 0.7 : 후기/블로그 스타일 적합
slack = WebhookClient(SLACK_WEBHOOK_URL)

# ---- 텍스트 정리 ----
def clean_text(text):
    text = re.sub(r"\n{2,}", "\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()

def generate_post(request_text: str, past_style_docs: list[str]) -> str:
    style_snippets = "\n\n".join(past_style_docs[:2])  # 너무 길면 2개로 줄이기
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

    for _ in range(3):  # 최대 3번까지 이어붙이기 시도
        response = llm(remaining_prompt)
        full_output += response

        if len(response.strip()) < 1800:  # 적당한 길이면 종료
            break
        remaining_prompt = "계속 이어서 작성해줘.\n"

    return full_output.strip()


# ---- 슬래시 명령어 대응용 엔드포인트 (/blog 후기요청)
@app.post("/slack/slash")
async def slack_slash_command(request: Request):
    form = await request.form()
    user_text = form.get("text")
    response_url = form.get("response_url")

    # 즉시 "작업 중" 메시지 보내기
    ack_text = {
        "response_type": "ephemeral",
        "text": "후기를 생성 중입니다. 잠시만 기다려 주세요 ⏳"
    }

    # 백그라운드 작업 실행
    import threading

    def generate_and_send():
        docs = retriever.get_relevant_documents("블로그 스타일")
        past_texts = [doc.page_content for doc in docs]
        result = generate_post(user_text, past_texts)

        # Slack 메시지 4000자 제한 대비
        MAX_LEN = 3900
        chunks = [result[i:i+MAX_LEN] for i in range(0, len(result), MAX_LEN)]

        for idx, chunk in enumerate(chunks):
            prefix = "📝 블로그 후기 초안:\n\n" if idx == 0 else ""
            requests.post(response_url, json={
                "response_type": "in_channel",
                "text": prefix + chunk
            })


    threading.Thread(target=generate_and_send).start() # 후기 생성 별도 스레드 사용

    # 3초 내에 바로 응답
    return JSONResponse(content=ack_text)
