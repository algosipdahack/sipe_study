from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import chromadb
from openai import OpenAI
import os
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# 초기화
app = FastAPI()
model = SentenceTransformer("BAAI/bge-small-en")
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="rag_knowledge")
client = OpenAI()

# 입력 목록
class TextInput(BaseModel):
    text: str

# 1. /embed - 텍스트 저장
@app.post("/embed")
def embed(input: TextInput):
    text = input.text.strip()
    if not text:
        return {"error": "빈 텍스트는 저장할 수 없습니다."}

    vector = model.encode(text).tolist()
    collection.add(
        documents=[text],
        embeddings=[vector],
        ids=[f"doc_{collection.count() + 1}"]
    )
    return {"message": "저장 완료", "text": text}

# 2. 의도 분류 함수
def classify_intent(text: str) -> str:
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "다음 문장이 어느 의도인지 분류해줘. 결과는 \"정의질문\", \"비교질문\", \"요약요청\", \"추천요청\", \"기타\" 중 하나만 말해."},
            {"role": "user", "content": text}
        ],
        temperature=0
    )
    return response.choices[0].message.content.strip()

# 3. GPT 응답 생성 함수 (RAG)
def generate_answer(query: str, context: str) -> str:
    prompt = f"""
        답변 수행하기 위해, 다음 문서를 참고해서 자세하고 정확한 답변을 제공해줘.

        [문서]
        {context}

        [질문]
        {query}
        """
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "답변을 제공하는 AI 창작자입니다."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.5
    )
    return response.choices[0].message.content.strip()

# 4. /rag - 질의 → 의도 분류 → 검색 → 응답 생성
@app.post("/rag")
def rag(input: TextInput, top_k: int = 3):
    query = input.text.strip()
    if not query:
        return {"error": "질문이 비어있을 수 없습니다."}

    # (1) 의도 분류
    intent = classify_intent(query)

    # (2) 벡터 검색
    query_vector = model.encode(query).tolist()
    results = collection.query(query_embeddings=[query_vector], n_results=top_k)
    contexts = results["documents"][0]
    context_text = "\n".join(contexts)
    print(context_text)

    # (3) GPT 응답 생성
    answer = generate_answer(query, context_text)

    return {
        "query": query,
        "intent": intent,
        "context_count": len(contexts),
        "answer": answer
    }
