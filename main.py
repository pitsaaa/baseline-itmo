import time
from typing import List
import httpx

from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import HttpUrl
from schemas.request import PredictionRequest, PredictionResponse
from utils.logger import setup_logger
import re

# Initialize
app = FastAPI()
logger = None


@app.on_event("startup")
async def startup_event():
    global logger
    logger = await setup_logger()


# Функция для извлечения первого числа из строки
def extract_first_number(text: str) -> int:
    # Используем регулярное выражение для поиска чисел в строке
    match = re.search(r'\d+', text)  # \d+ ищет одно или несколько цифр
    if match:
        return int(match.group())  # Возвращаем первое найденное число как целое
    else:
        return None

api_key = 't1.9euelZqQysqZmZaZnJ6Uy82dy5fOl-3rnpWanJTGmp3Jzc-OmMyOypvMnYnl8_d7PAtD-e9LMnw6_t3z9ztrCEP570syfDr-zef1656VmsaYz5yZls6el43MyJyMyJmP7_zF656VmsaYz5yZls6el43MyJyMyJmP.AbQP6UTNyCUAsZZUGVb3TJqrqllUN-RFkzY2MRXRpDpojLSjxBxayOqnnXBA9TRe3MOerfIc4bDoNkUJAPEQCA'


async def get_gpt_response(prompt: str) -> int:
    url = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"

    # Формируем тело запроса
    payload = {
        "modelUri": f"gpt://b1ggutiddi1m04mbd5rm/yandexgpt-lite",  # Используем идентификатор каталога
        "completionOptions": {
            "stream": False,
            "temperature": 0.6,
            "maxTokens": "2000"
        },
        "messages": [
            {
                "role": "system",
                "text": "Напиши ответ - одно конкретное число"
            },
            {
                "role": "user",
                "text": prompt
            }
        ]
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    async with httpx.AsyncClient() as client:
        # Отправляем POST-запрос
        response = await client.post(url, json=payload, headers=headers)

        # Проверяем статус ответа
        if response.status_code == 200:
            response_data = response.json()
            # Извлекаем сгенерированный текст
            answer_ = response_data.get("result", {}).get("alternatives", [{}])[0].get("message", {}).get("text", "Ответ не найден.")
            return extract_first_number(answer_)
        else:
            return f"Ошибка запроса: {response.status_code}"


async def get_gpt_full_response(prompt: str) -> str:
    url = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"

    # Формируем тело запроса
    payload = {
        "modelUri": f"gpt://b1ggutiddi1m04mbd5rm/yandexgpt-lite",  # Используем идентификатор каталога
        "completionOptions": {
            "stream": False,
            "temperature": 0.6,
            "maxTokens": "2000"
        },
        "messages": [
            {
                "role": "system",
                "text": "Дай мне ответ в 2 предложения текстом"
            },
            {
                "role": "user",
                "text": prompt
            }
        ]
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    async with httpx.AsyncClient() as client:
        # Отправляем POST-запрос
        response = await client.post(url, json=payload, headers=headers)

        # Проверяем статус ответа
        if response.status_code == 200:
            response_data = response.json()
            # Извлекаем сгенерированный текст
            return response_data.get("result", {}).get("alternatives", [{}])[0].get("message", {}).get("text", "Ответ не найден.")
        else:
            return f"Ошибка запроса: {response.status_code}"


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()

    body = await request.body()
    await logger.info(
        f"Incoming request: {request.method} {request.url}\n"
        f"Request body: {body.decode()}"
    )

    response = await call_next(request)
    process_time = time.time() - start_time

    response_body = b""
    async for chunk in response.body_iterator:
        response_body += chunk

    await logger.info(
        f"Request completed: {request.method} {request.url}\n"
        f"Status: {response.status_code}\n"
        f"Response body: {response_body.decode()}\n"
        f"Duration: {process_time:.3f}s"
    )

    return Response(
        content=response_body,
        status_code=response.status_code,
        headers=dict(response.headers),
        media_type=response.media_type,
    )


@app.post("/api/request", response_model=PredictionResponse)
async def predict(body: PredictionRequest):
    try:
        await logger.info(f"Processing prediction request with id: {body.id}")

        # Здесь будет вызов вашей модели
        answer = await get_gpt_response(body.query)

        sources: List[HttpUrl] = [
            HttpUrl("https://itmo.ru/ru/"),
            HttpUrl("https://abit.itmo.ru/"),
        ]

        reasoning = await get_gpt_full_response(body.query)

        response = PredictionResponse(
            id=body.id,
            answer=answer,
            reasoning= reasoning,
            sources=sources,
        )
        await logger.info(f"Successfully processed request {body.id}")
        return response
    except ValueError as e:
        error_msg = str(e)
        await logger.error(f"Validation error for request {body.id}: {error_msg}")
        raise HTTPException(status_code=400, detail=error_msg)
    except Exception as e:
        await logger.error(f"Internal error processing request {body.id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")
