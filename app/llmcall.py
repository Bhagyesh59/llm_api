from fastapi import HTTPException, Depends
from app.models import ChatCompletionRequest, Message
from fastapi.security.api_key import APIKeyHeader
from dotenv import load_dotenv
from openai import OpenAI
from typing import List
import asyncio
import json
import os


load_dotenv()

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

API_KEY = "1234"  # Replace with your actual API key
API_KEY_NAME = "Authorization"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)


def get_groq_client():
    client = OpenAI(api_key=GROQ_API_KEY, base_url=GROQ_BASE_URL)
    return client


def verify_api_key(api_key: str = Depends(api_key_header)):
    if api_key is None:
        print("API key is missing")
        raise HTTPException(status_code=403, detail="API key is missing")
    if api_key != f"Bearer {API_KEY}":
        print(f"Invalid API key: {api_key}")
        raise HTTPException(status_code=403, detail="Could not validate API key")
    print(f"API key validated: {api_key}")


async def _resp_async_generator(
    messages: List[Message], model: str, max_tokens: int, temperature: float
):
    async with get_groq_client() as client:
        response = await client.post(
            "/chat/completions",
            json={
                "model": model,
                "messages": [{"role": m.role, "content": m.content} for m in messages],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": True,
            },
        )

        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=response.text)

        # Stream the response content
        async for chunk in response.aiter_text():
            chunk_data = json.loads(chunk)
            yield f"data: {json.dumps(chunk_data)}\n\n"
            await asyncio.sleep(0.01)  # Small delay to simulate streaming behavior
        yield "data: [DONE]\n\n"


def not_streaming(messages):
    with get_groq_client() as client:
        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=messages,
            temperature=0.7,
            max_tokens=512,
            stream=False,
        )
        print(response.choices[0].message.content)
        return response
