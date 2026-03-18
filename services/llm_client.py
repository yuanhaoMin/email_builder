import os
from typing import Optional, Type, TypeVar

from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel

load_dotenv()

_CLIENT: Optional[AsyncOpenAI] = None

T = TypeVar("T", bound=BaseModel)


def _get_env(name: str, default: str = "") -> str:
    return (os.getenv(name) or default).strip()


def get_async_client() -> AsyncOpenAI:
    global _CLIENT

    if _CLIENT is not None:
        return _CLIENT

    api_key = _get_env("OPENAI_API_KEY")
    timeout_s = float(_get_env("OPENAI_TIMEOUT_SECONDS", "40"))

    if not api_key:
        raise RuntimeError(
            "Missing OPENAI_API_KEY. Please set it in your environment variables or .env file."
        )

    _CLIENT = AsyncOpenAI(api_key=api_key, timeout=timeout_s)
    return _CLIENT


async def generate_structured_output(
    prompt: str,
    model: str,
    output_model: Type[T],
    reasoning_effort: Optional[str] = None,
) -> tuple[Optional[T], str]:
    if not prompt or not prompt.strip():
        raise ValueError("Prompt must not be empty.")

    if not model or not model.strip():
        raise ValueError("Model must not be empty.")

    client = get_async_client()

    request_kwargs = {
        "model": model.strip(),
        "input": [{"role": "user", "content": prompt.strip()}],
        "text_format": output_model,
    }

    if reasoning_effort:
        request_kwargs["reasoning"] = {"effort": reasoning_effort}

    response = await client.responses.parse(**request_kwargs)

    raw_text = getattr(response, "output_text", "") or ""
    parsed = getattr(response, "output_parsed", None)

    return parsed, raw_text
