from __future__ import annotations

from dataclasses import dataclass

import structlog
from openai import AsyncOpenAI

from app.config.rag_config import RAGConfig
from app.config.settings import get_settings
from app.generation.prompts.qa_v1 import get_prompt
from app.retrieval.filters import RetrievedChunk

logger = structlog.get_logger(__name__)


@dataclass
class GenerationResult:
    answer: str
    sources: list[dict]
    prompt_version: str
    model: str
    input_tokens: int
    output_tokens: int


class LLMGenerator:
    """
    Generates answers using an OpenAI-compatible chat completion API.
    Receives pre-built context + sources from ContextBuilder.
    """

    def __init__(self) -> None:
        settings = get_settings()
        self._client = AsyncOpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
        )

    async def generate(
        self,
        query: str,
        context_str: str,
        sources: list[dict],
        rag_config: RAGConfig,
    ) -> GenerationResult:
        prompt_template = get_prompt(rag_config.prompt.version)
        system_msg = prompt_template.system_prompt()
        user_msg = prompt_template.user_prompt(question=query, context=context_str)

        gen_cfg = rag_config.generation
        log = logger.bind(model=gen_cfg.model, prompt_version=rag_config.prompt.version)
        log.info("calling llm")

        response = await self._client.chat.completions.create(
            model=gen_cfg.model,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=gen_cfg.temperature,
            max_tokens=gen_cfg.max_tokens,
        )

        answer = response.choices[0].message.content or ""
        usage = response.usage

        return GenerationResult(
            answer=answer,
            sources=sources,
            prompt_version=rag_config.prompt.version,
            model=gen_cfg.model,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
        )
