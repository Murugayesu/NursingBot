"""
LCEL (LangChain Expression Language) RAG chain.

Replaces the custom LLMGenerator, ContextBuilder, and prompt template files.

Chain design:
    {context: str, question: str}
    → ChatPromptTemplate
    → ChatOpenAI
    → StrOutputParser
    → str (answer)

The context string is built *before* invoking the chain so that
sources can be extracted from the reranked documents and returned
alongside the answer in the API response.
"""
from __future__ import annotations

from typing import Any

import structlog
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI

from app.config.rag_config import RAGConfig
from app.config.settings import get_settings

logger = structlog.get_logger(__name__)

# ── Prompt templates ──────────────────────────────────────────────────────────

_SYSTEM_V1 = (
    "You are a precise, helpful assistant that answers questions strictly "
    "based on the provided context.\n\n"
    "Rules:\n"
    "1. Use ONLY the information in the context. Do NOT fabricate facts.\n"
    "2. Cite sources using their index number, e.g. [1], [2].\n"
    "3. If the context does not contain enough information say exactly: "
    "\"I don't have enough information in the provided context to answer this question.\"\n"
    "4. Be concise and direct. Prefer bullet points for multi-part answers."
)

_SYSTEM_V2 = (
    "You are an expert assistant. Answer questions using only the provided context. "
    "Structure your response as:\n"
    "**Answer**: <your answer with inline citations [N]>\n"
    "**Sources used**: <list the [N] citations you referenced>\n\n"
    "If context is insufficient, respond: \"Insufficient context to answer.\"\n"
    "Never fabricate information."
)

_PROMPT_REGISTRY: dict[str, ChatPromptTemplate] = {
    "qa_v1": ChatPromptTemplate.from_messages([
        ("system", _SYSTEM_V1),
        ("human", "Context:\n\n{context}\n\n---\n\nQuestion: {question}\n\nAnswer:"),
    ]),
    "qa_v2": ChatPromptTemplate.from_messages([
        ("system", _SYSTEM_V2),
        ("human", "Context:\n\n{context}\n\n---\n\nQuestion: {question}"),
    ]),
}


def get_prompt(version: str) -> ChatPromptTemplate:
    if version not in _PROMPT_REGISTRY:
        raise ValueError(
            f"Unknown prompt version: '{version}'. Available: {list(_PROMPT_REGISTRY)}"
        )
    return _PROMPT_REGISTRY[version]


# ── Context formatting ────────────────────────────────────────────────────────

def format_docs_for_context(docs: list) -> tuple[str, list[dict]]:
    """
    Format reranked LangChain Documents into a context string + sources list.
    Returns (context_str, sources) where sources is a list of dicts
    for inclusion in the API response.
    """
    parts: list[str] = []
    sources: list[dict] = []

    for i, doc in enumerate(docs, start=1):
        meta = doc.metadata
        title = meta.get("title", "Source")
        url = meta.get("source_url") or meta.get("url")
        header = f"[{i}] {title}"
        if url:
            header += f" ({url})"
        parts.append(f"{header}\n{doc.page_content}")
        sources.append({
            "index": i,
            "document_id": meta.get("document_id", ""),
            "title": title,
            "source_url": url,
            "chunk_index": meta.get("chunk_index", 0),
            "score": meta.get("reranker_score", meta.get("score", 0.0)),
        })

    return "\n\n---\n\n".join(parts), sources


# ── Chain factory ─────────────────────────────────────────────────────────────

_llm_cache: dict[str, ChatOpenAI] = {}


def get_llm(model: str, temperature: float, max_tokens: int) -> ChatOpenAI:
    """Return a cached ChatOpenAI instance."""
    key = f"{model}:{temperature}:{max_tokens}"
    if key not in _llm_cache:
        settings = get_settings()
        api_key = settings.openai_api_key or "not-needed"
        _llm_cache[key] = ChatOpenAI(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
            base_url=settings.openai_base_url or None,
        )
    return _llm_cache[key]


def build_chain(rag_config: RAGConfig) -> Runnable:
    """
    Build and return an LCEL chain:
        {context: str, question: str} → str (answer)

    Invoke with:
        answer = await chain.ainvoke(
            {"context": ctx, "question": query},
            config={"callbacks": [langfuse_handler]},
        )
    """
    prompt = get_prompt(rag_config.prompt.version)
    gen = rag_config.generation
    llm = get_llm(gen.model, gen.temperature, gen.max_tokens)
    return prompt | llm | StrOutputParser()

