import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies.db import get_db
from app.config.rag_config import RAGConfig
from app.embeddings import bge_m3
from app.generation.context import ContextBuilder
from app.generation.generator import LLMGenerator
from app.models.knowledge_base import KnowledgeBase
from app.observability.langfuse_tracer import get_tracer
from app.reranking import jina
from app.retrieval.hybrid import HybridRetriever
from app.schemas.query import QueryRequest, QueryResponse, SourceResponse

router = APIRouter(prefix="/knowledge-bases", tags=["query"])

# Module-level singletons (lazy-initialised on first request)
_retriever: HybridRetriever | None = None
_generator: LLMGenerator | None = None


def _get_retriever() -> HybridRetriever:
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever()
    return _retriever


def _get_generator() -> LLMGenerator:
    global _generator
    if _generator is None:
        _generator = LLMGenerator()
    return _generator


@router.post("/{kb_id}/query", response_model=QueryResponse)
async def query_knowledge_base(
    kb_id: uuid.UUID,
    body: QueryRequest,
    db: AsyncSession = Depends(get_db),
):
    kb = await db.get(KnowledgeBase, kb_id)
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")

    rag_config = RAGConfig.from_dict(kb.rag_config)
    tracer = get_tracer()

    with tracer.new_trace("rag_query", metadata={"kb_id": str(kb_id), "query": body.query}) as trace:

        # 1. Embed query
        with tracer.span(trace, "embed_query"):
            query_vector = await bge_m3.embed_query(
                body.query,
                model_name=rag_config.embedding.model,
                device=rag_config.embedding.device,
            )

        # 2. Hybrid retrieval
        with tracer.span(trace, "hybrid_retrieval", input_data=body.query):
            retriever = _get_retriever()
            candidates = await retriever.retrieve(
                query=body.query,
                query_vector=query_vector,
                knowledge_base_id=kb_id,
                tenant_id=kb.tenant_id,
                rag_config=rag_config,
                extra_filters=body.filters or None,
            )

        # 3. Reranking
        if rag_config.reranking.enabled and candidates:
            with tracer.span(trace, "reranking", metadata={"candidates": len(candidates)}):
                reranked = await jina.rerank(
                    query=body.query,
                    chunks=candidates,
                    top_k=body.top_k or rag_config.reranking.top_k,
                    model_name=rag_config.reranking.model,
                    device=rag_config.reranking.device,
                )
        else:
            reranked = candidates[: (body.top_k or rag_config.reranking.top_k)]

        # 4. Context building
        with tracer.span(trace, "context_building"):
            ctx_builder = ContextBuilder(
                max_context_tokens=rag_config.generation.max_context_tokens
            )
            context_str, sources = ctx_builder.build(reranked)

        # 5. LLM generation
        with tracer.span(trace, "generation", input_data=body.query):
            gen = _get_generator()
            result = await gen.generate(
                query=body.query,
                context_str=context_str,
                sources=sources,
                rag_config=rag_config,
            )

    return QueryResponse(
        query=body.query,
        answer=result.answer,
        sources=[SourceResponse(**s) for s in result.sources] if body.include_sources else [],
        prompt_version=result.prompt_version,
        model=result.model,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
    )
