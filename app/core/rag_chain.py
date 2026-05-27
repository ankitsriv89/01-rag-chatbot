"""
RAG Chain
=========
Builds and runs the Retrieval-Augmented Generation pipeline.

CONCEPT — LCEL (LangChain Expression Language):
  LangChain uses the pipe operator (|) to chain steps:
    retriever | format_docs | prompt | llm | output_parser
  Each step's output becomes the next step's input.

CONCEPT — Fallback models:
  Groq has per-model rate limits on pay-as-you-go.
  We try the primary model first; on RateLimitError or APIError
  we iterate through groq_fallback_models until one succeeds.
  This is standard production practice — never let one model's outage
  take down your whole app.

CONCEPT — System vs Human prompt:
  System prompt: sets the LLM's persona and strict rules (unchanged per request).
  Human prompt:  the user's question + retrieved context (changes every request).
"""

from typing import AsyncGenerator, List
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from groq import RateLimitError, APIStatusError
from loguru import logger

from app.config import settings, LLMProvider


# ── System Prompt ──────────────────────────────────────────────────────────────
# Strictly grounds the LLM in provided context — prevents hallucination.
SYSTEM_PROMPT = """You are an expert document analyst and question-answering assistant.

Your job is to answer questions based STRICTLY on the provided context excerpts from the user's documents.

Rules you MUST follow:
1. ONLY use information present in the provided context. Never use prior knowledge.
2. If the context does not contain enough information to answer, say:
   "I couldn't find a clear answer to that in the uploaded documents."
3. Always cite which document/section your answer comes from when possible.
4. Be concise but complete. Prefer bullet points for multi-part answers.
5. If asked to summarize, cover all key points from the context.
6. Never make up facts, statistics, or quotes not present in the context.

Context from the uploaded documents:
{context}
"""

HUMAN_PROMPT = "Question: {question}"


def _make_groq_llm(model: str, streaming: bool) -> ChatGroq:
    """Create a ChatGroq instance for a given model name."""
    return ChatGroq(
        model=model,
        api_key=settings.groq_api_key,
        temperature=0.1,
        max_tokens=2048,
        streaming=streaming,
    )


def get_llm(streaming: bool = False):
    """
    Return the configured LLM. For Groq, returns the primary model.
    Fallback logic is handled separately in get_llm_with_fallback().
    """
    if settings.llm_provider == LLMProvider.OPENAI:
        logger.info(f"Using OpenAI LLM: {settings.openai_model}")
        return ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            temperature=0.1,
            max_tokens=2048,
            streaming=streaming,
        )

    logger.info(f"Using Groq LLM: {settings.groq_model}")
    return _make_groq_llm(settings.groq_model, streaming)


def get_groq_llm_with_fallback(streaming: bool = False):
    """
    Returns (llm, model_name) trying the primary model first, then the fallback chain.

    CONCEPT — Why return the model name?
      The caller needs to know which model actually answered so it can
      include that in the API response and logs.

    This function doesn't make any API calls — it just constructs the client.
    The actual fallback retry happens in invoke_with_fallback().
    """
    all_models = [settings.groq_model] + settings.groq_fallback_models
    return all_models, streaming


def _format_docs(docs) -> str:
    """
    Format retrieved Documents (no scores) into a readable context string
    for injection into the system prompt.
    """
    if not docs:
        return "No relevant context found in the uploaded documents."

    sections = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source_filename", "unknown")
        page = doc.metadata.get("page", "")
        page_str = f", page {int(page) + 1}" if page != "" else ""
        sections.append(
            f"[Excerpt {i} | Source: {source}{page_str}]\n"
            f"{doc.page_content.strip()}"
        )
    return "\n\n---\n\n".join(sections)


def _build_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT),
        HumanMessagePromptTemplate.from_template(HUMAN_PROMPT),
    ])


async def invoke_with_fallback(question: str, retriever) -> tuple[str, str]:
    """
    Run the RAG chain with automatic Groq model fallback.

    CONCEPT — Fallback execution flow:
      1. Try primary model (e.g. llama-3.3-70b-versatile)
      2. On RateLimitError or APIStatusError → log warning, try next model
      3. Repeat until one succeeds or all models exhausted
      4. If all fail → raise the last exception

    Args:
        question: The user's question.
        retriever: LangChain retriever from vector_store_manager.

    Returns:
        (answer_text, model_name_used)
    """
    if settings.llm_provider == LLMProvider.OPENAI:
        llm = get_llm(streaming=False)
        chain = (
            {"context": retriever | _format_docs, "question": RunnablePassthrough()}
            | _build_prompt()
            | llm
            | StrOutputParser()
        )
        answer = await chain.ainvoke(question)
        return answer, settings.openai_model

    # Groq with fallback chain
    all_models = [settings.groq_model] + settings.groq_fallback_models
    last_error = None

    for model in all_models:
        try:
            logger.info(f"Trying Groq model: {model}")
            llm = _make_groq_llm(model, streaming=False)
            chain = (
                {"context": retriever | _format_docs, "question": RunnablePassthrough()}
                | _build_prompt()
                | llm
                | StrOutputParser()
            )
            answer = await chain.ainvoke(question)
            if model != settings.groq_model:
                logger.info(f"Succeeded with fallback model: {model}")
            return answer, model

        except RateLimitError as e:
            logger.warning(f"Rate limit on '{model}': {e}. Trying next model...")
            last_error = e
        except APIStatusError as e:
            logger.warning(f"API error on '{model}': {e}. Trying next model...")
            last_error = e

    logger.error("All Groq models exhausted. Raising last error.")
    raise last_error or RuntimeError("All Groq models failed with no specific error.")


async def stream_with_fallback(question: str, retriever) -> AsyncGenerator[tuple[str, str], None]:
    """
    Stream tokens with Groq fallback. Yields (token, model_name) tuples.

    For streaming, we cannot retry mid-stream — so we first identify which
    model is available by doing a tiny non-streaming probe, then stream with that model.

    CONCEPT — Probe before stream:
      Streaming starts immediately and can't be rewound.
      We detect the working model upfront (cheap 1-token probe),
      then open the full streaming connection.

    Yields:
        (token_string, model_name) — model_name only on the first token.
    """
    if settings.llm_provider == LLMProvider.OPENAI:
        llm = get_llm(streaming=True)
        chain = (
            {"context": retriever | _format_docs, "question": RunnablePassthrough()}
            | _build_prompt()
            | llm
            | StrOutputParser()
        )
        async for token in chain.astream(question):
            yield token, settings.openai_model
        return

    # Find the first working Groq model via non-streaming probe
    all_models = [settings.groq_model] + settings.groq_fallback_models
    working_model = None
    last_error = None

    for model in all_models:
        try:
            probe_llm = _make_groq_llm(model, streaming=False)
            # Minimal probe — just check connectivity with a trivial prompt
            await probe_llm.ainvoke("Reply with the single word: ok")
            working_model = model
            logger.info(f"Probe succeeded — streaming with: {model}")
            break
        except RateLimitError as e:
            logger.warning(f"Rate limit on '{model}' during probe. Trying next...")
            last_error = e
        except APIStatusError as e:
            logger.warning(f"API error on '{model}' during probe. Trying next...")
            last_error = e

    if working_model is None:
        raise last_error or RuntimeError("All Groq models failed during probe.")

    # Stream with the confirmed working model
    stream_llm = _make_groq_llm(working_model, streaming=True)
    chain = (
        {"context": retriever | _format_docs, "question": RunnablePassthrough()}
        | _build_prompt()
        | stream_llm
        | StrOutputParser()
    )

    async for token in chain.astream(question):
        yield token, working_model
