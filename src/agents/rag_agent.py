"""
RAG pipeline — manual LangGraph StateGraph.

Why not create_agent + ToolStrategy?
-------------------------------------
Anthropic's API raises "When specifying tool_choice, you must provide exactly
one tool" when the agent has BOTH a retrieval tool AND the structured-output
tool that ToolStrategy injects.  The create_agent abstraction doesn't expose
a way to suppress tool_choice, so we build the graph by hand instead.

Architecture
------------
  START
    │
    ▼
  [retrieve]   – pure Python: FAISS similarity search, no LLM tool call
    │
    ▼
  [generate]   – model.with_structured_output(RAGResponse) called once,
                 context injected into the system prompt
    │
    ▼
  END

Short-term memory
-----------------
The graph is compiled with the session's InMemorySaver checkpointer.
thread_id = session_id ensures conversation history is isolated per session
and persists across multiple /chat calls exactly as shown in the LangChain
short-term memory docs.

Structured output
-----------------
RAGResponse is a TypedDict (no Pydantic).  with_structured_output uses
provider-native structured output (works with Claude / OpenAI / HuggingFace
that support it) — no extra tool is registered so the tool_choice conflict
cannot occur.
"""

import logging
from typing import Any, List, Sequence

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict

from src.config.config import settings
from src.embeddings.llm_model import get_chat_model
from src.memory.session_registry import SessionMeta, get_session
from src.schemas.responses import RAGResponse
from src.vectorstore.session_store import get_session_retriever

logger = logging.getLogger(__name__)


# ── Graph state ───────────────────────────────────────────────────────────────


class GraphState(TypedDict):
    # add_messages reducer: new messages are appended, not replaced
    messages: Annotated[Sequence[BaseMessage], add_messages]
    context: str  # retrieved chunks injected at generate step
    response: RAGResponse  # final structured output


# ── System prompt ─────────────────────────────────────────────────────────────

_SYSTEM = (
    "You are a precise document assistant (Insurance Domain).\n\n"
    "First determine the intent of the user's message.\n"
    "1. If the message is a greeting "
    "(e.g., 'hi', 'hello', 'thanks', 'good morning'), respond naturally and politely. "
    "Do NOT reference the context. Leave the sources list empty.\n\n"
    "2. If the message is a question, answer ONLY using the information "
    "from the provided CONTEXT.\n\n"
    "Rules for answering questions:\n"
    "- Use only the information in the CONTEXT.\n"
    "- Do NOT use outside knowledge.\n"
    "- If the CONTEXT does not contain enough information, say that the answer "
    "is not available in the provided context.\n"
    "- If the question is unrelated to the CONTEXT, politely explain that you "
    "can only answer questions based on the provided documents.\n\n"
    "Sources:\n"
    "- Populate the 'sources' list only with documents actually used in the answer.\n"
    "- Do NOT invent sources.\n"
    "- Each relevant_excerpt must be under 200 characters.\n\n"
    "CONTEXT:\n{context}"
)


# ── Node: retrieve ────────────────────────────────────────────────────────────


def _make_retrieve_node(session_id: str):
    """
    Returns a node function that performs FAISS retrieval.
    Pure Python — no LLM call, no tools registered on the model.
    """

    def retrieve(state: GraphState) -> dict:
        # Use the last human message as the retrieval query
        last_human = next(
            (m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
            None,
        )
        query = last_human.content if last_human else ""

        retriever = get_session_retriever(session_id, top_k=settings.RETRIEVER_TOP_K)
        if retriever is None or not query:
            return {"context": "No documents available."}

        docs: List[Document] = retriever.invoke(query)
        if not docs:
            return {"context": "No relevant passages found for this query."}

        parts = []
        for i, doc in enumerate(docs, 1):
            src = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page_label", "")
            page_str = f" (page {page})" if page else ""
            parts.append(f"[{i}] Source: {src}{page_str}\n{doc.page_content.strip()}")

        context = "\n\n---\n\n".join(parts)
        logger.debug("Retrieved %d chunks for session %s", len(docs), session_id)
        return {"context": context}

    return retrieve


# ── Node: generate ────────────────────────────────────────────────────────────


def _make_generate_node():
    """
    Returns a node function that calls the model with structured output.
    with_structured_output does NOT register an extra tool — it uses
    provider-native JSON mode or response_format, so there is no tool_choice
    conflict.
    """

    def generate(state: GraphState) -> dict:
        model = get_chat_model()

        # Bind structured output schema (TypedDict, no Pydantic)
        structured_model = model.with_structured_output(RAGResponse)

        # Build the message list: system (with context) + full conversation history
        system_msg = SystemMessage(content=_SYSTEM.format(context=state["context"]))
        history = list(state["messages"])

        response: RAGResponse = structured_model.invoke([system_msg] + history)

        # Store the plain-text answer back into message history for memory
        ai_msg = AIMessage(content=response.get("answer", ""))
        return {
            "messages": [ai_msg],
            "response": response,
        }

    return generate


# ── Graph builder ─────────────────────────────────────────────────────────────


def _build_graph(session_id: str, checkpointer: InMemorySaver) -> Any:
    """
    Compile and return the LangGraph StateGraph for one session.
    The graph is cheap to build (no I/O) and is recreated per request so
    it always captures the latest session_id closure.
    """
    builder = StateGraph(GraphState)

    builder.add_node("retrieve", _make_retrieve_node(session_id))
    builder.add_node("generate", _make_generate_node())

    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "generate")
    builder.add_edge("generate", END)

    return builder.compile(checkpointer=checkpointer)


# ── Public interface ──────────────────────────────────────────────────────────


def run_rag_query(session_id: str, user_query: str) -> RAGResponse:
    """
    Run a RAG query for the given session and return a typed RAGResponse.

    The InMemorySaver checkpointer (stored on the session) persists the full
    message history across calls — short-term memory is automatic.

    Raises
    ------
    ValueError   – session not found / expired.
    RuntimeError – model returned no structured response.
    """
    session_meta = get_session(session_id)
    if session_meta is None:
        raise ValueError(f"Session '{session_id}' not found or has expired.")

    graph = _build_graph(session_id, session_meta.checkpointer)

    config: RunnableConfig = {
        "configurable": {
            "thread_id": session_id  # isolates conversation history per session
        }
    }

    result = graph.invoke(
        {"messages": [HumanMessage(content=user_query)]},
        config,
    )

    response: RAGResponse | None = result.get("response")

    if response is None:
        # Graceful fallback — pull the last AI message if structured output failed
        last_ai = next(
            (
                m
                for m in reversed(result.get("messages", []))
                if isinstance(m, AIMessage)
            ),
            None,
        )
        plain = last_ai.content if last_ai else "No answer generated."
        logger.warning(
            "No structured response for session %s — using plain text.", session_id
        )
        return RAGResponse(answer=plain, sources=[])

    return response
