"""
Gradio Frontend
===============
The user-facing chat interface deployed on Hugging Face Spaces.

CONCEPT — Why Gradio?
  Gradio turns Python functions into interactive web UIs with zero HTML/CSS/JS.
  It's the industry standard for ML demos and is natively supported on HF Spaces.
  Gradio 6.x uses a Blocks API that gives full layout control.

CONCEPT — How this frontend communicates with the FastAPI backend:
  Gradio runs as a separate process (different port).
  It calls our FastAPI endpoints via httpx (async HTTP client).
  For streaming, it reads Server-Sent Events from /api/v1/query
  and yields tokens progressively to the chat UI.

Architecture:
  Browser → Gradio (port 7860) → FastAPI (port 8000) → Groq API → Response
"""

import os
import json
import httpx
import gradio as gr
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
# In production (HF Spaces), set BACKEND_URL as a Space secret.
# Locally, it points to your running FastAPI server.
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
API_BASE = f"{BACKEND_URL}/api/v1"

# Timeout for non-streaming requests (upload, status)
REQUEST_TIMEOUT = 120.0


# ── Backend API Calls ─────────────────────────────────────────────────────────

def upload_document(file_obj) -> str:
    """
    Upload a document to the FastAPI backend for indexing.

    Args:
        file_obj: Gradio file object with .name (path) attribute.

    Returns:
        Status message string shown in the UI.
    """
    if file_obj is None:
        return "Please select a file first."

    file_path = Path(file_obj.name)
    filename = file_path.name

    try:
        with open(file_path, "rb") as f:
            files = {"file": (filename, f, "application/octet-stream")}
            response = httpx.post(
                f"{API_BASE}/upload",
                files=files,
                timeout=600.0,  # 10 min — large docs with local CPU embeddings can be slow
            )

        if response.status_code == 201:
            data = response.json()
            return (
                f"✅ **{data['filename']}** indexed successfully!\n\n"
                f"- Chunks created: **{data['chunks_created']}**\n"
                f"- Total chunks in store: **{data['total_chunks_in_store']}**\n\n"
                f"You can now ask questions about this document."
            )
        else:
            error = response.json().get("detail", "Unknown error")
            return f"❌ Upload failed: {error}"

    except httpx.ConnectError:
        return "❌ Cannot connect to backend. Is the FastAPI server running?"
    except Exception as e:
        return f"❌ Error: {str(e)}"


def get_status() -> str:
    """Fetch and display current system status from the backend."""
    try:
        response = httpx.get(f"{API_BASE}/status", timeout=10.0)
        data = response.json()

        ready_icon = "🟢" if data["is_ready"] else "🔴"
        return (
            f"{ready_icon} **System Status**\n\n"
            f"- Documents indexed: **{data['document_count']} chunks**\n"
            f"- LLM: **{data['llm_provider']} / {data['llm_model']}**\n"
            f"- Embeddings: **{data['embedding_model']}**\n"
            f"- Vector store: **{data['vector_store_type']}**\n"
            f"- Ready: **{data['is_ready']}**"
        )
    except httpx.ConnectError:
        return "🔴 **Backend offline** — start the FastAPI server."
    except Exception as e:
        return f"❌ Status check failed: {str(e)}"


def clear_documents() -> str:
    """Clear all indexed documents from the vector store."""
    try:
        response = httpx.delete(f"{API_BASE}/clear", timeout=10.0)
        if response.status_code == 200:
            return "🗑️ All documents cleared. Upload new documents to continue."
        return f"❌ Clear failed: {response.text}"
    except Exception as e:
        return f"❌ Error: {str(e)}"


def stream_chat(message: str, history: list):
    """
    Send a question to the backend and stream the response token-by-token.

    CONCEPT — Gradio 6.x streaming with type="messages":
      The function receives history as a list of {"role": ..., "content": ...} dicts.
      It must yield the FULL updated history list on each token — Gradio replaces
      the entire chatbot state, not appends. We append the user message and a
      growing bot message, yielding after each token arrives.

    Args:
        message: The user's typed question.
        history: List of {"role": "user"|"assistant", "content": str} dicts.

    Yields:
        Updated full history list after each streamed token.
    """
    if not message.strip():
        return

    # Add user message to history immediately
    history = history + [{"role": "user", "content": message}]
    history = history + [{"role": "assistant", "content": ""}]
    yield history

    payload = {"question": message, "stream": True}
    full_response = ""

    try:
        # httpx streaming context — reads SSE events as they arrive
        with httpx.stream(
            "POST",
            f"{API_BASE}/query",
            json=payload,
            timeout=120.0,
            headers={"Accept": "text/event-stream"},
        ) as response:

            if response.status_code == 400:
                error_data = json.loads(response.read())
                history[-1]["content"] = f"⚠️ {error_data.get('detail', 'No documents uploaded yet.')}"
                yield history
                return

            if response.status_code != 200:
                history[-1]["content"] = f"❌ Backend error (HTTP {response.status_code})"
                yield history
                return

            # Read SSE lines: "data: {json}\n\n"
            for line in response.iter_lines():
                if not line.startswith("data: "):
                    continue

                raw = line[len("data: "):]

                try:
                    event = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                if "error" in event:
                    history[-1]["content"] = f"❌ {event['error']}"
                    yield history
                    return

                if event.get("done"):
                    break

                token = event.get("token", "")
                full_response += token
                history[-1]["content"] = full_response
                yield history

    except httpx.ConnectError:
        history[-1]["content"] = "❌ Cannot connect to backend. Is the FastAPI server running on port 8000?"
        yield history
    except Exception as e:
        history[-1]["content"] = f"❌ Streaming error: {str(e)}"
        yield history


# ── Gradio UI Layout ──────────────────────────────────────────────────────────

DESCRIPTION = """
# 📄 Production RAG Chatbot

Upload your documents (PDF, DOCX, TXT) and ask questions about them.
Powered by **LLaMA 3.3 70B** via Groq + **LangChain** + **FAISS**.

### How to use:
1. Upload one or more documents in the **Documents** tab
2. Switch to **Chat** and ask questions
3. The bot answers using ONLY content from your documents
"""

with gr.Blocks(title="RAG Chatbot") as demo:

    gr.Markdown(DESCRIPTION)

    with gr.Tabs():

        # ── Tab 1: Document Upload ─────────────────────────────────────────
        with gr.Tab("📁 Documents"):
            gr.Markdown("### Upload Documents")
            gr.Markdown("Supported formats: PDF, DOCX, TXT (max 50MB each)")

            with gr.Row():
                file_input = gr.File(
                    label="Select document",
                    file_types=[".pdf", ".docx", ".txt"],
                    type="filepath",
                )

            with gr.Row():
                upload_btn = gr.Button("📤 Upload & Index", variant="primary", scale=2)
                clear_btn = gr.Button("🗑️ Clear All Documents", variant="secondary", scale=1)

            upload_status = gr.Markdown(label="Status")

            gr.Markdown("---")
            gr.Markdown("### System Status")
            status_btn = gr.Button("🔄 Refresh Status", size="sm")
            status_output = gr.Markdown()

            # Wire up button actions
            upload_btn.click(fn=upload_document, inputs=[file_input], outputs=[upload_status])
            clear_btn.click(fn=clear_documents, outputs=[upload_status])
            status_btn.click(fn=get_status, outputs=[status_output])

            # Auto-load status on tab open
            demo.load(fn=get_status, outputs=[status_output])

        # ── Tab 2: Chat Interface ──────────────────────────────────────────
        with gr.Tab("💬 Chat"):
            gr.Markdown("### Ask questions about your documents")
            gr.Markdown(
                "_Answers are grounded strictly in your uploaded documents. "
                "Sources are shown in the non-streaming API response._"
            )

            chatbot = gr.Chatbot(
                label="RAG Chatbot",
                height=500,
                show_label=True,
            )

            with gr.Row():
                msg_input = gr.Textbox(
                    label="Your question",
                    placeholder="e.g. What is the main topic of the document?",
                    scale=5,
                    autofocus=True,
                )
                send_btn = gr.Button("Send", variant="primary", scale=1)

            clear_chat_btn = gr.ClearButton([msg_input, chatbot], value="Clear Chat")

            # Gradio 6.x chat streaming pattern
            msg_input.submit(
                fn=stream_chat,
                inputs=[msg_input, chatbot],
                outputs=[chatbot],
            )
            send_btn.click(
                fn=stream_chat,
                inputs=[msg_input, chatbot],
                outputs=[chatbot],
            )

        # ── Tab 3: About ───────────────────────────────────────────────────
        with gr.Tab("ℹ️ About"):
            gr.Markdown("""
### Architecture

```
User → Gradio UI → FastAPI Backend → Groq LLM (LLaMA 3.3 70B)
                         ↓
               OpenAI Embeddings (text-embedding-3-small)
                         ↓
                    FAISS Vector Store
```

### RAG Pipeline Steps
1. **Ingest**: PDF/DOCX/TXT → text extraction → chunking (1000 chars, 200 overlap)
2. **Embed**: Each chunk → OpenAI embedding → 1536-dim vector
3. **Store**: Vectors stored in FAISS (in-memory) or Chroma (persistent)
4. **Retrieve**: Query embedded → top-4 similar chunks fetched via cosine similarity
5. **Generate**: Chunks + question sent to LLaMA 3.3 70B → answer streamed back

### Tech Stack
| Component | Technology |
|-----------|-----------|
| API Backend | FastAPI + Uvicorn |
| LLM | Groq (LLaMA 3.3 70B) / OpenAI (GPT-4o) |
| Embeddings | OpenAI text-embedding-3-small |
| Vector Store | FAISS / Chroma |
| Orchestration | LangChain 1.3 |
| Frontend | Gradio 6.x |
| Deployment | Hugging Face Spaces + Docker |
            """)


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        theme=gr.themes.Soft(primary_hue="blue"),
        css=".gradio-container { max-width: 900px !important; margin: auto; }",
    )
