"""
chat_app.py
100% local: Ollama (requests) + LlamaIndex (retriever) + HuggingFace embeddings + Streamlit UI
No 'openai' package required. No API keys.
"""

import os
import json
import requests
from pathlib import Path
import streamlit as st

# LlamaIndex imports (new-style)
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# For PDFs/images OCR if you want to extend later (optional)
# import fitz       # PyMuPDF
# import pytesseract
# from PIL import Image

# ----------------------------
# Safety: remove any OPENAI env var so nothing accidentally picks it up
# ----------------------------
os.environ.pop("OPENAI_API_KEY", None)
os.environ.pop("OPENAI_API_KEY", "")

# ----------------------------
# Config
# ----------------------------
OLLAMA_BASE_URL = "http://localhost:11434"  # default Ollama server
OLLAMA_CHAT_ENDPOINT = f"{OLLAMA_BASE_URL}/v1/chat/completions"
DEFAULT_MODEL = "phi3:mini"
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# ----------------------------
# Ollama client (pure requests)
# ----------------------------
def ollama_chat(messages, model=DEFAULT_MODEL, stream: bool = False, timeout: int = 300):
    """
    messages: list of {"role": "user"/"system"/"assistant", "content": "..."}
    If stream=True, yields text chunks (as they arrive).
    If stream=False, returns the full content string.
    """
    payload = {"model": model, "messages": messages, "stream": stream}
    headers = {"Content-Type": "application/json"}

    try:
        resp = requests.post(OLLAMA_CHAT_ENDPOINT, headers=headers, json=payload, stream=stream, timeout=timeout)
        resp.raise_for_status()
    except Exception as e:
        # Raise a clear error for Streamlit display
        raise RuntimeError(f"Error contacting Ollama at {OLLAMA_CHAT_ENDPOINT}: {e}")

    if stream:
        # Ollama uses server-sent-event style lines starting with "data: "
        for raw_line in resp.iter_lines(decode_unicode=True):
            if not raw_line:
                continue
            # lines look like: b'data: {"id":...}'
            line = raw_line.strip()
            if line.startswith("data:"):
                line = line[len("data:"):].strip()
            # Skip non-json markers like [DONE]
            if not line or line == "[DONE]":
                continue
            try:
                chunk = json.loads(line)
                # chunk structure: {"choices":[{"delta": {"content": "..."} }], ...}
                delta = chunk.get("choices", [])[0].get("delta", {})
                content_piece = delta.get("content")
                if content_piece:
                    yield content_piece
            except json.JSONDecodeError:
                # ignore malformatted lines
                continue
        return
    else:
        data = resp.json()
        # non-stream response structure: {"choices":[{"message": {"role":"assistant","content": "..."} }]}
        try:
            return data["choices"][0]["message"]["content"]
        except Exception:
            # Fallback: return JSON as string for debugging
            return json.dumps(data, indent=2)


# ----------------------------
# Build / load index (retriever-only)
# ----------------------------
def build_index(data_dir: Path = DATA_DIR, embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """
    Builds a VectorStoreIndex from files in data_dir.
    This function explicitly sets Settings.embed_model (and crucially does not set any LLM in Settings).
    """
    # Configure HuggingFace embedding (local)
    Settings.embed_model = HuggingFaceEmbedding(model_name=embed_model_name, device="cpu")

    # Use SimpleDirectoryReader to load files (txt, md, pdf, etc.)
    # SimpleDirectoryReader handles many file types. If you want custom parsing, replace this.
    reader = SimpleDirectoryReader(str(data_dir))
    docs = reader.load_data()
    if not docs:
        return None

    index = VectorStoreIndex.from_documents(docs)
    return index


# ----------------------------
# Helper: robust node -> text extractor
# ----------------------------
def node_to_text(node) -> str:
    """
    LlamaIndex node structures vary slightly by version. Try several access patterns.
    """
    # Node could be Document-like with .get_content()
    try:
        return node.get_content()
    except Exception:
        pass
    # Node may be NodeWithScore: node.node.get_content()
    try:
        inner = getattr(node, "node", None)
        if inner is not None and hasattr(inner, "get_content"):
            return inner.get_content()
    except Exception:
        pass
    # Try 'text' or 'get_text' or str()
    try:
        return getattr(node, "text", None) or getattr(node, "get_text", lambda: None)() or str(node)
    except Exception:
        return str(node)


# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Local PDF Q&A (Ollama only)", layout="wide")
st.title("📚 Local PDF Q&A — Ollama only (no OpenAI)")

# Left panel: upload + index controls
with st.sidebar:
    st.header("Index / Data")
    st.markdown("Drop PDFs / text files here. Files are saved into `./data/` and the index is (re)built.")
    uploaded = st.file_uploader("Upload files (pdf, txt, md, png/jpg)", accept_multiple_files=True)

    if uploaded:
        saved = []
        for f in uploaded:
            dest = DATA_DIR / f.name
            with open(dest, "wb") as out:
                out.write(f.getbuffer())
            saved.append(f.name)
        st.success(f"Saved {len(saved)} files: {', '.join(saved)}")
        # Force rebuild: clear session index so main flow rebuilds below
        if "index" in st.session_state:
            st.session_state.pop("index")

    if st.button("Rebuild index now"):
        if "index" in st.session_state:
            st.session_state.pop("index")
        st.experimental_rerun()

    st.write("---")
    st.markdown("Ollama endpoint:")
    st.text(OLLAMA_CHAT_ENDPOINT)
    st.markdown("Model:")
    model_choice = st.text_input("Model name", value=DEFAULT_MODEL)
    if model_choice:
        DEFAULT_MODEL = model_choice  # update local var for this run

# Prepare index in session state
if "index" not in st.session_state:
    with st.spinner("Building index from ./data (this may take a bit the first time)..."):
        idx = build_index(DATA_DIR)
        if idx is None:
            st.warning("No documents found in ./data/. Upload PDFs or text files in the sidebar.")
            st.session_state["index"] = None
        else:
            st.session_state["index"] = idx
            st.success("Index built.")

index = st.session_state.get("index")
retriever = None
if index:
    try:
        retriever = index.as_retriever(similarity_top_k=3)
    except Exception as e:
        st.error(f"Error creating retriever from index: {e}")
        retriever = None

# Chat state
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Show previous conversation
for m in st.session_state["messages"]:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Input box
prompt = st.chat_input("Ask a question about your documents (PDFs/texts) ...")

if prompt:
    # record user message
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Retrieve context using the retriever-only pipeline
    retrieved_text = ""
    if retriever:
        try:
            nodes = retriever.retrieve(prompt)
            # nodes may be a list of NodeWithScore or Document objects - extract text robustly
            pieces = []
            for n in nodes:
                try:
                    t = node_to_text(n)
                    if t and t.strip():
                        pieces.append(t.strip())
                except Exception:
                    continue
            # join a few top pieces
            retrieved_text = "\n\n".join(pieces[:6])
        except Exception as e:
            st.error(f"Retrieval error: {e}")
            retrieved_text = ""
    else:
        st.info("No index/retriever available — upload files and rebuild the index from the sidebar.")

    # Create final prompt sent to Ollama (system + user, inject context)
    system_msg = {
        "role": "system",
        "content": "You are a helpful assistant. Use the provided document context to answer the user's question. If the answer is not present, say you don't know."
    }

    user_with_context = {
        "role": "user",
        "content": f"Context:\n\n{retrieved_text}\n\n\nQuestion: {prompt}"
    }

    # Stream the Ollama response to the UI
    with st.chat_message("assistant"):
        placeholder = st.empty()
        assistant_text = ""
        try:
            for chunk in ollama_chat([system_msg, user_with_context], model=DEFAULT_MODEL, stream=True):
                assistant_text += chunk
                placeholder.markdown(assistant_text + "▌")
            placeholder.markdown(assistant_text)
        except Exception as e:
            st.error(f"Ollama error: {e}")
            assistant_text = f"(error) {e}"
            placeholder.markdown(assistant_text)

    # Save assistant response in session history
    st.session_state["messages"].append({"role": "assistant", "content": assistant_text})





