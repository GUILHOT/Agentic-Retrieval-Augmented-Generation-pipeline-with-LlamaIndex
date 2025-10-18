# Local RAG Assistant with Ollama & LlamaIndex

A 100% offline, secure RAG system using **Ollama (Phi-3 Mini)** and **LlamaIndex**, designed for sensitive enterprise documents (HR, finance, legal).

## ✨ Features

- Upload PDFs/txt → ask questions with source-grounded answers
- Zero cloud dependency → no API keys, no data leakage
- Built for compliance and data governance

## 🛠️ Tech Stack

Python, Streamlit, LlamaIndex, Ollama, Hugging Face Embeddings, PyPDF

## ▶️ How to Run

```bash
ollama pull phi3:mini
pip install -r requirements.txt
streamlit run chat_app.py
```

![RAG in action](screenshot.png)
