# GenAI Knowledge Assistant (RAG demo)

A Streamlit app that chunks uploaded TXT/Markdown documents, retrieves relevant passages with TF-IDF, and optionally asks a local Ollama model to generate a grounded answer.

## Run locally
1. Install Python 3.10+.
2. In this folder run:
   ```bash
   python -m venv .venv
   # Windows: .venv\\Scripts\\activate
   # macOS/Linux: source .venv/bin/activate
   pip install -r requirements.txt
   streamlit run app.py
   ```
3. Upload TXT or Markdown files and ask a question.
4. Retrieval-only mode works without an LLM. For answer generation, install Ollama, then run `ollama pull llama3.2` and ensure the Ollama service is running. Enable the Ollama checkbox in the app.

## Components
- Text chunking with overlap.
- TF-IDF similarity retrieval for context passages.
- Optional local Ollama LLM answer generation grounded in retrieved passages.

## Limitations
This is a learning prototype; TF-IDF retrieval is lexical rather than embedding-based, and the optional LLM may produce errors. Verify answers against source passages.
