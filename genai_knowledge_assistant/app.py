import streamlit as st
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title='Knowledge Assistant', page_icon='📚')
st.title('📚 GenAI Knowledge Assistant')
st.caption('Ask questions about text documents using retrieval and an optional local LLM (Ollama).')

uploaded = st.file_uploader('Upload text or Markdown files', type=['txt','md'], accept_multiple_files=True)
question = st.text_input('Ask a question about your documents')
use_llm = st.checkbox('Use local Ollama model for answer generation (if installed and running)', value=False)
model = st.text_input('Ollama model name', value='llama3.2')


def chunk_text(text, size=700, overlap=100):
    words = text.split()
    chunks = []
    step = max(size - overlap, 1)
    for start in range(0, len(words), step):
        chunk = ' '.join(words[start:start+size])
        if chunk:
            chunks.append(chunk)
    return chunks


def retrieve(query, chunks):
    if not chunks:
        return []
    matrix = TfidfVectorizer(stop_words='english').fit_transform(chunks + [query])
    scores = cosine_similarity(matrix[-1], matrix[:-1]).ravel()
    idx = scores.argsort()[::-1][:min(3, len(chunks))]
    return [(chunks[i], float(scores[i])) for i in idx if scores[i] > 0]

if st.button('Answer question', type='primary'):
    if not uploaded:
        st.warning('Upload at least one text document first.')
    elif not question.strip():
        st.warning('Type a question first.')
    else:
        documents = []
        for f in uploaded:
            try:
                documents.extend(chunk_text(f.getvalue().decode('utf-8', errors='replace')))
            except Exception:
                st.error(f'Could not read {f.name}')
        sources = retrieve(question, documents)
        context = '\n\n'.join(chunk for chunk, score in sources)
        if not context:
            st.warning('No relevant text chunks found for this question.')
        else:
            st.subheader('Retrieved context')
            for i, (chunk, score) in enumerate(sources, 1):
                st.markdown(f'**Passage {i} (similarity: {score:.2f})**')
                st.write(chunk)
            if use_llm:
                prompt = ('Answer only using the context below. If the context is insufficient, say so.\n\n'
                          f'Context:\n{context}\n\nQuestion: {question}\nAnswer:')
                try:
                    response = requests.post('http://localhost:11434/api/generate',
                                            json={'model': model, 'prompt': prompt, 'stream': False}, timeout=120)
                    response.raise_for_status()
                    answer = response.json().get('response', 'No answer returned.')
                    st.subheader('Generated answer')
                    st.write(answer)
                except Exception as e:
                    st.error('Ollama request failed. Confirm Ollama is running and the model is available. Details: ' + str(e))
            else:
                st.subheader('Answer (retrieval-only mode)')
                st.write('Below are the most relevant document passages. Enable Ollama to generate a natural-language answer.')
                st.write(context)
            st.caption('This demo uses TF-IDF retrieval. Generated answers depend on the local Ollama model and may be inaccurate.')

st.markdown('### Setup Ollama (optional)')
st.markdown('Install Ollama from its official site, then in a terminal run `ollama pull llama3.2` and `ollama serve` if the service is not already running.')
