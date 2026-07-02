from langchain_community.retrievers import PineconeHybridSearchRetriever
from pinecone import Pinecone,ServerlessSpec
import os

from dotenv import load_dotenv
load_dotenv()
import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import time
import uuid

load_dotenv()

# Config
PINECONE_API_KEY = ""
PINECONE_ENV = "us-east-1"
INDEX_NAME =  "kiran-app"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Initialize Pinecone
if not PINECONE_API_KEY or not PINECONE_ENV:
    st.error("Set PINECONE_API_KEY and PINECONE_ENV in .env")
    st.stop()

# Embedding model
embedder = SentenceTransformer(EMBED_MODEL)
EMBED_DIM = embedder.get_sentence_embedding_dimension()

# Create/connect to index using the modern Pinecone client API
pc = Pinecone(api_key=PINECONE_API_KEY)


def ensure_index_ready():
    try:
        existing_indexes = pc.list_indexes()
        if hasattr(existing_indexes, "indexes"):
            index_names = [idx.name for idx in existing_indexes.indexes]
        elif isinstance(existing_indexes, dict):
            index_names = [idx.get("name", "") for idx in existing_indexes.get("indexes", [])]
        else:
            index_names = [idx.get("name", "") for idx in existing_indexes]
    except Exception as exc:
        st.error(f"Unable to list Pinecone indexes: {exc}")
        st.stop()

    if INDEX_NAME not in index_names:
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBED_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region=PINECONE_ENV),
        )

    for _ in range(20):
        try:
            pc.describe_index(name=INDEX_NAME)
            return pc.Index(INDEX_NAME)
        except Exception:
            time.sleep(5)

    st.error(f"Pinecone index '{INDEX_NAME}' did not become ready in time.")
    st.stop()


index = ensure_index_ready()

# Helpers
def pdf_to_text(file) -> str:
    reader = PdfReader(file)
    text = []
    for p in reader.pages:
        page_text = p.extract_text()
        if page_text:
            text.append(page_text)
    return "\n".join(text)

def chunk_text(text, chunk_size=800, overlap=150):
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = text[start:end]
        chunks.append(chunk.strip())
        start += chunk_size - overlap
    return chunks

def embed_texts(texts):
    embs = embedder.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    return embs

def upsert_document(name, chunks):
    namespace = name.replace(".", "_")
    vectors = []
    embs = embed_texts(chunks)
    for i, (chunk, emb) in enumerate(zip(chunks, embs)):
        vid = f"{namespace}_{i}_{uuid.uuid4().hex[:8]}"
        meta = {"text": chunk, "doc": namespace, "chunk_index": i}
        vectors.append({
            "id": vid,
            "values": emb.tolist(),
            "metadata": meta,
        })

    batch_size = 100
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i+batch_size]
        try:
            index.upsert(vectors=batch)
        except Exception as exc:
            raise RuntimeError(f"Pinecone upsert failed for batch {i//batch_size + 1}: {exc}") from exc
    return len(vectors)


def query_pinecone(query, top_k=4, namespace=None):
    q_emb = embedder.encode([query], convert_to_numpy=True)[0].tolist()
    query_params = {"vector": q_emb, "top_k": top_k, "include_metadata": True}
    if namespace:
        query_params["namespace"] = namespace
    resp = index.query(**query_params)
    matches = resp.get("matches", [])
    results = []
    for m in matches:
        meta = m.get("metadata", {})
        text = meta.get("text", "")
        score = m.get("score", None)
        results.append({"text": text, "score": score})
    return results

def build_answer(question, results):
    if not results:
        return "No relevant context was found in the uploaded document."

    best_match = results[0]["text"]
    answer = (
        f"Based on the retrieved context, the most relevant information is: {best_match[:1000]}"
    )
    if len(results) > 1:
        supporting = "\n\nAdditional relevant snippets:\n" + "\n".join(
            f"- {r['text'][:400]}" for r in results[1:3]
        )
        answer += supporting
    return answer

# Streamlit UI
st.set_page_config(page_title="PDF Pinecone Hugging Face Chat", layout="wide")
st.title("PDF Pinecone Hugging Face Chatbot")

with st.sidebar:
    st.header("Settings")
    chunk_size = st.number_input("Chunk size (chars)", value=800, step=100)
    overlap = st.number_input("Chunk overlap (chars)", value=150, step=50)
    top_k = st.number_input("Pinecone top_k", value=4, min_value=1, max_value=10)

# Upload and ingest
uploaded = st.file_uploader("Upload PDF to ingest", type=["pdf"])
if uploaded:
    st.info("Extracting text from PDF...")
    raw_text = pdf_to_text(uploaded)
    if not raw_text.strip():
        st.error("No text extracted from PDF.")
    else:
        st.success("Text extracted.")
        doc_name = uploaded.name
        chunks = chunk_text(raw_text, chunk_size=int(chunk_size), overlap=int(overlap))
        st.write(f"Document: **{doc_name}** — Chunks: **{len(chunks)}**")
        if st.button("Ingest to Pinecone"):
            with st.spinner("Embedding & uploading to Pinecone..."):
                try:
                    count = upsert_document(doc_name, chunks)
                    st.success(f"Upserted {count} vectors to Pinecone (index: {INDEX_NAME}).")
                except Exception as exc:
                    st.error(f"Upload failed: {exc}")

# Chat UI
if "history" not in st.session_state:
    st.session_state.history = []

st.subheader("Ask a question about ingested documents")
question = st.text_input("Your question", key="question_input")
if st.button("Ask") and question:
    with st.spinner("Retrieving relevant chunks..."):
        results = query_pinecone(question, top_k=int(top_k))
    if not results:
        st.warning("No relevant chunks found.")
    else:
        st.write("**Retrieved snippets (top results):**")
        for r in results:
            st.write(f"- (score: {r['score']:.4f}) {r['text'][:400]}")

        # Build prompt for HF LLM
        context_text = "\n\n---\n\n".join([r["text"] for r in results])
        prompt = (
            "You are a helpful assistant. Use the following context extracted from a document to answer the user's question.\n\n"
            f"Context:\n{context_text}\n\n"
            f"Question: {question}\n\n"
            "Answer concisely and cite the context when relevant. If the answer is not in the context, say you don't know."
        )

        with st.spinner("Generating answer from retrieved context..."):
            answer = build_answer(question, results)
        st.subheader("Answer")
        st.write(answer)

        # Save to history
        st.session_state.history.append({"q": question, "a": answer, "time": time.time()})

# Conversation history
if st.session_state.history:
    st.subheader("Conversation history")
    for item in reversed(st.session_state.history[-10:]):
        st.markdown(f"**Q:** {item['q']}")
        st.markdown(f"**A:** {item['a']}")
        st.write("---")