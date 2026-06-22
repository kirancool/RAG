# streamlit_app.py
import os
from io import BytesIO
from typing import List

import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer 
import torch
from transformers import pipeline
from pinecone import Pinecone, ServerlessSpec

# --- Config / secrets ---
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_ENV = os.getenv("PINECONE_ENV", "us-east1")  # e.g., "us-east1"
INDEX_NAME = os.getenv("PINECONE_INDEX", "kiran-app")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "test")  # set to "" to use default

if not PINECONE_API_KEY or not PINECONE_ENV:
    st.error("Set PINECONE_API_KEY and PINECONE_ENV environment variables.")
    st.stop()

pc = Pinecone(api_key=PINECONE_API_KEY)

existing = pc.list_indexes()
existing_names = []
if isinstance(existing, dict) and "names" in existing:
    existing_names = existing["names"]
elif hasattr(existing, "names"):
    existing_names = existing.names()
elif isinstance(existing, list):
    existing_names = existing

# Embedding model chosen below (all-MiniLM-L6-v2 -> dim 384)
EMBED_DIM = 384

if INDEX_NAME not in existing_names:
    st.warning(f"Index '{INDEX_NAME}' not found. Creating now...")
    # Create a simple index; adjust ServerlessSpec if you need serverless config
    pc.create_index(
        name=INDEX_NAME,
        dimension=EMBED_DIM,
        metric="cosine",
        # spec=ServerlessSpec(cloud="aws", region="us-east-1")  # optional
    )
    st.success(f"Index '{INDEX_NAME}' created successfully!")
else:
    st.info(f"Index '{INDEX_NAME}' already exists. Using it.")

# Get index handle
index = pc.Index(INDEX_NAME)

# --- Models ---
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Use text-generation pipeline instead of text2text-generation
qa_pipeline = pipeline("text-generation", model="google/flan-t5-base", device=-1)
# --- Helpers ---
@st.cache_resource
def pdf_to_text(file_like) -> str:
    """
    Accepts a Streamlit uploaded file (BytesIO-like) or raw bytes.
    Returns concatenated page text.
    """
    if hasattr(file_like, "read"):
        data = file_like.read()
    else:
        data = file_like

    if not data:
        return ""

    pages = []
    try:
        reader = PdfReader(BytesIO(data))
        for p in reader.pages:
            try:
                t = p.extract_text() or ""
            except Exception:
                t = ""
            if t.strip():
                pages.append(t)
    except Exception:
        return ""

    return "\n\n---PAGE_BREAK---\n\n".join(pages)


def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)


def embed_texts(texts: List[str]) -> List[List[float]]:
    embs = embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return [e.tolist() for e in embs]

@st.cache_resource
def upsert_batches(index, vectors, batch_size=32, namespace: str = None):
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i : i + batch_size]
        if namespace:
            index.upsert(vectors=batch, namespace=namespace)
        else:
            index.upsert(vectors=batch)


# --- Streamlit UI ---
st.title("PDF Chatbot with Pinecone")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
if uploaded_file is not None:
    with st.spinner("Extracting text from PDF..."):
        text = pdf_to_text(uploaded_file)
    if not text.strip():
        st.error("No extractable text found in the PDF.")
    else:
        st.info("Splitting text into chunks...")
        chunks = chunk_text(text, chunk_size=500, chunk_overlap=50)
        st.write(f"Created {len(chunks)} chunks.")

        st.info("Creating embeddings and upserting to Pinecone (batched)...")
        embs = embed_texts(chunks)
        vectors = []
        for i, emb in enumerate(embs):
            meta = {"chunk_text": chunks[i]}
            vectors.append((f"chunk-{i}", emb, meta))

        upsert_batches(index, vectors, batch_size=64,namespace=PINECONE_NAMESPACE)
        st.success("PDF uploaded and stored in Pinecone!")

# --- Chat UI ---
st.subheader("Chat with your PDF")
query = st.text_input("Ask a question:")

if query:
    if not query.strip():
        st.warning("Please enter a non-empty question.")
    else:
        with st.spinner("Searching Pinecone for relevant context..."):
            q_emb = embedder.encode(query, convert_to_numpy=True).tolist()
            # Query: handle different response shapes defensively
            if PINECONE_NAMESPACE:
                resp = index.query(vector=q_emb, top_k=3, include_metadata=True, namespace=PINECONE_NAMESPACE)
            else:
                    resp = index.query(vector=q_emb, top_k=3, include_metadata=True)
            matches = []
            if isinstance(resp, dict):
                matches = resp.get("matches") or resp.get("results") or []
            elif hasattr(resp, "matches"):
                matches = resp.matches
            # Normalize matches to list of dicts
            if not matches:
                st.info("No relevant context found in the index.")
                context = ""
            else:
                # Some responses have metadata under 'metadata', others under 'payload'
                ctx_parts = []
                for m in matches:
                    meta = m.get("metadata") if isinstance(m, dict) else getattr(m, "metadata", None)
                    if not meta:
                        meta = m.get("payload") if isinstance(m, dict) else getattr(m, "payload", None)
                    chunk_text_val = ""
                    if isinstance(meta, dict):
                        chunk_text_val = meta.get("chunk_text", "")
                    elif hasattr(meta, "get"):
                        chunk_text_val = meta.get("chunk_text", "")
                    ctx_parts.append(chunk_text_val)
                context = " ".join([p for p in ctx_parts if p])

        prompt = f"Context: {context}\n\nQuestion: {query}\nAnswer:"
        with st.spinner("Generating answer from the model..."):
            out = qa_pipeline(prompt, max_length=256, do_sample=False)
            answer = out[0].get("generated_text") or out[0].get("text") or str(out[0])

        st.markdown("**Answer:**")
        st.write(answer)