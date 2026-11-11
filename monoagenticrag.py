import os
import sys
import datetime
import urllib.parse
from io import BytesIO
from typing import List, Dict, Any, Optional

# Work around an incompatible installed pyOpenSSL/cryptography pair.
# PyMongo uses Python's ssl module directly; this prevents unrelated
# dependencies from importing the broken optional OpenSSL binding.
import streamlit as st
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.server_api import ServerApi
import certifi
import gridfs
from bson.objectid import ObjectId

# LangChain components
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# PDF loader
import pypdf

# Load environment variables from .env
load_dotenv()
# --- Load Credentials Silently from Backend / .env ---
NVIDIA_API_KEY=os.getenv("NVIDIA_API_KEY","NVIDIA_API_KEY")
MONGO_URI =os.getenv("MONGO_URI","MONGO_URI")
DB_NAME = os.getenv("MONGO_DB_NAME", "rag_vault")
# --- Page Configuration ---
st.set_page_config(
    page_title="MongoDB Document Vault & Chatbot",
    page_icon="🍃",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Session State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "active_file" not in st.session_state:
    st.session_state.active_file = "All Documents"


# --- MongoDB Connection ---
@st.cache_resource(show_spinner=False)
def init_mongo_connection(uri: str):
    """Establishes and caches the MongoDB connection."""
    if not uri:
        return None
    # Atlas connections require TLS. Use certifi's current CA bundle so
    # Python can validate Atlas certificates consistently on Windows.
    return MongoClient(
        uri,
        tls=True,
        tlsCAFile=certifi.where(),
        server_api=ServerApi(version="1", strict=True, deprecation_errors=True),
        connectTimeoutMS=20000,
        serverSelectionTimeoutMS=30000,
    )


def get_mongo_collections(client: MongoClient, db_name: str = "rag_vault"):
    """Returns database, GridFS bucket (for full files), and chunks collection (for RAG)."""
    db = client[db_name]
    fs = gridfs.GridFS(db)
    chunks_col = db["document_chunks"]
    return db, fs, chunks_col


# --- Text Extraction & Chunking ---
def extract_text(uploaded_file) -> List[Dict[str, Any]]:
    """Extracts text content and page numbers from uploaded file."""
    file_bytes = uploaded_file.getvalue()
    file_name = uploaded_file.name
    pages = []

    if file_name.lower().endswith(".pdf"):
        reader = pypdf.PdfReader(BytesIO(file_bytes))
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                pages.append({"text": text, "page": page_num + 1})
    else:
        # Text, Markdown, CSV
        try:
            content = file_bytes.decode("utf-8", errors="ignore")
            pages.append({"text": content, "page": 1})
        except Exception as e:
            st.error(f"Error reading file content: {e}")

    return pages


def upload_and_index_file(
    uploaded_file,
    client: MongoClient,
    db_name: str,
    embed_model: Optional[NVIDIAEmbeddings] = None
):
    """
    1. Uploads the raw binary file into MongoDB GridFS.
    2. Chunks extracted text and stores chunks in MongoDB collection.
    3. Optionally attaches vector embeddings to each chunk.
    """
    db, fs, chunks_col = get_mongo_collections(client, db_name)
    file_bytes = uploaded_file.getvalue()
    filename = uploaded_file.name

    # Remove previous version if the file was uploaded before
    existing = fs.find_one({"filename": filename})
    if existing:
        fs.delete(existing._id)
        chunks_col.delete_many({"filename": filename})

    # 1. Store raw binary file in MongoDB GridFS
    file_id = fs.put(
        file_bytes,
        filename=filename,
        content_type=uploaded_file.type,
        upload_date=datetime.datetime.utcnow(),
        size_bytes=len(file_bytes)
    )

    # 2. Extract and split text into chunks
    pages = extract_text(uploaded_file)
    if not pages:
        return file_id, 0

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=60)
    chunk_docs = []

    for page_data in pages:
        split_chunks = text_splitter.split_text(page_data["text"])
        for chunk_idx, chunk_text in enumerate(split_chunks):
            doc = {
                "file_id": file_id,
                "filename": filename,
                "page": page_data["page"],
                "chunk_index": chunk_idx,
                "text": chunk_text,
                "created_at": datetime.datetime.utcnow()
            }

            # Optional vector embedding
            if embed_model:
                try:
                    doc["embedding"] = embed_model.embed_query(chunk_text)
                except Exception:
                    pass

            chunk_docs.append(doc)

    # Insert text chunks into MongoDB
    if chunk_docs:
        chunks_col.insert_many(chunk_docs)

    return file_id, len(chunk_docs)


# --- Search & Retrieval from MongoDB ---
def retrieve_relevant_chunks(
    query: str,
    client: MongoClient,
    db_name: str,
    filter_filename: Optional[str] = None,
    embed_model: Optional[NVIDIAEmbeddings] = None,
    limit: int = 4
) -> List[Dict[str, Any]]:
    """Retrieves relevant chunks from MongoDB using vector similarity or keyword matching."""
    _, _, chunks_col = get_mongo_collections(client, db_name)
    filter_query = {}
    if filter_filename and filter_filename != "All Documents":
        filter_query["filename"] = filter_filename

    # 1. Try Vector Cosine Similarity if embedding model is active
    if embed_model:
        try:
            query_vector = embed_model.embed_query(query)
            cursor = chunks_col.find({**filter_query, "embedding": {"$exists": True}})
            candidates = list(cursor)

            if candidates:
                import numpy as np

                def cosine_sim(a, b):
                    denom = np.linalg.norm(a) * np.linalg.norm(b)
                    return float(np.dot(a, b) / denom) if denom != 0 else 0.0

                scored = [(cosine_sim(query_vector, doc["embedding"]), doc) for doc in candidates]
                scored.sort(key=lambda x: x[0], reverse=True)
                return [item[1] for item in scored[:limit]]
        except Exception:
            pass

    # 2. Fallback: Keyword search across MongoDB chunks
    words = [w.strip() for w in query.split() if len(w.strip()) > 2]
    if words:
        regex_pattern = "|".join([f"(?i){w}" for w in words[:4]])
        filter_query["text"] = {"$regex": regex_pattern}

    results = list(chunks_col.find(filter_query).limit(limit))
    if not results and "text" in filter_query:
        del filter_query["text"]
        results = list(chunks_col.find(filter_query).limit(limit))

    return results


# Connect to MongoDB in the backend
mongo_client = None
if MONGO_URI:
    try:
        mongo_client = init_mongo_connection(MONGO_URI)
    except Exception:
        mongo_client = None


# ==========================================
# SIDEBAR: Upload & Document Management
# ==========================================
with st.sidebar:
    st.header("⚙️ Settings")

    # Model Selection
    selected_model = st.selectbox(
        "NVIDIA Model:",
        [
            "meta/llama-3.1-70b-instruct",
            "nvidia/nemotron-3-ultra-550b-a55b",
            "meta/llama-3.1-8b-instruct",
            "mistralai/mixtral-8x7b-instruct-v0.1"
        ],
        index=0
    )
    use_embeddings = st.checkbox("Enable Vector Embeddings", value=False)

    st.divider()

    # File Upload into MongoDB
    st.subheader("Upload Document")
    uploaded_file = st.file_uploader("Upload PDF or Text", type=["pdf", "txt", "md"])

    if uploaded_file and mongo_client:
        if st.button("Upload & Index in MongoDB", use_container_width=True):
            with st.spinner(f"Uploading '{uploaded_file.name}' to MongoDB..."):
                embed_model = None
                if use_embeddings and NVIDIA_API_KEY:
                    try:
                        embed_model = NVIDIAEmbeddings(model="nvidia/nv-embedqa-e5-v5", api_key=NVIDIA_API_KEY)
                    except Exception as e:
                        st.warning(f"Embedding initialization skipped: {e}")

                try:
                    file_id, total_chunks = upload_and_index_file(
                        uploaded_file=uploaded_file,
                        client=mongo_client,
                        db_name=DB_NAME,
                        embed_model=embed_model
                    )
                    st.success(f"Uploaded '{uploaded_file.name}' ({total_chunks} chunks indexed in MongoDB)!")
                    st.rerun()
                except Exception as e:
                    error_text = str(e)
                    if 'SSL handshake failed' in error_text or 'TLSV1_ALERT_INTERNAL_ERROR' in error_text:
                        st.error(
                            'MongoDB Atlas rejected the TLS connection. Update PyMongo and certifi, ' 
                            'then verify this machine’s public IP is in the Atlas Network Access IP access list ' 
                            'and that the network/firewall allows outbound TCP 27017.'
                        )
                    else:
                        st.error(f"Upload failed: {e}")

    st.divider()

    # Stored Documents Management
    st.subheader("Stored Documents")
    file_list = ["All Documents"]
    if mongo_client:
        try:
            db, fs, chunks_col = get_mongo_collections(mongo_client, DB_NAME)
            stored = list(fs.find())
            for item in stored:
                file_list.append(item.filename)

            if not stored:
                st.caption("No files uploaded in MongoDB yet.")
            else:
                st.caption(f"{len(stored)} document(s) in database.")
                for item in stored:
                    col_txt, col_btn = st.columns([4, 1])
                    with col_txt:
                        st.text(f"📄 {item.filename}")
                    with col_btn:
                        if st.button("🗑️", key=f"del_{item._id}", help=f"Delete {item.filename}"):
                            fs.delete(item._id)
                            chunks_col.delete_many({"filename": item.filename})
                            st.rerun()
        except Exception:
            pass

    st.session_state.active_file = st.selectbox(
        "Chat with Document:",
        file_list,
        index=file_list.index(st.session_state.active_file) if st.session_state.active_file in file_list else 0
    )

    if st.button("🧹 Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ==========================================
# MAIN APP: Header & Chatbot Interface
# ==========================================
st.title("🍃 MongoDB Document Chatbot")
st.caption(f"Active Document Scope: **{st.session_state.active_file}**")

# Display previous conversation messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander("🔍 View Retrieved MongoDB Chunks"):
                for idx, src in enumerate(msg["sources"]):
                    st.markdown(f"**Chunk {idx+1}** (File: `{src.get('filename')}`, Page: {src.get('page', 'N/A')}):")
                    st.text(src.get("text", "")[:400] + ("..." if len(src.get("text", "")) > 400 else ""))

# Chat input prompt
if user_prompt := st.chat_input("Ask a question about your uploaded document..."):
    # Append & display user message
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # Validations
    if not mongo_client:
        with st.chat_message("assistant"):
            st.error("MongoDB connection is not established. Please check your credentials in .env.")
    elif not NVIDIA_API_KEY:
        with st.chat_message("assistant"):
            st.error("NVIDIA API Key not found in .env. Please configure it in your environment.")
    else:
        with st.chat_message("assistant"):
            with st.spinner("Retrieving document context from MongoDB..."):
                embed_model = None
                if use_embeddings and NVIDIA_API_KEY:
                    try:
                        embed_model = NVIDIAEmbeddings(model="nvidia/nv-embedqa-e5-v5", api_key=NVIDIA_API_KEY)
                    except Exception:
                        pass

                # Retrieve relevant chunks from MongoDB
                matched_chunks = retrieve_relevant_chunks(
                    query=user_prompt,
                    client=mongo_client,
                    db_name=DB_NAME,
                    filter_filename=st.session_state.active_file,
                    embed_model=embed_model,
                    limit=4
                )

                if matched_chunks:
                    context_str = "\n\n---\n\n".join(
                        [f"[File: {c.get('filename')}, Page: {c.get('page')}]\n{c.get('text')}" for c in matched_chunks]
                    )
                else:
                    context_str = "No specific document chunks found in MongoDB for this query."

                system_prompt = (
                    "You are a helpful and accurate document assistant. "
                    "Use the provided context retrieved from MongoDB to answer the user's question. "
                    "If the answer is not contained in the context, clearly explain that.\n\n"
                    f"Context from MongoDB:\n{context_str}"
                )

                messages = [SystemMessage(content=system_prompt)]
                # Include recent conversation turns
                for m in st.session_state.messages[-4:]:
                    if m["role"] == "user":
                        messages.append(HumanMessage(content=m["content"]))
                    elif m["role"] == "assistant":
                        messages.append(AIMessage(content=m["content"]))

                try:
                    llm = ChatNVIDIA(
                        model=selected_model,
                        api_key=NVIDIA_API_KEY,
                        temperature=0.7,
                        max_completion_tokens=2048
                    )

                    # Stream assistant response
                    response_container = st.empty()
                    accumulated_response = ""

                    for chunk in llm.stream(messages):
                        accumulated_response += chunk.content
                        response_container.markdown(accumulated_response + "▌")

                    response_container.markdown(accumulated_response)

                    # Store assistant message in history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": accumulated_response,
                        "sources": matched_chunks
                    })

                    if matched_chunks:
                        top_chunk = matched_chunks[0]
                        st.write(f"Top Match (File: {top_chunk.get('filename')}, Page: {top_chunk.get('page', 'N/A')}):")
                        st.text(top_chunk.get('text', '')[:400] + ("..." if len(top_chunk.get('text', '')) > 400 else ""))

                except Exception as e:
                    st.error(f"Error querying NVIDIA LLM: {e}")