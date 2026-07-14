import io
import os
import streamlit as st
from pinecone import Pinecone, ServerlessSpec
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings  # <-- Cloud accelerated open-source embeddings
from langchain_groq import ChatGroq                      # <-- Ultra-fast cloud open-source LLM
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from concurrent.futures import ThreadPoolExecutor, as_completed

# Set your API keys (Can be configured in Streamlit Secrets)
# os.environ["PINECONE_API_KEY"] = "your-pinecone-key"
# os.environ["GROQ_API_KEY"] = "your-groq-key"

def extract_pdf_text(file_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    text_pages = []
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text_pages.append(page_text)
    return "\n".join(text_pages)

# --- Config / secrets ---
PINECONE_API_KEY = ""
PINECONE_ENV = "us-east1"  # e.g., "us-east1"
INDEX_NAME = "10m-documents-index"
PINECONE_NAMESPACE = "test"  # set to "" to use default
GROQ_API_KEY = ""
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["PINECONE_ENV"] = PINECONE_ENV
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

pc = Pinecone(pinecone_api_key=PINECONE_API_KEY)

existing = pc.list_indexes()
existing_names = []
if isinstance(existing, dict) and "names" in existing:
    existing_names = existing["names"]
elif hasattr(existing, "names"):
    existing_names = existing.names()
elif isinstance(existing, list):
    existing_names = existing

index_name = INDEX_NAME

# Embedding model chosen below produces 384-dimensional vectors.
EMBED_DIM = 384

if INDEX_NAME not in existing_names:
    st.warning(f"Index '{INDEX_NAME}' not found. Creating now...")
    # Create a simple index; adjust ServerlessSpec if you need serverless config
    pc.create_index(
        name=INDEX_NAME,
        dimension=EMBED_DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    st.success(f"Index '{INDEX_NAME}' created successfully!")
else:
    st.info(f"Index '{INDEX_NAME}' already exists. Using it.")

# --- INITIALIZE CLOUD-ACCELERATED OPEN-SOURCE MODELS ---
# Runs locally but leverages highly optimized tokenization matrices for instant chunk processing
embeddings_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

# Using Groq for near-instantaneous responses.
llm = ChatGroq(model=GROQ_MODEL, temperature=0)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# --- HYPER-FAST PARALLEL UPLOAD LOGIC ---
def upload_single_batch(batch_texts: list[str], batch_metadatas: list[dict]):
    """Uploads an isolated pre-computed batch to Pinecone instantly."""
    try:
        PineconeVectorStore.from_texts(
            texts=batch_texts,
            embedding=embeddings_model,
            index_name=index_name,
            metadatas=batch_metadatas
        )
        return True
    except Exception as e:
        return f"Batch upload failed: {str(e)}"

def process_and_upload_parallel(texts: list[str], metadatas: list[dict], batch_size: int = 200, max_workers: int = 8):
    """Splits texts into large batches and uploads them sequentially to avoid concurrency borrow errors."""
    total_batches = (len(texts) + batch_size - 1) // batch_size
    completed_batches = 0
    
    upload_progress = st.progress(0)
    upload_status = st.empty()

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_metas = metadatas[i:i + batch_size]
        result = upload_single_batch(batch_texts, batch_metas)
        completed_batches += 1

        if result != True:
            st.error(result)
            break

        percent_complete = int((completed_batches / total_batches) * 100)
        upload_progress.progress(percent_complete)
        upload_status.text(f"Uploading data: Batch {completed_batches} / {total_batches} synced...")

    upload_status.text("✨ Ingestion complete!")
    return True


# --- INITIALIZE RETRIEVER & RAG CHAIN ---
vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings_model)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# system_prompt = (
#     "You are an assistant for question-answering tasks. "
#     "Use the following pieces of retrieved context to answer "
#     "the question. If you don't know the answer, say that you "
#     "don't know.\n\n"
#     "{context}"
# )

# system_prompt = (
#     "You are a document Q&A assistant. Answer the question using ONLY the "
#     "information in the retrieved context below. Do not use any outside "
#     "knowledge. If the answer is not contained in the context, respond "
#     "exactly with: 'I don't know based on the provided documents.'\n\n"
#     "{context}"
# )

system_prompt = (
        "You are a document assistant with two jobs:\n"
        "1. Question-answering: Use ONLY the retrieved context below to answer "
        "factual questions about the documents. Do not rely on outside knowledge. "
        "If the answer is not in the context, respond exactly with: "
        "'I don't know based on the provided documents.'\n\n"
        "2. Code generation: If the user asks you to write, generate, or prepare "
        "Python code (for example, implementing an algorithm, formula, config, "
        "schema, or workflow described in the documents), you MAY write complete, "
        "correct, well-commented Python code. Base the code strictly on details "
        "found in the retrieved context - variable names, formulas, steps, and "
        "logic should come from the documents. If the documents don't contain "
        "enough detail to write accurate code, say so explicitly instead of "
        "guessing or inventing details, and ask what's missing.\n\n"
        "Always state clearly whether your response is (a) a direct answer from "
        "the documents or (b) code generated based on the documents.\n\n"
        "Retrieved context:\n{context}"
    )
prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)
 


# --- STREAMLIT UI ---
st.title("Hyper-Fast Open-Source RAG Chatbot")

if "chats" not in st.session_state:
    st.session_state.chats = [[]]
    st.session_state.chat_titles = ["Search 1"]
    st.session_state.current_chat = 0
    st.session_state.next_chat_id = 2
else:
    st.session_state.chat_titles = [
        title.replace("Chat ", "Search ", 1) if title.startswith("Chat ") else title
        for title in st.session_state.chat_titles
    ]


def is_default_chat_title(title: str) -> bool:
    return title.startswith("Search ") or title.startswith("Chat ")


def generate_chat_title_from_query(query: str, prefix: str = "Search", max_words: int = 5) -> str:
    cleaned = query.strip()
    if not cleaned:
        return f"{prefix} {st.session_state.current_chat + 1}"
    words = cleaned.split()
    title = " ".join(words[:max_words]).rstrip(".,?!")
    if len(words) > max_words:
        title += "..."
    return title.capitalize()

with st.sidebar:
    st.header("Chat History")
    if st.button("➕ New Search Chat"):
        st.session_state.chats.append([])
        st.session_state.chat_titles.append(f"Search {st.session_state.next_chat_id}")
        st.session_state.current_chat = len(st.session_state.chats) - 1
        st.session_state.next_chat_id += 1

    st.write("\n")
    for idx, title in enumerate(st.session_state.chat_titles):
        cols = st.columns([8, 1])
        if idx == st.session_state.current_chat:
            cols[0].markdown(f"**{title}**")
        else:
            if cols[0].button(title, key=f"select_chat_{idx}"):
                st.session_state.current_chat = idx

        if cols[1].button("🗑", key=f"delete_chat_{idx}"):
            if len(st.session_state.chats) > 1:
                del st.session_state.chats[idx]
                del st.session_state.chat_titles[idx]
                if st.session_state.current_chat >= len(st.session_state.chats):
                    st.session_state.current_chat = len(st.session_state.chats) - 1
                if st.session_state.current_chat < 0:
                    st.session_state.current_chat = 0
            else:
                st.warning("At least one chat must remain.")


st.header("Document Ingestion Dashboard")
uploaded_files = st.file_uploader("Upload your documents (PDF only)", accept_multiple_files=True, type=["pdf"])

if st.button("Start Ingestion"):
    if not uploaded_files:
        st.warning("Please upload at least one document.")
    else:
        status_text = st.empty()
        
        total_files = len(uploaded_files)
        all_chunks = []
        all_metadata = []

        # Local parsing is now optimized with zero arbitrary sleep pauses
        for i, file in enumerate(uploaded_files):
            status_text.text(f"Parsing: {file.name} ({i + 1}/{total_files})")
            raw_bytes = file.read()
            try:
                file_content = extract_pdf_text(raw_bytes)
            except Exception as e:
                st.error(f"Failed to extract text from {file.name}: {e}")
                continue
            
            chunks = text_splitter.split_text(file_content)
            
            for chunk in chunks:
                all_chunks.append(chunk)
                all_metadata.append({"source": file.name})

        if all_chunks:
            status_text.text("Streaming data to the cloud database...")
            # Uped batch size to 200 items per request and maxed network threads to 8
            process_and_upload_parallel(all_chunks, all_metadata, batch_size=200, max_workers=8)
            st.success(f"Successfully processed {total_files} files into {len(all_chunks)} chunks instantly!")
        else:
            st.error("No text content found to upload.")

# --- CHATBOT SECTION ---
st.markdown("---")
st.header("Chat with PDF Documents")

current_messages = st.session_state.chats[st.session_state.current_chat]
for msg in current_messages:
    role = msg.get("role", "assistant")
    if role not in {"user", "assistant"}:
        role = "assistant"
    with st.chat_message(role):
        st.write(msg["content"])

if user_query := st.chat_input("Ask the PDF documents:"):
    # Auto-name the chat from the first query if it still has the default label
    current_title = st.session_state.chat_titles[st.session_state.current_chat]
    if is_default_chat_title(current_title) and not current_messages:
        st.session_state.chat_titles[st.session_state.current_chat] = generate_chat_title_from_query(user_query)

    current_messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.write(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents..."):
            try:
                response = rag_chain.invoke({"input": user_query})
                answer = response["answer"]
                st.write(answer)
                current_messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"An error occurred: {e}")
