import os
import sys
import psycopg  # psycopg v3
from uuid import UUID

from langchain_community.document_loaders import PyMuPDFLoader # CHANGED
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_postgres.vectorstores import PGVector


# --- Configuration ---
DB_USER = os.getenv("DB_USER", os.getenv("USER", "postgres"))
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "rag_chatbot")

# Models (lighter defaults)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
#OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3:latest")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")

PDF_DIR = os.getenv("PDF_DIR", "./docs")

# psycopg v3 URL
PG_CONN = f"postgresql+psycopg://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Optional: set to "1" to drop an existing collection before reindexing
#RESET_COLLECTION = os.getenv("RESET_COLLECTION", "0") == "1"
RESET_COLLECTION = "1"



def _auto_collection_name(base="documents", embed_model=EMBEDDING_MODEL):
    suffix = embed_model.lower().replace(":", "_").replace("-", "_").replace("/", "_")
    return f"{base}__{suffix}"


# Auto-name collection to avoid cross-model dimension collisions
COLLECTION_NAME = _auto_collection_name()


# --- 1) Data ingestion & splitting ---
def load_and_split_documents(pdf_directory):
    print("Loading documents...")
    
    pdf_files = [os.path.join(pdf_directory, f) for f in os.listdir(pdf_directory) if f.endswith('.pdf')]
    if not pdf_files:
        print(f"No PDF files found in '{pdf_directory}'. Please add some PDFs and try again.")
        sys.exit(0)

    all_docs_loaded = []
    for pdf_file in pdf_files:
        try:
            loader = PyMuPDFLoader(pdf_file)
            all_docs_loaded.extend(loader.load())
        except Exception as e:
            print(f"Warning: Could not load {pdf_file}. Error: {e}")

    if not all_docs_loaded:
        print("No documents could be loaded from the PDF files. Exiting.")
        sys.exit(0)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,   # smaller chunks help tiny models
        chunk_overlap=150
    )
    docs = text_splitter.split_documents(all_docs_loaded)
    print(f"Split {len(all_docs_loaded)} document pages into {len(docs)} chunks.")

    return docs

# --- Optional: drop a single collection (cleanup/reset) ---
def drop_collection_if_requested(collection_name: str):
    if not RESET_COLLECTION:
        return
    print(f"RESET_COLLECTION=1 -> dropping collection '{collection_name}'")
    with psycopg.connect(f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}") as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT uuid FROM langchain_pg_collection WHERE name = %s",
                (collection_name,)
            )
            row = cur.fetchone()
            if not row:
                print("No existing collection found; nothing to drop.")
                return
            collection_id = row[0]
            cur.execute(
                "DELETE FROM langchain_pg_embedding WHERE collection_id = %s",
                (collection_id,)
            )
            cur.execute(
                "DELETE FROM langchain_pg_collection WHERE uuid = %s",
                (collection_id,)
            )
            print("Collection dropped.")
        conn.commit()


# --- 2) Embedding & vector store (Postgres + pgvector) ---
def create_or_get_vectorstore(docs):
    print(f"Creating / loading vector store (collection='{COLLECTION_NAME}')...")
    drop_collection_if_requested(COLLECTION_NAME)

    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)

    # from_documents creates/loads the collection and inserts provided docs
    vs = PGVector.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        connection=PG_CONN,
        use_jsonb=True,
    )

    print("Vector store ready.")
    return vs


# --- Prompt: cautious, abstain if no/insufficient context ---
QA_PROMPT = PromptTemplate.from_template(
    """You are a helpful and detailed assistant for question-answering tasks.
Your goal is to synthesize a comprehensive answer using the provided context.
Combine information from multiple parts of the context if it helps to create a complete answer.
If the context does not contain the information needed to answer the question, just say that you don't know.

Context:
{context}

Question:
{question}

Answer:"""
)
#QA_PROMPT = PromptTemplate.from_template(e r
#    """You are a cautious assistant. Use ONLY the context to answer.
#If the context is empty or insufficient, reply exactly:
#"I don’t have enough information in the provided documents to answer that."

#Context:
#{context}

#Question:

#{question}

#Answer:"""
#)


# --- Helper to show short citations for transparency ---
def format_sources(source_documents, max_items=3):
    lines = []
    for i, d in enumerate(source_documents[:max_items], start=1):
        meta = d.metadata or {}
        src = meta.get("source") or meta.get("file_path") or "<unknown source>"
        page = meta.get("page")
        page_str = f", p.{page}" if page is not None else ""
        lines.append(f"[{i}] {src}{page_str}")
    return "\n".join(lines)


# --- 3) Chatbot (RAG) with pre-check & threshold guarding ---
def run_chatbot(vectorstore):
    print("Starting chatbot. Type 'exit' to quit.")

    llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)

    # Score-threshold retriever: avoids low-relevance matches
    retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 8, "score_threshold": 0.2},
    )

    # Build chain with cautious prompt + return sources
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": QA_PROMPT},
        return_source_documents=True,
    )

    while True:
        try:
            query = input("You: ")
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if query.strip().lower() == "exit":
            break

        # --- Pre-check: attempt retrieval first, skip LLM if empty ---
        try:
            docs = retriever.invoke(query)
        except Exception as e:
            print(f"Bot: Retrieval error: {e}")
            continue


        if not docs:
            print("Bot: I couldn’t find any relevant passages in your documents for that question.")
            continue



  # --- DEBUG: Print retrieved chunks ---
        print("\n--- Retrieved Chunks ---")
        for i, doc in enumerate(docs):
            print(f"Chunk {i} (source: {doc.metadata.get('source')}, page: {doc.metadata.get('page')}):")
            print(doc.page_content)
            print("------")



        # --- Invoke the QA chain ---
        try:
            result = qa_chain.invoke({"query": query})
        except Exception as e:
            print(f"Bot: An error occurred while answering: {e}")
            continue

        # Belt & suspenders: ensure sources exist
        srcs = result.get("source_documents") or []
        if not srcs:
            print("Bot: I couldn’t find any relevant passages in your documents for that question.")
            continue

        print(f"Bot: {result['result']}")
        print("-- Sources --")
        print(format_sources(srcs))


# --- Main ---
if __name__ == "__main__":
    if not os.path.exists(PDF_DIR):
        os.makedirs(PDF_DIR, exist_ok=True)
        print(f"Created directory: {PDF_DIR}. Please place your PDFs here.")
        sys.exit(0)

    try:
        documents = load_and_split_documents(PDF_DIR)
        vectorstore = create_or_get_vectorstore(documents)
        run_chatbot(vectorstore)

    except psycopg.OperationalError as e:
        print(f"Error connecting to PostgreSQL: {e}")
        print("Ensure PostgreSQL is running and DB 'rag_chatbot' exists.")
        print("Also verify pgvector is enabled:  CREATE EXTENSION IF NOT EXISTS vector;")

    except ModuleNotFoundError as e:
        print(f"Missing module: {e}.")
        print('Inside your virtual environment, install:')
        print('  python -m pip install -U langchain-postgres langchain-ollama "psycopg[binary]" pgvector SQLAlchemy')

    except Exception as e:
        print(f"An error occurred: {e}")