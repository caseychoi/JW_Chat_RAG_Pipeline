import os
import sys
import psycopg  # psycopg v3
from uuid import UUID

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain.chains import RetrievalQA
from langchain_postgres.vectorstores import PGVector

# --- Configuration ---
DB_USER = os.getenv("DB_USER", os.getenv("USER", "postgres"))
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "rag_chatbot")

# Models (lighter defaults)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:1b")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")

PDF_DIR = os.getenv("PDF_DIR", "./docs")

# psycopg v3 URL
PG_CONN = f"postgresql+psycopg://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Optional: set to "1" to drop an existing collection before reindexing
RESET_COLLECTION = os.getenv("RESET_COLLECTION", "0") == "1"

def _auto_collection_name(base="documents", embed_model=EMBEDDING_MODEL):
    # make a model-safe suffix for the collection name
    suffix = embed_model.lower().replace(":", "_").replace("-", "_").replace("/", "_")
    return f"{base}__{suffix}"

# Auto-name collection to avoid dimension collisions across different embed models
COLLECTION_NAME = _auto_collection_name()


# --- 1. Data Ingestion & Splitting ---
def load_and_split_documents(pdf_directory):
    print("Loading documents...")
    loader = PyPDFDirectoryLoader(pdf_directory)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,     # tiny models do better with smaller chunks
        chunk_overlap=150
    )
    docs = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(docs)} chunks.")
    return docs


# --- Optional: DROP just one collection (and its embeddings) ---
def drop_collection_if_requested(collection_name: str):
    if not RESET_COLLECTION:
        return
    print(f"RESET_COLLECTION=1 -> dropping collection '{collection_name}'")
    with psycopg.connect(f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}") as conn:
        with conn.cursor() as cur:
            # Find the collection UUID
            cur.execute(
                "SELECT uuid FROM langchain_pg_collection WHERE name = %s",
                (collection_name,)
            )
            row = cur.fetchone()
            if not row:
                print("No existing collection found; nothing to drop.")
                return

            collection_id = row[0]  # UUID
            # Delete embeddings first due to FK
            cur.execute(
                "DELETE FROM langchain_pg_embedding WHERE collection_id = %s",
                (collection_id,)
            )
            # Delete the collection record
            cur.execute(
                "DELETE FROM langchain_pg_collection WHERE uuid = %s",
                (collection_id,)
            )
            print("Collection dropped.")
        conn.commit()


# --- 2. Embedding Generation & Storage (Postgres + pgvector) ---
def create_or_get_vectorstore(docs):
    print(f"Creating / loading vector store (collection='{COLLECTION_NAME}')...")
    drop_collection_if_requested(COLLECTION_NAME)

    # Ensure you have `ollama pull nomic-embed-text` (or whichever model you set)
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)

    # Create or load the collection; will insert provided docs
    vs = PGVector.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        connection=PG_CONN,
        use_jsonb=True,
    )

    print("Vector store ready.")
    return vs


# --- 3. Chatbot Logic (RAG) ---
def run_chatbot(vectorstore):
    print("Starting chatbot. Type 'exit' to quit.")
    llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever
    )

    while True:
        try:
            query = input("You: ")
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if query.strip().lower() == "exit":
            break

        try:
            response = qa_chain.invoke({"query": query})
            print(f"Bot: {response['result']}")
        except Exception as e:
            print(f"An error occurred while answering: {e}")


# --- Main Execution ---
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