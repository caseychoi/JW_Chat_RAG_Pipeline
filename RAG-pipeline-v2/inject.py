import os
import sys
import psycopg

from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_postgres.vectorstores import PGVector

# --- Configuration ---
DB_USER = os.getenv("DB_USER", os.getenv("USER", "postgres"))
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "rag_chatbot")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
PDF_DIR = os.getenv("PDF_DIR", "./docs")

# psycopg v3 URL
PG_CONN = f"postgresql+psycopg://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Optional: set to "1" to drop an existing collection before reindexing
#RESET_COLLECTION = os.getenv("RESET_COLLECTION", "0") == "1"
RESET_COLLECTION = False

def _auto_collection_name(base="documents", embed_model=EMBEDDING_MODEL):
    """Generates a collection name based on the embedding model."""
    suffix = embed_model.lower().replace(":", "_").replace("-", "_").replace("/", "_")
    return f"{base}__{suffix}"

# --- 1) Data ingestion & splitting ---
def load_and_split_documents(pdf_directory):
    """Loads PDFs from a directory and splits them into manageable chunks."""
    print("Loading documents...")
    pdf_files = [os.path.join(pdf_directory, f) for f in os.listdir(pdf_directory) if f.endswith('.pdf')]
    if not pdf_files:
        print(f"No PDF files found in '{pdf_directory}'. Please add some PDFs and try again.")
        return None

    all_docs_loaded = []
    for pdf_file in pdf_files:
        try:
            loader = PyMuPDFLoader(pdf_file)
            all_docs_loaded.extend(loader.load())
        except Exception as e:
            print(f"Warning: Could not load {pdf_file}. Error: {e}")

    if not all_docs_loaded:
        print("No documents could be loaded from the PDF files.")
        return None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150)
    docs = text_splitter.split_documents(all_docs_loaded)
    print(f"Split {len(all_docs_loaded)} document pages into {len(docs)} chunks.")
    return docs

# --- Optional: drop a single collection (cleanup/reset) ---
def drop_collection_if_requested(collection_name: str):
    """Drops the database collection if the RESET_COLLECTION flag is set."""
    if not RESET_COLLECTION:
        return
    print(f"RESET_COLLECTION is set. Dropping collection '{collection_name}'...")
    with psycopg.connect(f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}") as conn:
        with conn.cursor() as cur:
            # Find the collection's UUID
            cur.execute("SELECT uuid FROM langchain_pg_collection WHERE name = %s", (collection_name,))
            row = cur.fetchone()
            if not row:
                print("Collection not found; nothing to drop.")
                return
            
            collection_id = row[0]
            # Delete associated embeddings and the collection entry
            cur.execute("DELETE FROM langchain_pg_embedding WHERE collection_id = %s", (collection_id,))
            cur.execute("DELETE FROM langchain_pg_collection WHERE uuid = %s", (collection_id,))
            print("Collection dropped successfully.")
        conn.commit()

# --- 2) Embedding & vector store creation ---
def create_vector_store(docs, collection_name):
    """Creates and populates the vector store with document embeddings."""
    print(f"Creating vector store (collection='{collection_name}')...")
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)

    PGVector.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=collection_name,
        connection=PG_CONN,
        use_jsonb=True,
    )
    print("Vector store created and populated successfully.")

# --- Main Execution ---
if __name__ == "__main__":
    if not os.path.exists(PDF_DIR):
        os.makedirs(PDF_DIR, exist_ok=True)
        print(f"Created directory: {PDF_DIR}. Please place your PDF files here and run again.")
        sys.exit(0)

    collection_name = _auto_collection_name()
    
    try:
        drop_collection_if_requested(collection_name)
        documents = load_and_split_documents(PDF_DIR)
        if documents:
            create_vector_store(documents, collection_name)
            print("\nIngestion complete.")
        else:
            print("\nIngestion skipped as no documents were loaded.")
            
    except psycopg.OperationalError as e:
        print(f"\n--- DATABASE CONNECTION ERROR ---")
        print(f"Error: {e}")
        print("Please ensure PostgreSQL is running and the database '{DB_NAME}' exists.")
        print("You may also need to enable the pgvector extension: CREATE EXTENSION IF NOT EXISTS vector;")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")