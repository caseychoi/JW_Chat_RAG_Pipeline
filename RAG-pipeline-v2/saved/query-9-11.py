import os
import sys
import psycopg

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

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3:latest")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")

# psycopg v3 URL
PG_CONN = f"postgresql+psycopg://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

def _auto_collection_name(base="documents", embed_model=EMBEDDING_MODEL):
    """Generates a collection name based on the embedding model to ensure consistency."""
    suffix = embed_model.lower().replace(":", "_").replace("-", "_").replace("/", "_")
    return f"{base}__{suffix}"

# --- Prompt Template ---
QA_PROMPT = PromptTemplate.from_template(
    """You are a helpful and detailed assistant for question-answering tasks.
Your goal is to synthesize a comprehensive answer using the provided context.
Combine information from multiple parts of the context if it helps to create a complete answer.
Always include any scriptural citations to support your answer.
If the context does not contain the information needed to answer the question, just say that you don't know.

Context:
{context}

Question:
{question}

Answer:"""
)

def format_sources(source_documents, max_items=20):
    """Formats source documents for display."""
    lines = []
    items_cnt = len(source_documents)
    for i, d in enumerate(source_documents[:items_cnt], start=1):
        meta = d.metadata or {}
        src = meta.get("source") or meta.get("file_path") or "<unknown source>"
        page = meta.get("page")
        page_str = f", p.{page}" if page is not None else ""
        lines.append(f"[{i}] {src}{page_str}")
    return "\n".join(lines)

# --- 1) Connect to existing vector store ---
def get_vectorstore(collection_name):
    """Connects to an existing vector store."""
    print(f"Connecting to vector store (collection='{collection_name}')...")
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
    
    vectorstore = PGVector(
        embeddings=embeddings,  # CHANGED from embedding_function
        collection_name=collection_name,
        connection=PG_CONN,
        use_jsonb=True,
    )
    print("Vector store connected.")
    return vectorstore

# --- 2) Chatbot (RAG) with pre-check & threshold guarding ---
def run_chatbot(vectorstore):
    """Runs the main chatbot loop."""
    print("\nStarting chatbot. Type 'exit' to quit.")
    llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)

    retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 10, "score_threshold": 0.2},
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": QA_PROMPT},
        return_source_documents=True,
    )

    while True:
        try:
            print("\n\n===============")
            query = input("You: ")
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if query.strip().lower() == "exit":
            break

        try:
            result = qa_chain.invoke({"query": query})
            srcs = result.get("source_documents") or []

            if not srcs:
                print("Bot: I couldn’t find any relevant passages in your documents for that question.")
                continue

            # --- Sort retrieved documents by ingestion date (most recent first) ---
            srcs.sort(key=lambda x: x.metadata.get('ingested_at', ''), reverse=True)
            # --------------------------------------------------------------------

            # --- DEBUG: Print retrieved chunks ---
            print("\n--- Retrieved Chunks (Sorted by Date) ---")
            for i, src in enumerate(srcs):
                print(f"Chunk {i} (source: {src.metadata.get('source')}, ingested: {src.metadata.get('ingested_at')}):")
                print(src.page_content)
                print("------")


            print(f"\n")
            print(f"ChatBot: {result['result']}")
            print("-- Sources --")
            print(format_sources(srcs))

        except Exception as e:
            print(f"Bot: An error occurred while answering: {e}")
            continue

# --- Main Execution ---
if __name__ == "__main__":
    collection_name = _auto_collection_name()
    try:
        vectorstore = get_vectorstore(collection_name)
        run_chatbot(vectorstore)
        
    except psycopg.OperationalError as e:
        print(f"\n--- DATABASE CONNECTION ERROR ---")
        print(f"Error: {e}")
        print("Please ensure PostgreSQL is running and the database '{DB_NAME}' exists.")
        print("Have you run the `ingest.py` script yet to create the collection?")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
