import os
import sys
import psycopg
import socket

# --- NEW: Enable LangChain's global verbose/debug mode ---
import langchain
langchain.debug = False
# -
# --- IMPORTS for a Multi-Source RAG chain ---
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from operator import itemgetter
from langchain.schema.document import Document
from langchain_community.tools.tavily_search import TavilySearchResults

from langchain_ollama import OllamaEmbeddings, ChatOllama
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

# --- Make sure Tavily API Key is set ---
if not os.getenv("TAVILY_API_KEY"):
    print("TAVILY_API_KEY environment variable not set. Please get a key from https://tavily.com and set it.")
    sys.exit(1)
# -----------------------------------------

# psycopg v3 URL
PG_CONN = f"postgresql+psycopg://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# --- RAG Prompt for synthesizing answers ---
RAG_PROMPT_TEMPLATE = """
You are a research assistant. Use the following retrieved context from local documents and web search results to answer the question.
If you don't know the answer from the context, just say that you don't know.
Do not use any outside knowledge. Provide scriptural citations if they are present in the context.

CONTEXT:
{context}

QUESTION:
{input}

ANSWER:
"""
# --------------------------------

def _auto_collection_name(base="documents", embed_model=EMBEDDING_MODEL):
    """Generates a collection name based on the embedding model."""
    suffix = embed_model.lower().replace(":", "_").replace("-", "_").replace("/", "_")
    return f"{base}__{suffix}"

def check_internet_connection(host="8.8.8.8", port=53, timeout=3):
    """Check for internet connectivity."""
    try:
        socket.setdefaulttimeout(timeout)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
        return True
    except socket.error:
        return False

# --- 1) Connect to existing vector store ---
def get_vectorstore(collection_name):
    """Connects to an existing vector store."""
    print(f"Connecting to vector store (collection='{collection_name}')...")
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
    vectorstore = PGVector(
        embeddings=embeddings,
        collection_name=collection_name,
        connection=PG_CONN,
        use_jsonb=True,
    )
    print("Vector store connected.")
    return vectorstore

# --- 2) Create the Multi-Source RAG Chain ---
def create_rag_chain(vectorstore):
    """Creates a RAG chain that queries local docs and the web in parallel."""
    print("Creating Multi-Source RAG chain...")
    llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)
    prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

    # Source 1: Local PDF Retriever
    retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 10, "score_threshold": 0.2},
    )

    # Source 2: Tavily Web Search Tool
    web_search_tool = TavilySearchResults(max_results=5)

    # This chain correctly extracts the 'input' string and passes it to the local retriever
    local_retriever_chain = itemgetter("input") | retriever

    # This chain correctly formats the query for the web search and executes it
    web_search_chain = (
        itemgetter("input")
        | RunnableLambda(lambda q: q + " site:www.jw.org")
        | web_search_tool
    )

    # This runnable runs both retrievers in parallel
    combined_retriever = RunnableParallel(
        retrieved_docs=local_retriever_chain,
        web_results=web_search_chain,
    )

    def format_docs(docs):
        """
        Helper function to format a list of documents (either Document objects or dicts)
        into a single string.
        """
        if not docs:
            return "No information found."

        # Check if the first item is a Document object (from local retriever)
        if isinstance(docs[0], Document):
            return "\n\n".join(doc.page_content for doc in docs)
        
        # Check if the first item is a dictionary (from Tavily search)
        elif isinstance(docs[0], dict):
            return "\n\n".join(f"URL: {doc.get('url', 'N/A')}\nContent: {doc.get('content', '')}" for doc in docs)
        
        # Handle the case where Tavily returns a single string (e.g., on error)
        elif isinstance(docs, str):
            return docs
            
        return "Could not format documents."

    # The final chain, now more efficient
    rag_chain = (
        {
            "context": combined_retriever, # Runs the parallel retrievers
            "input": itemgetter("input")  # Passes the original input string through
        }
        | RunnablePassthrough.assign(
            # Format the combined context from both sources into a single string
            context=lambda x: f"--- Local Documents ---\n{format_docs(x['context']['retrieved_docs'])}\n\n--- Web Search Results ---\n{format_docs(x['context']['web_results'])}"
        )
        | prompt
        | llm
    )
    
    print("RAG chain created.")
    return rag_chain

# --- 3) Main Chatbot Loop ---
def run_chatbot(rag_chain):
    """Runs the main chatbot loop using the RAG chain."""
    print("\nStarting chatbot. Type 'exit' to quit.")
    
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
            # The chain now expects a dictionary with an "input" key
            response_generator = rag_chain.stream({"input": query})
            
            print("\nChatBot: ", end="", flush=True)
            full_response = ""
            for chunk in response_generator:
                print(chunk.content, end="", flush=True)
                full_response += chunk.content
            print() # Newline after streaming is complete

        except Exception as e:
            print(f"Bot: An error occurred while answering: {e}")
            continue

# --- Main Execution ---
if __name__ == "__main__":
    collection_name = _auto_collection_name()
    try:
        vectorstore = get_vectorstore(collection_name)
        rag_chain = create_rag_chain(vectorstore)
        run_chatbot(rag_chain)
        
    except psycopg.OperationalError as e:
        print(f"\n--- DATABASE CONNECTION ERROR ---")
        print(f"Error: {e}")
        print(f"Please ensure PostgreSQL is running and the database '{DB_NAME}' exists.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")