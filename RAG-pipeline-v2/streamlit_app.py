import streamlit as st
import os
import sys
import psycopg
import socket
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

# psycopg v3 URL
PG_CONN = f"postgresql+psycopg://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# --- RAG Prompt for synthesizing answers ---
RAG_PROMPT_TEMPLATE = """
You are a research assistant. Use the following retrieved context from local documents and web search results to answer the question.
If you don't know the answer from the context, just say that you don't know.
Do not use any outside knowledge. Use only the local vector store and those found in www.jw.org.
Provide scriptural citations if they are present in the context.

CONTEXT:
{context}

QUESTION:
{input}

ANSWER:
"""

def _auto_collection_name(base="documents", embed_model=EMBEDDING_MODEL):
    """Generates a collection name based on the embedding model."""
    suffix = embed_model.lower().replace(":", "_").replace("-", "_").replace("/", "_")
    return f"{base}__{suffix}"

@st.cache_resource
def get_vectorstore(collection_name):
    """Connects to an existing vector store."""
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
    vectorstore = PGVector(
        embeddings=embeddings,
        collection_name=collection_name,
        connection=PG_CONN,
        use_jsonb=True,
    )
    return vectorstore

@st.cache_resource
def create_rag_chain(_vectorstore):
    """Creates a RAG chain that queries local docs and the web in parallel."""
    llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)
    prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

    retriever = _vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 10, "score_threshold": 0.2},
    )

    web_search_tool = TavilySearchResults(max_results=5)
    local_retriever_chain = itemgetter("input") | retriever
    web_search_chain = (
        itemgetter("input")
        | RunnableLambda(lambda q: q + " site:www.jw.org")
        | web_search_tool
    )

    combined_retriever = RunnableParallel(
        retrieved_docs=local_retriever_chain,
        web_results=web_search_chain,
    )

    def format_docs(docs):
        if not docs:
            return "No information found."
        if isinstance(docs[0], Document):
            return "\n\n".join(doc.page_content for doc in docs)
        elif isinstance(docs[0], dict):
            return "\n\n".join(f"URL: {doc.get('url', 'N/A')}\nContent: {doc.get('content', '')}" for doc in docs)
        elif isinstance(docs, str):
            return docs
        return "Could not format documents."

    rag_chain = RunnableParallel(
        {"context": combined_retriever, "input": itemgetter("input")}
    ) | RunnableParallel(
        answer=(
            RunnablePassthrough.assign(
                context=lambda x: f"--- Local Documents ---\n{format_docs(x['context']['retrieved_docs'])}\n\n--- Web Search Results ---\n{format_docs(x['context']['web_results'])}"
            )
            | prompt
            | llm
        ),
        sources=itemgetter("context"),
    )
    return rag_chain

st.title("JW Chat")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is your question?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            collection_name = _auto_collection_name()
            vectorstore = get_vectorstore(collection_name)
            rag_chain = create_rag_chain(vectorstore)
            
            response = rag_chain.invoke({"input": prompt})
            
            answer = response.get("answer", "")
            if hasattr(answer, 'content'):
                answer_content = answer.content
            else:
                answer_content = str(answer)

            st.markdown(answer_content)
            
            sources = response.get("sources", {})
            retrieved_docs = sources.get("retrieved_docs", [])
            web_results = sources.get("web_results", [])

            if retrieved_docs or web_results:
                with st.expander("Sources"):
                    if retrieved_docs:
                        st.markdown("**Local Documents:**")
                        for doc in retrieved_docs:
                            source = doc.metadata.get("source", "Unknown")
                            st.markdown(f"- {os.path.basename(source)}")
                    if web_results:
                        st.markdown("**Web Results:**")
                        for result in web_results:
                            url = result.get("url", "N/A")
                            st.markdown(f"- {url}")
            
            st.session_state.messages.append({"role": "assistant", "content": answer_content})

        except Exception as e:
            st.error(f"An error occurred: {e}")

