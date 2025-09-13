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
If there is a conflict or overlap between the local documents and the web search results, you must prioritize the information from the web search results from www.jw.org.
Always make sure to include scriptural citations if they are present in the context.

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
def create_rag_chain(_vectorstore, score_threshold=0.8):
    """Creates a RAG chain that queries local docs and the web in parallel."""
    llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)
    prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

    retriever = _vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 10, "score_threshold": 0.2},
    )

    web_search_tool = TavilySearchResults(max_results=10)
    local_retriever_chain = itemgetter("input") | retriever

    def filter_jw_org_only(docs):
        if not isinstance(docs, list): return []
        return [doc for doc in docs if "jw.org" in doc.get("url", "")]

    def filter_by_score(docs):
        if not isinstance(docs, list): return []
        return [doc for doc in docs if doc.get("score", 0) >= score_threshold]

    # --- Debuggable Web Search Chain ---
    # 1. Initial search
    raw_web_search_chain = (
        itemgetter("input")
        | RunnableLambda(lambda q: q + " site:www.jw.org")
        | web_search_tool
    )
    # 2. Filter for jw.org
    jw_filtered_chain = raw_web_search_chain | RunnableLambda(filter_jw_org_only)
    # 3. Filter by score
    score_filtered_chain = jw_filtered_chain | RunnableLambda(filter_by_score)
    # ---

    combined_retriever = RunnableParallel(
        retrieved_docs=local_retriever_chain,
        web_results_final=score_filtered_chain,
        # Pass through the intermediate steps for debugging
        web_results_raw=raw_web_search_chain,
        web_results_jw_filtered=jw_filtered_chain,
    )

    def format_docs(docs):
        if not docs: return "No information found."
        if isinstance(docs[0], Document):
            return "\n\n".join(doc.page_content for doc in docs)
        elif isinstance(docs[0], dict):
            return "\n\n".join(f"URL: {doc.get('url', 'N/A')}\nContent: {doc.get('content', '')}" for doc in docs)
        elif isinstance(docs, str):
            return docs
        return "Could not format documents."

    rag_chain = (
        {"context": combined_retriever, "input": itemgetter("input")}
    ) | RunnableParallel(
        answer=(
            RunnablePassthrough.assign(
                context=lambda x: f"--- Local Documents ---\n{format_docs(x['context']['retrieved_docs'])}\n\n--- Web Search Results ---\n{format_docs(x['context']['web_results_final'])}"
            )
            | prompt
            | llm
        ),
        sources=itemgetter("context"),
    )
    return rag_chain

st.markdown(
    '<h1 style="color: #0077be;">JW Research (LLM, RAG and JW.Org)</h1>',
    unsafe_allow_html=True
)

# --- UI Elements ---
st.sidebar.title("Settings")
score_threshold = st.sidebar.slider(
    "Web Relevance Threshold", 
    min_value=0.0, 
    max_value=1.0, 
    value=0.4, 
    step=0.05,
    help="Adjust how relevant web search results must be to be included. Lower values include more results."
)

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
            rag_chain = create_rag_chain(vectorstore, score_threshold)
            
            response = rag_chain.invoke({"input": prompt})
            
            answer = response.get("answer", "")
            answer_content = answer.content if hasattr(answer, 'content') else str(answer)
            st.markdown(answer_content)
            
            # --- Display Final Sources ---
            sources = response.get("sources", {})
            retrieved_docs = sources.get("retrieved_docs", [])
            web_results = sources.get("web_results_final", [])

            if retrieved_docs or web_results:
                with st.expander("Sources"):
                    if retrieved_docs:
                        st.markdown("**Local Documents:**")
                        # Group pages by source document
                        source_pages = {}
                        for doc in retrieved_docs:
                            source_file = os.path.basename(doc.metadata.get("source", "Unknown"))
                            page_num = doc.metadata.get("page", -1)
                            if source_file not in source_pages:
                                source_pages[source_file] = set()
                            if page_num != -1:
                                # Add 1 to page number as it's often 0-indexed
                                source_pages[source_file].add(str(page_num + 1))
                        
                        # Display the grouped sources
                        for source_file, pages in sorted(source_pages.items()):
                            if pages:
                                page_str = ", ".join(sorted(list(pages), key=int))
                                st.markdown(f"- {source_file} (pages: {page_str})")
                            else:
                                st.markdown(f"- {source_file}")

                    if web_results:
                        st.markdown("**Web Results:**")
                        for result in web_results:
                            st.markdown(f"- {result.get('url', 'N/A')}")
            
            # --- Display Debugging Info ---
            with st.expander("Debugging Info"):
                st.markdown("### Raw Web Search Results")
                st.json(sources.get("web_results_raw", []))
                st.markdown("### After Filtering for JW.ORG")
                st.json(sources.get("web_results_jw_filtered", []))
                st.markdown("### Final Web Results (After Score Filter)")
                st.json(sources.get("web_results_final", []))

            st.session_state.messages.append({"role": "assistant", "content": answer_content})

        except Exception as e:
            st.error(f"An error occurred: {e}")