# Map language code to full language name for prompt clarity
LANG_CODE_TO_NAME = {
    "en": "English",
    "ko": "Korean",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "ja": "Japanese",
    "zh-cn": "Chinese",
    # Add more as needed
}

# Map language code to Unicode ranges for character detection
LANG_UNICODE_RANGES = {
    "en": [('\u0041', '\u005a'), ('\u0061', '\u007a')],  # Latin uppercase/lowercase
    "ko": [('\u1100', '\u11ff'), ('\u3130', '\u318f'), ('\uac00', '\ud7af')],  # Korean Hangul
    "es": [('\u0041', '\u005a'), ('\u0061', '\u007a')],  # Latin
    "fr": [('\u0041', '\u005a'), ('\u0061', '\u007a')],  # Latin
    "de": [('\u0041', '\u005a'), ('\u0061', '\u007a')],  # Latin
    "ja": [('\u3040', '\u309f'), ('\u30a0', '\u30ff'), ('\u4e00', '\u9fff')],  # Hiragana, Katakana, Kanji
    "zh-cn": [('\u4e00', '\u9fff')],  # Han
    # Add more as needed
}
from langdetect import detect
import streamlit as st
import os
import sys
import psycopg
import socket
import requests
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from operator import itemgetter
from langchain.schema.document import Document
# from langchain_community.tools.tavily_search import TavilySearchResults  # Removed
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

###############################
## SYSTEM: ABSOLUTE RULE ##
###############################
You MUST answer ONLY in {language_name}. Do NOT use any other language. If you break this rule, your answer will be rejected.

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

def create_rag_chain_new(_vectorstore, score_threshold=0.9):
    """Creates a RAG chain that queries local docs and the web in parallel using MCP server."""
    llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)
    prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

    retriever = _vectorstore.as_retriever(
        search_kwargs={"k": 5, "score_threshold": 0.9},
    )

    local_retriever_chain = itemgetter("input") | retriever

    def mcp_web_search(inputs):
        # inputs: dict with 'input' (query) and 'lang_code', 'max_results'
        query = inputs.get("input", "")
        lang_code = inputs.get("lang_code", "en")
        max_results = inputs.get("max_results", 5)
        url = "http://127.0.0.1:8000/search/"
        payload = {"query": query, "lang_code": lang_code, "num": max_results, "full_text": False}
        debug_info = {"payload": payload}
        try:
            resp = requests.post(url, json=payload, timeout=30)
            debug_info["status_code"] = resp.status_code
            debug_info["response_text"] = resp.text
            resp.raise_for_status()
            data = resp.json()
            debug_info["json"] = data
            # Return both data and debug info
            return {"results": data if isinstance(data, list) else [], "_mcp_debug_info": debug_info}
        except Exception as e:
            debug_info["error"] = str(e)
            return {"results": [], "_mcp_debug_info": debug_info}


    # --- Debuggable Web Search Chain (MCP) ---
    def extract_results(obj):
        # Helper to extract 'results' from mcp_web_search output or just pass through if already a list
        if isinstance(obj, dict) and "results" in obj:
            return obj["results"]
        elif isinstance(obj, list):
            return obj
        return []

    raw_web_search_chain = RunnableLambda(mcp_web_search)

    # For all queries, retrieve from local docs (may return empty if no matches)
    def conditional_retriever(inputs):
        return {"retrieved_docs": local_retriever_chain.invoke({"input": inputs["input"]})}

    combined_retriever = RunnableLambda(conditional_retriever)

    def format_docs(docs, lang_code=None):
        if not docs: return "No information found."
        if isinstance(docs[0], Document):
            # Filter out garbled documents and those not in the target language
            def is_garbled(text):
                if not text: return True
                alpha_count = sum(1 for c in text if c.isalpha())
                total_count = len(text)
                if total_count == 0: return True
                alpha_ratio = alpha_count / total_count
                # Consider garbled if less than 40% alphabetic characters
                return alpha_ratio < 0.4
            
            def matches_language(text, target_lang):
                from langdetect import detect
                try:
                    detected = detect(text)
                    return detected == target_lang
                except:
                    return False  # If detection fails, assume mismatch
            
            filtered_docs = []
            for doc in docs:
                if not is_garbled(doc.page_content):
                    # Check language match (but skip for 'en' since local docs are English)
                    if lang_code == 'en' or matches_language(doc.page_content, lang_code):
                        filtered_docs.append(doc)
            
            return "\n\n".join(doc.page_content for doc in filtered_docs) if filtered_docs else "No information found."
        elif isinstance(docs[0], dict):
            # Prefer 'url' or 'link' for URL, 'content' or 'snippet' for content
            return "\n\n".join(
                f"Title: {doc.get('title', '')}\nSnippet: {doc.get('snippet', doc.get('content', ''))}"
                for doc in docs
            )
        elif isinstance(docs, str):
            return docs
        return "Could not format documents."


    def clean_and_truncate(text, max_length=8000):
        import re
        text = re.sub(r'\n+', '\n', text)
        text = re.sub(r'[\s\u200b]+', ' ', text)
        text = text.strip()
        return text[:max_length]

    # Store the last context for debugging
    if "last_llm_context" not in st.session_state:
        st.session_state.last_llm_context = None
    def build_context(x):
        try:
            local = format_docs(x['context']['retrieved_docs'], x["lang_code"])
            # Use pre-fetched MCP results instead of calling again
            web_results = x.get("mcp_results", [])
            web = format_docs(web_results)
            context = f"--- Web Search Results ---\n{web}\n\n--- Local Documents ---\n{local}"
            # Increase context length by 50% if source text is available
            has_sources = local != "No information found." or web.strip()
            max_length = 12000 if has_sources else 8000
            cleaned = clean_and_truncate(context, max_length)
            print(f"Built context: {cleaned[:500]}...")  # Print first 500 chars to terminal
            st.session_state.last_llm_context = cleaned
            return cleaned
        except Exception as e:
            print(f"Error in build_context: {e}")
            st.session_state['build_error'] = str(e)
            return "Error in build_context"

    def capture_prompt(x):
        st.session_state['llm_prompt'] = x
        print("LLM PROMPT BEING SENT:\n", x)
        return x

    rag_chain = (
        {"context": combined_retriever, "input": itemgetter("input"), "lang_code": itemgetter("lang_code"), "language_name": itemgetter("language_name"), "mcp_results": itemgetter("mcp_results")}
    ) | RunnableParallel(
        answer=(
            RunnablePassthrough.assign(
                context=build_context,
                lang_code=itemgetter("lang_code"),
                language_name=itemgetter("language_name"),
                mcp_results=itemgetter("mcp_results")
            )
            | prompt
            | RunnableLambda(capture_prompt)
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
max_results = st.sidebar.number_input(
    "Maximum Web Search Results",
    min_value=1,
    max_value=20,
    value=5,
    step=1,
    help="Set the maximum number of web search results to return from the MCP server."
)
show_debug = st.sidebar.checkbox("Show Debug Information", value=False, help="Toggle to show or hide debugging details.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is your question?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Auto-detect language of the query
    try:
        detected_lang = detect(prompt)
    except Exception:
        detected_lang = "en"
    lang_code = detected_lang  # Use auto-detected language
    language_name = LANG_CODE_TO_NAME.get(lang_code, lang_code)

    with st.chat_message("assistant"):
        # --- Direct MCP Web Search Call ---
        mcp_url = "http://127.0.0.1:8000/search/"
        mcp_payload = {"query": prompt, "lang_code": lang_code, "num": max_results, "full_text": False}
        mcp_results = []
        mcp_debug = {"payload": mcp_payload}
        try:
            mcp_resp = requests.post(mcp_url, json=mcp_payload, timeout=30)
            mcp_debug["status_code"] = mcp_resp.status_code
            mcp_debug["response_text"] = mcp_resp.text
            mcp_resp.raise_for_status()
            mcp_data = mcp_resp.json()
            mcp_results = mcp_data.get("results", []) if isinstance(mcp_data, dict) else []
            mcp_debug["json"] = mcp_data
        except Exception as e:
            mcp_debug["error"] = str(e)

        # --- RAG Chain for answer (local docs only) ---
        try:
            collection_name = _auto_collection_name()
            vectorstore = get_vectorstore(collection_name)
            retriever = vectorstore.as_retriever(search_kwargs={"k": 5, "score_threshold": 0.9})

            # Get retrieved docs for all queries
            retrieved_docs = retriever.invoke(prompt)

            # Filter retrieved docs: remove garbled and non-matching language
            def is_garbled(text):
                if not text: return True
                alpha_count = sum(1 for c in text if c.isalpha())
                total_count = len(text)
                if total_count == 0: return True
                alpha_ratio = alpha_count / total_count
                return alpha_ratio < 0.4
            
            def matches_language(text, target_lang):
                from langdetect import detect
                try:
                    detected = detect(text)
                    return detected == target_lang
                except:
                    return False
            
            filtered_retrieved_docs = []
            for doc in retrieved_docs:
                if not is_garbled(doc.page_content):
                    if lang_code == 'en' or matches_language(doc.page_content, lang_code):
                        filtered_retrieved_docs.append(doc)
            
            retrieved_docs = filtered_retrieved_docs  # Use filtered for the rest

                        # Check if no results from both sources
            if not retrieved_docs and not mcp_results:
                answer_content = "No information is found in both local documents and www.jw.org."
                st.markdown(answer_content)
            else:
                rag_chain = create_rag_chain_new(vectorstore)

                response = rag_chain.invoke({"input": prompt, "lang_code": lang_code, "language_name": language_name, "max_results": max_results, "mcp_results": mcp_results})

                answer = response.get("answer", "")
                answer_content = answer.content if hasattr(answer, 'content') else str(answer)

                # Check if the answer is gibberish (e.g., mostly punctuation or too short)
                def is_gibberish(text):
                    if not text or len(text.strip()) < 5:
                        return True
                    alpha_count = sum(1 for c in text if c.isalpha())
                    total_count = len(text)
                    alpha_ratio = alpha_count / total_count if total_count > 0 else 0
                    return alpha_ratio < 0.3  # Less than 30% alphabetic characters

                if is_gibberish(answer_content):
                    answer_content = f"Unable to generate a coherent response in {language_name} due to context quality issues. Please try rephrasing your question or check the sources for relevant information."
                    # Skip language check for system messages
                    detected_answer_lang = lang_code

                # Post-processing: check if answer is in the target language
                else:
                    from langdetect import detect as _detect_lang
                    try:
                        detected_answer_lang = _detect_lang(answer_content)
                    except Exception:
                        detected_answer_lang = "unknown"

                    # Additional check: if answer contains characters typical of the target language, override detection
                    ranges = LANG_UNICODE_RANGES.get(lang_code, [])
                    if ranges and any(any(start <= c <= end for start, end in ranges) for c in answer_content):
                        detected_answer_lang = lang_code

                if detected_answer_lang != lang_code and detected_answer_lang != "unknown":
                    st.warning(f"⚠️ The answer may not be fully in the requested language ({language_name}). Detected: {LANG_CODE_TO_NAME.get(detected_answer_lang, detected_answer_lang)}. Please try rephrasing your question or check the context.")
                st.markdown(answer_content)

            # --- Display Final Sources ---
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

                # --- MCP Web Results (direct call) ---
                st.markdown("**Web Results (Direct MCP Call):**")
                if mcp_results:
                    for result in mcp_results:
                        url = result.get('url', result.get('link', 'N/A'))
                        title = result.get('title', '')
                        st.markdown(f"- [{title}]({url})" if title else f"- {url}")
                else:
                    st.markdown("No web results found.")

            # --- Display Debugging Info ---
            if show_debug:
                with st.expander("Debugging Info"):
                    st.markdown("### MCP Server Debug Info (for last query)")
                    st.json(mcp_debug)

            st.session_state.messages.append({"role": "assistant", "content": answer_content})

        except Exception as e:
            import traceback
            st.error(f"An error occurred: {e}\n\n{traceback.format_exc()}")