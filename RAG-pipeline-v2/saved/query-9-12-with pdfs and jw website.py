import os
import sys
import psycopg

# --- New Imports for Agent and Tools ---
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools.retriever import create_retriever_tool
from langchain_community.tools.tavily_search import TavilySearchResults
# --- REMOVED hub import, we will use a custom prompt ---
# -----------------------------------------

from langchain_ollama import OllamaEmbeddings, ChatOllama
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

# --- Make sure Tavily API Key is set ---
if not os.getenv("TAVILY_API_KEY"):
    print("TAVILY_API_KEY environment variable not set. Please get a key from https://tavily.com and set it.")
    sys.exit(1)
# -----------------------------------------

# psycopg v3 URL
PG_CONN = f"postgresql+psycopg://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# --- NEW: Custom Agent Prompt ---
AGENT_PROMPT_TEMPLATE = """
You are a helpful research assistant. Answer the user's question based on the context provided.

You have access to the following tools:

{tools}

To use a tool, please use the following format:

```
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
```

When you have a response to say to the user, or if you do not need to use a tool, you MUST use the format:

```
Thought: Do I need to use a tool? No
Final Answer: [your response here]
```

Your instructions for using the tools are:
- For questions that seem general or could be answered by official public information from jw.org, you should ALWAYS prefer the `tavily_search_results_json` tool first.
- For questions that seem to reference specific, internal, or historical documents you have been trained on, you should prefer the `local_document_search` tool.
- If you are not sure, you can use both tools.
- When using `tavily_search_results_json`, you MUST include `site:www.jw.org` in the query to restrict the search.
- When using `local_document_search`, you should use the user's original question as the query. DO NOT include `site:www.jw.org`.
- After using a tool, analyze the observation. If the information is sufficient, provide a final answer.
- Always include scriptural citations found in the source materials or documents.
- If the information is not sufficient, you may use the other tool to get more context.
- Synthesize the information from all observations to provide a comprehensive final answer.

Begin!

Question: {input}
Thought: {agent_scratchpad}
"""
# --------------------------------

def _auto_collection_name(base="documents", embed_model=EMBEDDING_MODEL):
    """Generates a collection name based on the embedding model to ensure consistency."""
    suffix = embed_model.lower().replace(":", "_").replace("-", "_").replace("/", "_")
    return f"{base}__{suffix}"

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

# --- 2) Create Tools for the Agent ---
def create_agent_tools(vectorstore):
    """Creates the tools the agent can use: a local DB retriever and an internet searcher."""
    print("Creating agent tools...")
    
    # Tool 1: Retriever for your local PDF database
    retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 10, "score_threshold": 0.2},
    )
    local_db_tool = create_retriever_tool(
        retriever,
        "local_document_search",
        "Searches and returns relevant information from your personal PDF documents. Use the user's original question as the input."
    )

    # Tool 2: Internet search for www.jw.org
    tavily_tool = TavilySearchResults(
        max_results=5,
        name="tavily_search_results_json", # Explicitly name the tool
        description="A search engine optimized for searching the jw.org website. You must include 'site:www.jw.org' in the query."
    )
    
    tools = [local_db_tool, tavily_tool]
    print("Tools created.")
    return tools

# --- 3) Create the Agent ---
def create_agent(tools):
    """Creates the agent that will use the tools."""
    print("Creating agent...")
    llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)
    
    # Use our new custom prompt
    prompt = PromptTemplate.from_template(AGENT_PROMPT_TEMPLATE)
    
    agent = create_react_agent(llm, tools, prompt)
    
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        #verbose=True, 
        verbose=False,
        handle_parsing_errors=True
    )

    print("Agent created.")
    return agent_executor

# --- 4) Main Chatbot Loop ---
def run_chatbot(agent_executor):
    """Runs the main chatbot loop using the agent."""
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
            # The agent will now decide how to add "site:www.jw.org" based on the prompt
            result = agent_executor.invoke({"input": query})
            
            print(f"\nChatBot: {result['output']}")

        except Exception as e:
            print(f"Bot: An error occurred while answering: {e}")
            continue

# --- Main Execution ---
if __name__ == "__main__":
    collection_name = _auto_collection_name()
    try:
        vectorstore = get_vectorstore(collection_name)
        tools = create_agent_tools(vectorstore)
        agent_executor = create_agent(tools)
        run_chatbot(agent_executor)
        
    except psycopg.OperationalError as e:
        print(f"\n--- DATABASE CONNECTION ERROR ---")
        print(f"Error: {e}")
        print("Please ensure PostgreSQL is running and the database '{DB_NAME}' exists.")
        print("Have you run the `ingest.py` script yet to create the collection?")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")