# File: agentic_rag_openai.py
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import os
import glob
from dotenv import load_dotenv

load_dotenv()

# Configuration
DOCS_DIR = "./docs"
VECTOR_DB_DIR = "./vector_db"
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
LLM_MODEL = "gpt-3.5-turbo"

llm = ChatOpenAI(model=LLM_MODEL, temperature=0,openai_api_key=os.getenv("OPENAI_API_KEY"))  # Explicit key access


def initialize_components():
    try:
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        # Load documents with error handling
        documents = []

        # Find all text files using glob
        txt_files = glob.glob(os.path.join(DOCS_DIR, '**/*.txt'), recursive=True)

                
        if not txt_files:
            raise ValueError(f"No .txt files found in {DOCS_DIR}")

        print(f"Found {len(txt_files)} text files")

        for txt_file in txt_files:
            try:
                loader = TextLoader(txt_file, autodetect_encoding=True)
                documents.extend(loader.load())
                print(f"✓ Loaded text file: {os.path.basename(txt_file)}")
            except Exception as e:
                print(f"✗ Error loading {txt_file}: {str(e)}")

        # Load PDF files
        pdf_files = glob.glob(os.path.join(DOCS_DIR, '**/*.pdf'), recursive=True)
        for pdf_file in pdf_files:
            try:
                loader = PyPDFLoader(pdf_file)
                pages = loader.load()
                documents.extend(pages)
                print(f"✓ Loaded PDF file: {os.path.basename(pdf_file)} ({len(pages)} pages)")
            except Exception as e:
                print(f"✗ Error loading {pdf_file}: {str(e)}")

        if not documents:
            raise ValueError("No valid documents could be loaded")

        print(f"Successfully loaded {len(documents)} documents")
        
        # Document splitting
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        splits = text_splitter.split_documents(documents)
        print(f"Created {len(splits)} text chunks")

        # Embeddings
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        
        # Vector store setup
        if os.path.exists(VECTOR_DB_DIR):
            import shutil
            shutil.rmtree(VECTOR_DB_DIR)
            
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory=VECTOR_DB_DIR
        )
        retriever = vectorstore.as_retriever()
        
        # LLM setup

        
        # RAG chain
        prompt_template = """Answer based on context:
        {context}
        
        Question: {question}"""
        
        rag_chain = (
            {"question": RunnablePassthrough()}
            | ChatPromptTemplate.from_template(prompt_template)
            | llm
            | StrOutputParser())
        
        # Tavily tool
        tavily_tool = TavilySearchResults()
        
        return rag_chain, tavily_tool, retriever

    except Exception as e:
        print(f"Initialization failed: {str(e)}")
        raise

# Define state (same as before)
class GraphState(TypedDict):
    question: str
    context: List[str]
    answer: str

# Define nodes (same as before)
def retrieve_local_docs(state: GraphState):
    print("🕵️ Searching local documents...")
    question = state["question"]
    docs = retriever.invoke(question)
    context = [doc.page_content for doc in docs]
    
    if len(context) == 0:
        print("❌ No relevant local documents found")
        return {"context": None}
    
    print(f"✅ Found {len(context)} relevant documents")
    return {"context": context}

def generate_local_answer(state: GraphState):
    print("🧠 Generating answer from local documents...")
    question = state["question"]
    context = state["context"]
    
    # Create a dedicated answer generation chain
    prompt_template = """You are a HR Expert so answer the question based on the following context:
    {context}
    
    Question: {question}

    Answer:
        Summarize the information with the following information
        - Candidate Name
        - Candidate Skill Set
        - Candidate Total Experience
        - Last company worked
        - College Studies
        - Degree received

        Summary about this candidate compared to other candidates 
    """
    
    answer_chain = (
        ChatPromptTemplate.from_template(prompt_template)
        | llm
        | StrOutputParser()
    )
    
    formatted_context = "\n\n".join(context)
    response = answer_chain.invoke({
        "context": formatted_context,
        "question": question
    })
    
    return {"answer": response}

def search_web(state: GraphState):
    # Uses Tavily search results
    # Formats web context differently from local docs
    # Generates answer specifically from web sources
    print("🌐 Searching web with Tavily...")
    question = state["question"]
    
    # Get web results
    results = tavily_tool.invoke({"query": question})
    web_context = [result["content"] for result in results]
    
    # Generate answer from web
    prompt = ChatPromptTemplate.from_template(
        """Answer the question based on web search results:
        {context}
        
        Question: {question}"""
    )
    
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({"context": "\n\n".join(web_context), "question": question})
    return {"answer": response}

# Initialize components
rag_chain, tavily_tool, retriever = initialize_components()

# Create workflow (same as before)
workflow = StateGraph(GraphState)
workflow.add_node("retrieve_local", retrieve_local_docs)
workflow.add_node("generate_local_answer", generate_local_answer)
workflow.add_node("search_web", search_web)
workflow.set_entry_point("retrieve_local")

def route(state):
    if state.get("context"):
        return "generate_local_answer"
    return "search_web"

workflow.add_conditional_edges(
    "retrieve_local",
    route,
    {
        "generate_local_answer": "generate_local_answer",
        "search_web": "search_web"
    }
)
workflow.add_edge("generate_local_answer", END)
workflow.add_edge("search_web", END)

app = workflow.compile()

# Run interface (same as before)
def main():
    print("🤖 Agentic RAG System (OpenAI)")
    print("Type 'exit' to quit")
    
    while True:
        query = input("\nQuestion: ")
        if query.lower() == "exit":
            break
        
        response = app.invoke({"question": query})
        print(f"\nAnswer: {response['answer']}")

if __name__ == "__main__":
    main()