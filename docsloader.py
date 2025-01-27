# File: agentic_rag_openai.py
import os
import glob
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI, OpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
from operator import itemgetter

# from typing import Tuple
# from langchain_core.runnables import RunnableLambda

# Update the state definition to include sources
class GraphState(TypedDict):
    question: str
    context: List[str]
    answer: str
    sources: List[str]  # Track document/web sources


# Load environment variables
load_dotenv()

# Configuration
DOCS_DIR = "./docs"  # For .txt files
VECTOR_DB_DIR = "./vector_db"
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
LLM_MODEL = "gpt-4o-mini"

llm = ChatOpenAI(
    model=LLM_MODEL,
    temperature=0,
    openai_api_key=os.getenv("OPENAI_API_KEY")  # Explicit key access
    )

def initialize_components():
    try:
        # Find all text files using glob
        txt_files = glob.glob(os.path.join(DOCS_DIR, '**/*.txt'), recursive=True)
        
        if not txt_files:
            raise ValueError(f"No .txt files found in {DOCS_DIR}")

        print(f"Found {len(txt_files)} text files")
        
        # Load documents with error handling
        documents = []
        for txt_file in txt_files:
            try:
                loader = TextLoader(txt_file, autodetect_encoding=True)
                documents.extend(loader.load())
                print(f"✓ Loaded {txt_file}")
            except Exception as e:
                print(f"✗ Error loading {txt_file}: {str(e)}")
                continue

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
        # retriever = vectorstore.as_retriever()

        retriever = vectorstore.as_retriever(
                    search_type="similarity_score_threshold",
                    search_kwargs={"k": 5, "score_threshold": 0.6}
                )
        
        # LLM setup

        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable not set")
            
        # Rest of the initialization code remains the same as previous version...
        # [Keep the document loading, splitting, and vector store code here]
        

        # llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
        
        # RAG chain
        prompt_template = """Answer based on context:
        {context}
        
        Question: {question}"""
        
        rag_chain = (
            {
                 "context": itemgetter("context"),
                 "question": RunnablePassthrough()}
            | ChatPromptTemplate.from_template(prompt_template)
            | llm
            | StrOutputParser()
        )
        
        # Tavily tool
        tavily_tool = TavilySearchResults()
        
        return embeddings, vectorstore, rag_chain, tavily_tool, retriever

    except Exception as e:
        print(f"Initialization failed: {str(e)}")
        raise
# Define state (same as before)
class GraphState(TypedDict):
    question: str
    context: List[str]
    answer: str

# Define nodes (same as before)
def retrieve_local_docs(state: GraphState, vectorstore: Chroma, embeddings: HuggingFaceEmbeddings):
    print("🕵️ Searching local documents...")
    question = state["question"]
    
    # Get the Chroma collection directly
    chroma_collection = vectorstore._collection
    
    # Convert query to embedding
    query_embedding = embeddings.embed_query(question)
    
    # Query with score filtering
    results = chroma_collection.query(
        query_embeddings=[query_embedding],
        n_results=5,
        include=["documents", "metadatas", "distances"]
    )
    
    # Process results with scores
    docs = [
        (doc, meta, dist) 
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        )
    ]
    
    # Filter by score threshold (0.5 in this case)
    filtered_docs = [doc for doc, meta, dist in docs if dist < 0.5]
    
    context = [doc[0] for doc in filtered_docs]
    sources = list(set([meta.get("source", "Unknown") for _, meta, _ in filtered_docs]))
    
    if not context:
        print("❌ No relevant local documents found")
        return {"context": None, "sources": []}
    
    print(f"✅ Found {len(context)} relevant documents (Scores: {[dist for _, _, dist in docs]})")
    return {"context": context, "sources": sources}

def generate_local_answer(state: GraphState):
    print("🧠 Generating answer from local documents...")
    question = state["question"]
    context = state["context"]
    sources = state.get("sources", [])  # Safe access with default
    
    formatted_context = "\n\n".join([
        f"Document excerpt from {sources[i] if i < len(sources) else 'Unknown Source'}:\n{text}"
        for i, text in enumerate(context)
    ])
    
    prompt = ChatPromptTemplate.from_template(
        """Answer the question based only on the following context:
        {context}
        
        Question: {question}
        Include document sources in your answer."""
    )
    


    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({"context": formatted_context, "question": question})
    
    return {
        "answer": f"{response}\n\nSources: {', '.join(sources) if sources else 'Local Documents'}",
        "sources": sources
    }

def search_web(state: GraphState):
    print("🌐 Searching web with Tavily...")
    question = state["question"]
    
    # Get web results with proper error handling
    try:
        results = tavily_tool.invoke({"query": question, "max_results": 5})
        context = [f"Web Result {i+1}: {result['content']}" 
                  for i, result in enumerate(results)]
        sources = [result['url'] for result in results]
        
        # Create web-specific chain
        web_prompt = ChatPromptTemplate.from_template(
            "Answer based on web results:\n{context}\n\nQuestion: {question}"
        )
        web_chain = web_prompt | llm | StrOutputParser()
        
        response = web_chain.invoke({
            "context": "\n\n".join(context),
            "question": question
        })
        
        return {
            "answer": f"{response}\n\nSources: {', '.join(sources)}",
            "sources": sources
        }
        
    except Exception as e:
        print(f"⚠️ Tavily search failed: {str(e)}")
        return {
            "answer": "I couldn't access external sources. Please try again later.",
            "sources": []
        }


# Initialize components
embeddings, vectorstore, rag_chain, tavily_tool, retriever = initialize_components()

# Create workflow (same as before)
workflow = StateGraph(GraphState)
workflow.add_node("retrieve_local", lambda state: retrieve_local_docs(state, vectorstore, embeddings))
workflow.add_node("generate_local_answer", generate_local_answer)
workflow.add_node("search_web", search_web)
workflow.set_entry_point("retrieve_local")

def should_use_local_docs(state: GraphState):
    """Improved routing logic with better null checks"""
    # Check if we have valid context (not None and not empty list)
    if state.get("context") and len(state["context"]) > 0:
        return "generate_local_answer"
    else: 
        return "search_web"

workflow.add_conditional_edges(
    "retrieve_local",
    should_use_local_docs,
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
        
        # Initialize state with all required keys
        response = app.invoke({
            "question": query,
        })
        
        print(f"\nAnswer: {response['answer']}")

if __name__ == "__main__":
    main()