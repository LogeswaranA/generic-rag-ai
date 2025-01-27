import streamlit as st
from PIL import Image
import os
import tempfile
import matplotlib.pyplot as plt

from transformers import pipeline
from sentence_transformers import SentenceTransformer
import chromadb
from langgraph.graph import StateGraph, END

from openai import OpenAI
from diffusers import StableDiffusionXLPipeline
import torch
import matplotlib.pyplot as plt

# Initialize models
embedding_model = SentenceTransformer('clip-ViT-B-32')
caption_model = pipeline("image-to-text", model="Salesforce/blip-image-captioning-large")

# Initialize ChromaDB
client = chromadb.PersistentClient(path="lighting_db")
collection = client.get_or_create_collection(name="interior_design")

# Process image pairs
def process_image_pairs(before_dir, after_dir):
    before_images = sorted([os.path.join(before_dir, f) for f in os.listdir(before_dir)])
    after_images = sorted([os.path.join(after_dir, f) for f in os.listdir(after_dir)])

    ids, embeddings, metadatas = [], [], []
    
    for idx, (before_path, after_path) in enumerate(zip(before_images, after_images)):
        # Process before image
        before_img = Image.open(before_path)
        before_embedding = embedding_model.encode(before_img).tolist()
        
        # Process after image
        after_img = Image.open(after_path)
        after_caption = caption_model(after_img)[0]['generated_text']
        
        # Store data
        ids.append(str(idx))
        embeddings.append(before_embedding)
        metadatas.append({"after_caption": after_caption})
    
    collection.upsert(ids=ids, embeddings=embeddings, metadatas=metadatas)

# Run preprocessing (adjust paths)
process_image_pairs("/Users/lokesh/projects/AIProjects/generic-ai-rag/before", "/Users/lokesh/projects/AIProjects/generic-ai-rag/after")

# STEP 3: LANGGRAPH WORKFLOW
from langgraph.graph import StateGraph, END
from typing import TypedDict, List
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

class GraphState(TypedDict):
    input_image: Image.Image
    query_embedding: List[float]
    retrieved_captions: List[str]
    recommendation: str

# Initialize LLM
llm = ChatOpenAI(model="gpt-4-1106-preview", temperature=0.3)

# Define nodes
def encode_image(state: GraphState):
    image = state["input_image"]
    return {"query_embedding": embedding_model.encode(image).tolist()}

def retrieve_context(state: GraphState):
    results = collection.query(
        query_embeddings=[state["query_embedding"]],
        n_results=3
    )
    return {"retrieved_captions": [m["after_caption"] for m in results["metadatas"][0]]}

def generate_recommendation(state: GraphState):
    captions = "\n- ".join(state["retrieved_captions"])
    
    messages = [
        HumanMessage(content=f"""
        You are an expert interior lighting designer. Analyze these successful transformations:
        
        {captions}
        
        Provide detailed lighting recommendations for this new space considering:
        1. Fixture types and placement
        2. Lighting temperature and intensity
        3. Energy efficiency considerations
        4. Decorative elements
        5. Smart lighting options
        
        Structure your response in markdown with clear sections.
        """)
    ]
    
    response = llm.invoke(messages)
    return {"recommendation": response.content}

# Build workflow
workflow = StateGraph(GraphState)
workflow.add_node("encode_image", encode_image)
workflow.add_node("retrieve_context", retrieve_context)
workflow.add_node("generate_recommendation", generate_recommendation)

workflow.set_entry_point("encode_image")
workflow.add_edge("encode_image", "retrieve_context")
workflow.add_edge("retrieve_context", "generate_recommendation")
workflow.add_edge("generate_recommendation", END)

app = workflow.compile()

# STEP 4: USAGE EXAMPLE
def get_lighting_recommendation(image_path):
    user_image = Image.open(image_path)
    results = app.invoke({"input_image": user_image})
    return results["recommendation"]

# Example usage
recommendation = get_lighting_recommendation("/Users/lokesh/projects/AIProjects/generic-ai-rag/image.png")
print(recommendation)