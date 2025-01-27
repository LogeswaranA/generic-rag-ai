import streamlit as st
from PIL import Image
import os
import tempfile
import matplotlib.pyplot as plt
from typing import TypedDict, List

from transformers import pipeline
from sentence_transformers import SentenceTransformer
import chromadb
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from openai import OpenAI
from diffusers import StableDiffusionXLPipeline
import torch
import matplotlib.pyplot as plt
from dotenv import load_dotenv

load_dotenv()
# Initialize models with caching

class GraphState(TypedDict):
    input_image: Image.Image
    query_embedding: List[float]
    retrieved_captions: List[str]
    text_recommendation: str
    visualization_prompt: str
    generated_image: Image.Image


@st.cache_resource
def load_llm():
    return ChatOpenAI(model="gpt-4-1106-preview", temperature=0.3, openai_api_key=os.getenv("OPENAI_API_KEY"))

@st.cache_resource
def load_models():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    models = {
        "embedding": SentenceTransformer('clip-ViT-B-32').to(device),
        "caption": pipeline("image-to-text", 
                          model="Salesforce/blip-image-captioning-large",
                          device=0 if device == "mps" else -1),
        "chroma": chromadb.PersistentClient(path="lighting_db").get_collection("interior_design")
    }
    
    if not st.secrets.get("USE_DALLE", False):
        from diffusers import DiffusionPipeline
        
        # Use CPU-friendly model
        models["diffusion"] = DiffusionPipeline.from_pretrained(
            "OFA-Sys/small-stable-diffusion-v0",
            torch_dtype=torch.float32
        ).to(device)
    
    return models

# Modified workflow functions

def encode_image(state: GraphState):
    models = load_models()
    image = state["input_image"]
    embedding = models["embedding"].encode(image).tolist()
    return {**state, "query_embedding": embedding}

def retrieve_context(state: GraphState):
    models = load_models()
    results = models["chroma"].query(
        query_embeddings=[state["query_embedding"]],
        n_results=3
    )
    captions = [m["after_caption"] for m in results["metadatas"][0]]
    return {**state, "retrieved_captions": captions}

def generate_recommendation(state: GraphState):
    print("generate_recomendation;;;;")
    llm = load_llm()
    print("llm;;;;",llm)

    captions = "\n- ".join(state["retrieved_captions"])
    
    response = llm.invoke([
        HumanMessage(content=f"""
        Analyze these lighting transformations:
        {captions}
        
        Generate both:
        1. Text recommendations (markdown formatted)
        2. A detailed visualization prompt for image generation
        """)
    ])
    
    content = response.content.split("## Visualization Prompt")[-1].strip()
    return {
        **state,
        "text_recommendation": response.content,
        "visualization_prompt": content
    }

def generate_visualization(state: GraphState):
    models = load_models()
    prompt = state["visualization_prompt"]
    
    if st.secrets.get("USE_DALLE", False):
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.images.generate(
            model="dall-e-3",
            prompt=f"Professional interior design visualization, {prompt}",
            size="1024x1024",
            quality="hd",
            n=1
        )
        image = Image.open(requests.get(response.data[0].url, stream=True).raw)
    else:
        image = models["diffusion"](
            prompt=f"High-quality interior design visualization, {prompt}",
            negative_prompt="poor quality, blurry, unrealistic",
            num_inference_steps=25
        ).images[0]
    
    return {**state, "generated_image": image}

def process_image(uploaded_image):
    models = load_models()
    
    # Process through workflow
    state = {"input_image": Image.open(uploaded_image)}
    state = encode_image(state)
    state = retrieve_context(state)
    state = generate_recommendation(state)
    state = generate_visualization(state)
    
    return state

# Streamlit UI
st.set_page_config(page_title="AI Lighting Designer", layout="wide")

st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6
    }
    .sidebar .sidebar-content {
        background: #ffffff
    }
</style>
""", unsafe_allow_html=True)

st.title("✨ AI Lighting Design Assistant")
st.subheader("Upload your space photo for lighting recommendations and visualization")

with st.sidebar:
    st.header("Configuration")
    use_dalle = st.checkbox("Use DALL-E 3 (requires OpenAI key)", value=True)
    if use_dalle:
        os.environ["OPENAI_API_KEY"] = st.text_input("OpenAI API Key", type="password")
    
    st.markdown("---")
    st.info("""
    **Instructions:**
    1. Upload a clear photo of your space
    2. Wait for processing (30-60 seconds)
    3. View lighting recommendations and visualization
    """)

uploaded_file = st.file_uploader("Upload room photo", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    with st.spinner("Analyzing your space..."):
        # Save uploaded file to temp
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        # Process image
        results = process_image(tmp_path)
        os.unlink(tmp_path)  # Cleanup temp file

    # Display results
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Your Original Space")
        st.image(results["input_image"], use_column_width=True)
    
    with col2:
        st.subheader("Proposed Lighting Design")
        st.image(results["generated_image"], use_column_width=True)
    
    with st.expander("📝 Detailed Lighting Recommendations"):
        st.markdown(results["text_recommendation"])
    
    # Add download buttons
    st.download_button(
        label="Download Recommendations",
        data=results["text_recommendation"],
        file_name="lighting_recommendations.md"
    )
    
    with tempfile.NamedTemporaryFile(suffix=".png") as tmp_img:
        results["generated_image"].save(tmp_img.name)
        st.download_button(
            label="Download Visualization",
            data=open(tmp_img.name, "rb"),
            file_name="lighting_proposal.png"
        )
else:
    st.info("👈 Upload a photo to get started")

# Add footer
st.markdown("---")
st.markdown("""
<style>
.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: white;
    color: black;
    text-align: center;
}
</style>
<div class="footer">
<p>Made with ❤️ using Streamlit, LangGraph, and AI</p>
</div>
""", unsafe_allow_html=True)