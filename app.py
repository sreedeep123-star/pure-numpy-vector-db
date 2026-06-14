import streamlit as st
import numpy as np
import os
from google import genai
from google.genai import types

# Import your custom data structures
from src.database import NumPyVectorDB
from src.hnsw import HNSWIndex
from src.chunker import DocumentProcessor

st.set_page_config(page_title="Custom HNSW Vector DB RAG", page_icon="📚", layout="wide")

# --- Persistent Session State Configuration ---
# Streamlit re-runs the whole script on user actions; session_state ensures our DB stays in memory
if "db" not in st.session_state:
    st.session_state.db = NumPyVectorDB(dimension=768) # text-embedding-004 outputs 768 dims
if "hnsw" not in st.session_state:
    st.session_state.hnsw = HNSWIndex(st.session_state.db, M=16, ef_construction=64, ef_search=32)
if "processed_chunks" not in st.session_state:
    st.session_state.processed_chunks = {} # Maps vector_id -> string chunk text

# --- Sidebar Configuration ---
st.sidebar.title("⚙️ Engine Configuration")
api_key_input = st.sidebar.text_input("Enter Gemini API Key:", type="password")

# Initialize GenAI Client if key is provided
client = None
if api_key_input:
    client = genai.Client(api_key=api_key_input)
elif os.environ.get("GEMINI_API_KEY"):
    client = genai.Client()
else:
    st.sidebar.warning("🔑 Please provide a Gemini API Key to run embeddings and completions.")

# --- Main Dashboard UI ---
st.title("📚 Pure NumPy Vector DB & HNSW RAG App")
st.write("Upload a technical document or textbook chapter to build a zero-framework semantic index entirely in memory.")

uploaded_file = st.file_uploader("Upload Document (PDF format)", type=["pdf"])

if uploaded_file and client:
    # Check if this document has already been processed to avoid re-embedding on rerun
    if "current_file" not in st.session_state or st.session_state.current_file != uploaded_file.name:
        with st.spinner("📄 Parsing file lines and generating sliding-window token chunks..."):
            processor = DocumentProcessor(chunk_size=150, chunk_overlap=30)
            payloads = processor.process_pdf(uploaded_file)
            
        if payloads:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            with st.spinner("🧠 Vectorizing chunks via Google GenAI & adding to HNSW Graph..."):
                for idx, payload in enumerate(payloads):
                    chunk_text = payload["text"]
                    meta = payload["metadata"]
                    vector_id = 20000 + idx
                    
                    # 1. Compute text embedding via official SDK
                    response = client.models.embed_content(
                        model="text-embedding-004",
                        contents=chunk_text
                    )
                    embedding_vector = response.embeddings[0].values
                    
                    # 2. Append to our local pure NumPy matrix database
                    st.session_state.db.add_vectors(
                        ids=[vector_id], 
                        vectors=[embedding_vector], 
                        metadatas=[meta]
                    )
                    
                    # 3. Route node through HNSW layers
                    st.session_state.hnsw.add_element(node_id=vector_id)
                    
                    # Keep track of textual context mapped to ID
                    st.session_state.processed_chunks[vector_id] = chunk_text
                    
                    # Update status indicator animations
                    prog_percentage = int(((idx + 1) / len(payloads)) * 100)
                    progress_bar.progress(prog_percentage)
                    status_text.text(f"Indexed chunk {idx+1}/{len(payloads)}")
                    
            st.success(f"🎉 Successfully indexed '{uploaded_file.name}'! (Max HNSW Level: {st.session_state.hnsw.max_level})")
            st.session_state.current_file = uploaded_file.name

# --- Chat / Query Engine Layer ---
st.write("---")
user_query = st.text_input("💬 Ask a question about your indexed document:")

if user_query:
    if not client:
        st.error("Missing valid API authentication parameters. Provide an API key first.")
    elif len(st.session_state.processed_chunks) == 0:
        st.warning("Please upload a document to build the search graph before asking questions.")
    else:
        with st.spinner("🔍 Routing query through HNSW layers..."):
            # 1. Embed user query string
            query_resp = client.models.embed_content(
                model="text-embedding-004",
                contents=user_query
            )
            query_vector = query_resp.embeddings[0].values
            
            # 2. Query our custom HNSW graph index for the top 3 contextual hits
            hnsw_hits = st.session_state.hnsw.query_hnsw(query_vector, k=3)
            
        if hnsw_hits:
            # Gather matching text chunks to form our context block
            retrieved_contexts = []
            for node_id, distance in hnsw_hits:
                text_chunk = st.session_state.processed_chunks.get(node_id, "")
                retrieved_contexts.append(f"[Context ID: {node_id}]: {text_chunk}")
                
            context_block = "\n\n".join(retrieved_contexts)
            
            # 3. Craft the grounded prompt payload for generation
            rag_prompt = (
                "You are an expert academic assistant. Answer the user's question using ONLY the provided "
                "document context below. If the answer cannot be found in the context, politely state that you do not know.\n\n"
                f"--- CONTEXT ---\n{context_block}\n\n"
                f"--- QUESTION ---\n{user_query}"
            )
            
            with st.spinner("🤖 Synthesizing grounded response via Gemini..."):
                # Call Gemini for a hallucination-free answer
                ai_response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=rag_prompt
                )
                
            # Display results split by architectural responsibility
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("🤖 Grounded AI Response")
                st.write(ai_response.text)
                
            with col2:
                st.subheader("🛠️ HNSW Retrieval Contexts")
                for node_id, distance in hnsw_hits:
                    with st.expander(f"Chunk ID: {node_id} (Dist: {distance:.4f})"):
                        st.write(st.session_state.processed_chunks[node_id])
        else:
            st.error("HNSW graph routing failed to locate candidate nodes.")