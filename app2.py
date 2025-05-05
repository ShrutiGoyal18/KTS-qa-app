import streamlit as st
import os
import sys
import uuid
import re
import torch
import hashlib
import numpy as np
import faiss
import pickle
from tqdm import tqdm
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer

# Add this to the top of app2.py
import torch

# Force models to use CPU if needed (to avoid CUDA out of memory)
os.environ["CUDA_VISIBLE_DEVICES"] = ""
torch.set_num_threads(4)  # Limit CPU threads

# Import the main functionality from your main2.py
from main2 import (
    FAISSIndex, 
    doc_hash, 
    extract_text_from_docx, 
    load_doc_chunks, 
    embed_and_store,
    chat_with_bot,
    rerank
)

# Configure Streamlit page
st.set_page_config(
    page_title="Document QA Assistant (BGE)",
    page_icon="📄",
    layout="wide",
)

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []
    
if "index" not in st.session_state:
    st.session_state.index = None
    
if "document_processed" not in st.session_state:
    st.session_state.document_processed = False
    
if "document_path" not in st.session_state:
    st.session_state.document_path = None
    
# Store embedding model settings
if "embedding_model" not in st.session_state:
    st.session_state.embedding_model = "BAAI/bge-large-en-v1.5"
    
# Store reranker settings
if "use_reranker" not in st.session_state:
    st.session_state.use_reranker = True
    
if "reranker_top_k" not in st.session_state:
    st.session_state.reranker_top_k = 5
    
if "reranker_model" not in st.session_state:
    st.session_state.reranker_model = "BAAI/bge-reranker-base"
    
# Function to process the document
def process_document(uploaded_file):
    # Save the uploaded file to a temporary location
    temp_path = os.path.join("temp", uploaded_file.name)
    os.makedirs("temp", exist_ok=True)
    
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.session_state.document_path = temp_path
    
    # Generate document ID
    doc_id = doc_hash(temp_path)
    
    # Remove old index files if they exist
    index_path = os.path.join("faiss_indexes", doc_id)
    if os.path.exists(f"{index_path}.index"):
        os.remove(f"{index_path}.index")
    if os.path.exists(f"{index_path}.metadata"):
        os.remove(f"{index_path}.metadata")
    
    # Load and chunk the document
    with st.spinner("Processing document..."):
        chunks = load_doc_chunks(temp_path)
        
        if not chunks:
            st.error("Error: Failed to extract any text from the document.")
            return False
        
        # Create FAISS index and store embeddings
        progress_text = "Creating embeddings with BGE-large model (this may take a few minutes)..."
        progress_bar = st.progress(0)
        
        # Create a custom progress callback for tqdm
        def progress_callback(current, total):
            progress_bar.progress(current/total)
            
        # Patch tqdm to use our callback
        old_init = tqdm.__init__
        def new_init(*args, **kwargs):
            kwargs['disable'] = True
            old_init(*args, **kwargs)
        tqdm.__init__ = new_init
        
        index = embed_and_store(chunks, doc_id=doc_id, model_name=st.session_state.embedding_model)
        
        # Reset tqdm
        tqdm.__init__ = old_init
        progress_bar.progress(1.0)
        
        st.session_state.index = index
        st.session_state.document_processed = True
        
        return True

# Sidebar for document upload and processing
st.sidebar.title("Document Assistant")
st.sidebar.markdown("Upload a document to start asking questions.")

uploaded_file = st.sidebar.file_uploader("Choose a DOCX file", type=["docx"])

if uploaded_file and not st.session_state.document_processed:
    if st.sidebar.button("Process Document", key="process_btn"):
        if uploaded_file.name.endswith(".docx"):
            if process_document(uploaded_file):
                st.sidebar.success(f"Document processed: {uploaded_file.name}")
        else:
            st.sidebar.error("Please upload a DOCX file.")

# Reset functionality
if st.session_state.document_processed:
    if st.sidebar.button("Process a new document", key="reset_btn"):
        st.session_state.document_processed = False
        st.session_state.index = None
        st.session_state.messages = []
        st.rerun()  # Rerun to clear the UI

# Display vector search settings in sidebar if document is processed
if st.session_state.document_processed:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Advanced Settings")
    
    # Just show the current model being used with no dropdown
    st.sidebar.markdown(f"**Embedding Model:** BAAI/bge-large-en-v1.5 (1024d)")
    st.sidebar.info("Using BGE Large model for optimal search quality.")
    
    # Reranker settings
    st.sidebar.markdown("### Reranker Settings")
    st.session_state.use_reranker = st.sidebar.checkbox(
        "Use reranker for better results", 
        value=st.session_state.use_reranker
    )
    
    if st.session_state.use_reranker:
        st.session_state.reranker_top_k = st.sidebar.slider(
            "Number of reranked passages", 
            min_value=1, 
            max_value=10, 
            value=st.session_state.reranker_top_k
        )

# Add API settings with local fallback
st.sidebar.markdown("### API and Search Settings")
use_local = st.sidebar.checkbox("Use local search (no API calls)", value=False)

# Define the local answer function at an appropriate location
def local_answer(query, index, model_name='BAAI/bge-large-en-v1.5'):
    """Local answer generation without using API"""
    try:
        # Get vector embedding
        embed_model = SentenceTransformer(model_name)
        
        # Check if dimensions match
        dimension = embed_model.get_sentence_embedding_dimension()
        if index.index.d != dimension:
            st.warning(f"⚠️ Dimension mismatch: Index was built with dimension {index.index.d}, but current model uses {dimension}. Using hybrid search.")
            # Use keyword search instead
            return keyword_search(query, index)
        
        # For BGE models, add query prefix
        query_text = "Represent this sentence for searching relevant passages: " + query
        query_vec = embed_model.encode(query_text).tolist()
        
        # Get results
        res = index.query(vector=query_vec, top_k=5)
        chunks = [match['metadata']['text'] for match in res['matches']]
        
        # Return simple concatenated chunks
        if chunks:
            return "\n\n---\n\n".join(chunks)
        else:
            return "No relevant information found."
    except Exception as e:
        # Fallback to keyword search if vector search fails
        st.warning(f"Vector search failed: {str(e)}. Using keyword search instead.")
        return keyword_search(query, index)

def keyword_search(query, index):
    """Simple keyword-based search through metadata"""
    query_terms = set(query.lower().split())
    scored_chunks = []
    
    # Search through all metadata
    for id_str, metadata in index.metadata.items():
        if "text" in metadata:
            text = metadata["text"]
            text_lower = text.lower()
            # Count keyword matches
            score = sum(1 for term in query_terms if term in text_lower and len(term) > 3)
            if score > 0:
                scored_chunks.append((score, text))
    
    # Sort by score
    scored_chunks.sort(reverse=True)
    
    if scored_chunks:
        # Return top chunks
        return "\n\n---\n\n".join(chunk for _, chunk in scored_chunks[:3])
    else:
        return "No relevant information found in the document."

# Display custom instructions in the sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### Instructions")
st.sidebar.markdown("""
1. Upload a DOCX file
2. Click "Process Document" button
3. Ask questions about the document content
4. Type 'clear' to reset the chat history
""")

# Main area for chat interface
st.title("Document QA Assistant - BGE Enhanced")
st.markdown("Powered by the BGE Large Embeddings model for better semantic search")

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input
if prompt := st.chat_input("Ask a question about your document"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Check if we have a processed document
    if not st.session_state.document_processed:
        response = "Please upload and process a document first."
    elif prompt.lower() == "clear":
        st.session_state.messages = []
        response = "Chat history cleared."
    else:
        # Get a response from our QA system
        with st.spinner("Searching document..."):
            try:
                if use_local:
                    # Use local search without API calls
                    response = local_answer(
                        prompt, 
                        st.session_state.index, 
                        model_name=st.session_state.embedding_model
                    )
                else:
                    # Use the main chat_with_bot function
                    try:
                        # Original approach with reranker settings
                        if st.session_state.use_reranker:
                            response = chat_with_bot(
                                prompt, 
                                st.session_state.index,
                                embed_model_name=st.session_state.embedding_model,
                                reranker_top_k=st.session_state.reranker_top_k
                            )
                        else:
                            response = chat_with_bot(
                                prompt, 
                                st.session_state.index,
                                embed_model_name=st.session_state.embedding_model,
                                reranker_top_k=0
                            )
                    except AssertionError as ae:
                        # Specific handling for dimension mismatch
                        st.warning("Dimension mismatch detected. Falling back to local search.")
                        response = local_answer(
                            prompt, 
                            st.session_state.index, 
                            model_name=st.session_state.embedding_model
                        )
            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                st.error(f"**Error details:**\n\n```\n{str(e)}\n\n{tb}\n```")
                response = "An error occurred while processing your question. Please see the error details above."
    
    # Display assistant response
    with st.chat_message("assistant"):
        st.markdown(response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

# Add document info display
if st.session_state.document_processed and st.session_state.document_path:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Document Information")
    doc_name = os.path.basename(st.session_state.document_path)
    st.sidebar.markdown(f"**File:** {doc_name}")

# Add footer
st.sidebar.markdown("---")
st.sidebar.markdown("Built with Streamlit & FAISS")