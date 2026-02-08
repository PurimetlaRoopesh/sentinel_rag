import streamlit as st
import os
import ollama
from langchain_community.document_loaders import PyMuPDFLoader # UPGRADE: Better structure
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

# --- UI CONFIGURATION ---
st.set_page_config(page_title="Sentinel RAG: High Accuracy", layout="wide")
st.title("🛡️ Sentinel RAG: Secure Resume Chat")

# --- SIDEBAR: DYNAMIC UPLOAD ---
with st.sidebar:
    st.header("Upload Document")
    uploaded_file = st.file_uploader("Upload your resume", type="pdf")
    
    if uploaded_file:
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        if st.button("Re-Index Document"):
            with st.spinner("Deep-scanning resume..."):
                # UPGRADE: PyMuPDFLoader preserves layout better
                loader = PyMuPDFLoader("temp.pdf")
                pages = loader.load()
                
                # OPTIMIZATION: Ensure each project fits in a chunk
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=800, 
                    chunk_overlap=200
                )
                chunks = text_splitter.split_documents(pages)
                
                # UPGRADE: Adding task_type for Nomic precision
                embeddings = OllamaEmbeddings(
                    model="nomic-embed-text"
                )
                
                st.session_state.vector_db = Chroma.from_documents(
                    documents=chunks, 
                    embedding=embeddings,
                    persist_directory="./chroma_db"
                )
                st.success(f"Indexed {len(chunks)} chunks!")

# --- CHAT LOGIC ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about your projects..."):
    if "vector_db" not in st.session_state:
        st.error("Upload a PDF first!")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing all sections..."):
                # RETRIEVAL: Increase 'k' to ensure we see all projects
                docs = st.session_state.vector_db.similarity_search(prompt, k=6)
                context = "\n\n---\n\n".join([d.page_content for d in docs])
                
                full_prompt = f"""
                Analyze the context and list ALL projects mentioned. 
                If the document has a 'Projects' section, summarize every entry.
                
                CONTEXT:
                {context}

                QUESTION: {prompt}
                """
                
                response = ollama.generate(model="llama3.2", prompt=full_prompt)
                st.markdown(response['response'])
                st.session_state.messages.append({"role": "assistant", "content": response['response']})