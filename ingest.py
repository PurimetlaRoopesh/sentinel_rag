import os
# These imports match the "pip install" we just did
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma 
from langchain_community.embeddings import OllamaEmbeddings

def process_document(file_path):
    print("Starting ingestion process...")
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        return

    loader = PyPDFLoader(file_path)
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(docs)

    print(f"Creating embeddings for {len(chunks)} chunks...")

    persist_dir = "./chroma_db"

    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=OllamaEmbeddings(model="nomic-embed-text"),
        persist_directory=persist_dir
    )
    
    print(f"Database created! Total items: {vector_db._collection.count()}")
    return vector_db

if __name__ == "__main__":
    process_document("test.pdf.pdf") # Add the extra .pdf here