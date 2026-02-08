import ollama
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings # Updated to the new version

def ask_my_pdf(question):
    # 1. Load the database we just made
    persist_dir = "./chroma_db"
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    db = Chroma(persist_directory=persist_dir, embedding_function=embeddings)

    # 2. Find the 2 most relevant chunks from your PDF
    docs = db.similarity_search(question, k=2)
    context = "\n".join([d.page_content for d in docs])

    # 3. Ask Llama 3.2 to answer using ONLY that context
    prompt = f"""
    You are a Secure Assistant. Answer the question using ONLY the context below. 
    If the answer isn't in the context, say you don't know.
    
    Context: {context}
    Question: {question}
    """

    print("\nThinking...")
    response = ollama.generate(model="llama3.2", prompt=prompt)
    return response['response']

if __name__ == "__main__":
    # Before running, make sure you have the new package: 
    # Type 'pip install langchain-ollama' in your terminal
    user_input = input("\nAsk a question about your PDF: ")
    answer = ask_my_pdf(user_input)
    print("\n--- AI ANSWER ---")
    print(answer)