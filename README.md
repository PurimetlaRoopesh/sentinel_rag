# 🛡️ Sentinel-RAG: Local Intelligence & Secure Document Retrieval

Sentinel-RAG is a high-performance, **privacy-first** Retrieval-Augmented Generation (RAG) system. It is designed to bridge the gap between Large Language Models and private data, ensuring that sensitive documents (Resumes, Financials, Legal Papers) are processed entirely on local hardware without cloud exposure.



## 🎯 Project Impact
In a landscape where data privacy is paramount, Sentinel-RAG demonstrates a secure alternative to centralized AI. By utilizing a local stack, this project eliminates the risk of data leakage while providing precise, document-grounded answers.

## 🏗️ Technical Architecture
* **LLM Engine:** Llama 3.2 (3B Parameters) via Ollama.
* **Vector Embeddings:** Nomic-Embed-Text (Matryoshka-capable embeddings for high-dimensional semantic search).
* **Vector Database:** ChromaDB (Self-hosted persistent storage).
* **Orchestration:** LangChain (Managing the Retrieval-Augmentation loop).
* **Frontend:** Streamlit (Reactive Web Interface).

## 🧩 Engineering Challenges & Solutions

### 1. The Columnar Layout Hurdle
**Problem:** Standard PDF parsers read multi-column resumes linearly, scrambling the logical flow of project descriptions.
**Solution:** Migrated to `PyMuPDF` (Fitz) to implement **layout-aware extraction**, preserving the structural integrity of complex document formats.

### 2. Eliminating LLM Hallucinations
**Problem:** LLMs tend to "autocomplete" missing information using training data rather than the provided document.
**Solution:** Engineered a **Strict-Context Prompt**. By wrapping the retrieval output in a restrictive system instruction, the model is forced to verify answers against the provided context or admit a lack of information.

### 3. Optimization for Local Hardware
**Problem:** High latency during the retrieval-generation phase on standard CPUs.
**Solution:** Optimized chunking strategy (**600 tokens with 150-token overlap**) and implemented **Streamed Response Windows** to provide real-time user feedback during inference.

## 🛠️ Installation & Setup

1. **Clone the Repository:**
   ```bash
   git clone [https://github.com/PurimetlaRoopesh/sentinel_rag](https://github.comPurimetlaRoopesh/sentinel_rag.git)
   cd sentinel-rag
   
## Set up Environment
'''bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt

## Initialize Local Models
'''bash
ollama pull llama3.2
ollama pull nomic-embed-text

## Launch Application
'''bash
streamlit run app.py
