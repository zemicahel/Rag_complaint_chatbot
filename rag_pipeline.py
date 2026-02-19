import os
import logging
import torch

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import pipeline

# -----------------------
# Ensure logs directory exists
# -----------------------
os.makedirs("logs", exist_ok=True)

# -----------------------
# Logging Configuration
# -----------------------
logging.basicConfig(
    filename="logs/app.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# -----------------------
# Device Detection
# -----------------------
device = 0 if torch.cuda.is_available() else -1
logging.info(f"Using device: {'cuda' if device == 0 else 'cpu'}")

# -----------------------
# Embedding Model
# -----------------------
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cuda" if device == 0 else "cpu"},
    encode_kwargs={"normalize_embeddings": False}
)

# -----------------------
# Load Vector Store
# -----------------------
VECTOR_DB_PATH = "vector_store/complaints_db"

if not os.path.exists(VECTOR_DB_PATH):
    raise FileNotFoundError(f"Vector DB not found at {VECTOR_DB_PATH}")

vector_store = Chroma(
    persist_directory=VECTOR_DB_PATH,
    embedding_function=embedding_model
)

retriever = vector_store.as_retriever(search_kwargs={"k": 3})

logging.info("Vector store loaded successfully.")

# -----------------------
# Load FLAN-T5 Model
# -----------------------
llm_model_name = "google/flan-t5-base"

generator = pipeline(
    "text2text-generation",
    model=llm_model_name,
    device=device
)

logging.info("FLAN-T5 model loaded successfully.")

# -----------------------
# RAG Function
# -----------------------
def generate_rag_response(query):
    try:
        logging.info(f"Received query: {query}")

        # Retrieve relevant documents
        raw_docs = retriever.invoke(query)

        if not raw_docs:
            logging.warning("No documents retrieved.")
            return {
                "answer": "Information not found in complaint database.",
                "sources": []
            }

        # -----------------------
        # Deduplicate overlapping chunks
        # -----------------------
        seen_chunks = set()
        docs = []

        for doc in raw_docs:
            chunk_id = doc.metadata.get("chunk_index")
            complaint_id = doc.metadata.get("complaint_id")

            unique_key = (complaint_id, chunk_id)

            if unique_key not in seen_chunks:
                docs.append(doc)
                seen_chunks.add(unique_key)

        # -----------------------
        # Combine Context
        # -----------------------
        context = "\n\n".join([doc.page_content for doc in docs])

        prompt = f"""
You are a financial complaint assistant.

Answer ONLY using the context below.
If the answer is not contained in the context, say "Information not found".

Context:
{context}

Question:
{query}

Answer:
"""

        # -----------------------
        # Generate Response
        # -----------------------
        result = generator(
            prompt,
            max_length=512,
            do_sample=False
        )

        answer = result[0]["generated_text"].strip()

        # -----------------------
        # Prepare Sources (Correct Metadata Keys)
        # -----------------------
        sources = []

        for doc in docs:
            metadata = doc.metadata

            sources.append({
                "complaint_id": metadata.get("complaint_id"),
                "product": metadata.get("product"),
                "company": metadata.get("company"),
                "state": metadata.get("state"),
                "date_received": metadata.get("date_received"),
                "chunk_index": metadata.get("chunk_index"),
                "word_count": metadata.get("word_count")
            })

        logging.info("Query processed successfully.")

        return {
            "answer": answer,
            "sources": sources
        }

    except Exception as e:
        logging.error(f"Error processing query: {str(e)}")
        return {
            "error": "An error occurred while processing the request."
        }
