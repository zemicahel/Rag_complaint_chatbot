
````markdown
# 🤖 Consumer Complaint RAG Chatbot

A **Retrieval-Augmented Generation (RAG)** system that enables accurate, source-grounded question answering over real-world consumer complaint data.  
Built using **LangChain, Hugging Face, Chroma, and Streamlit**.

---

## 📌 Project Overview

Financial institutions and regulators receive millions of consumer complaints each year, most of them written as long, unstructured narratives. Searching and extracting insights from this data using traditional methods is slow and error-prone.

This project solves that problem by building a **RAG-based AI assistant** that:
- Retrieves relevant complaint narratives using semantic search
- Generates answers strictly grounded in retrieved evidence
- Avoids hallucinations by design
- Provides source transparency to the user

---

## 🏗️ System Architecture

**High-level flow:**

1. Consumer complaint narratives are cleaned and preprocessed  
2. Text is chunked and embedded using a sentence transformer  
3. Embeddings are stored in a Chroma vector database  
4. User queries retrieve relevant chunks via semantic similarity  
5. A Large Language Model (LLM) generates answers using only retrieved context  
6. Results are displayed in an interactive Streamlit chat UI  

---

## 🧠 Technologies Used

### Core Stack
- **Python 3.10+**
- **LangChain**
- **Hugging Face Transformers & Endpoints**
- **Chroma Vector Database**
- **Streamlit**

### Models
- **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2`
- **LLM:** `HuggingFaceH4/zephyr-7b-beta`

### Data
- **CFPB Consumer Complaint Database**
- Focused on products such as:
  - Credit cards
  - Personal & installment loans
  - Checking & savings accounts
  - Money transfer services

---

## 📂 Repository Structure

```text
.
├── app.py                     # Streamlit chat application
├── notebooks/
│   ├── Task_01_Data_Cleaning.ipynb
│   ├── Task_02_Sampling.ipynb
│   ├── Task_03_RAG_core.ipynb
│   └──app.py        
├── data/
│   ├── raw/                    # Original CFPB dataset
│   └── processed/              # Cleaned & filtered data
├── vector_store/
│   └── complaints_db/          # Persisted Chroma vector database
├── .env                        # Environment variables (not committed)
├── .gitignore
├── requirements.txt
└── README.md
````

---

## ⚙️ Setup Instructions

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/zemicahel/Rag_complaint_chatbot.git
cd consumer-complaint-rag
```

### 2️⃣ Create a Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\\Scripts\\activate  # Windows
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Configure Environment Variables

Create a `.env` file in the project root:

```env
HUGGINGFACEHUB_API_TOKEN=your_huggingface_api_token
```

---

## 📊 Data Processing & Indexing

Run the notebooks in order:

1. **Task 01:** Data cleaning and preprocessing
2. **Task 02:** Stratified sampling (15,000 complaints)
3. **Task 03:** Chunking, embedding, and vector database creation

This will generate the persisted Chroma vector store in:

```text
vector_store/complaints_db
```

---

## 🚀 Running the Application

Once the vector store is created:

```bash
streamlit run app.py
```

Open your browser at:

```
http://localhost:8501
```

---

## 💬 Example Questions

* *What are common complaints about unauthorized credit card charges?*
* *How do consumers report fraudulent card activity?*
* *What issues are reported with money transfer services?*

If relevant information is not found, the system responds with:

> **“Information not found”**

---

## 🧪 Evaluation Summary

The system was tested with real-world consumer dispute questions and demonstrated:

* High factual accuracy
* Strong grounding in retrieved sources
* Correct refusal behavior for out-of-scope questions
* No observed hallucinations

---

## 🔐 Design Principles

* **Grounded Generation:** Answers are restricted to retrieved context
* **Transparency:** Users can view source complaint excerpts
* **Safety First:** Conservative responses when information is incomplete
* **Modularity:** Easy to swap models or vector databases

---

## 🛠️ Future Improvements

* Re-ranking retrieved chunks for higher precision
* Sentence-level citation highlighting
* Multi-product filtering in the UI
* GPU-accelerated embeddings for scale
* Deployment on cloud infrastructure

---

## 📜 License

This project is for educational and research purposes.
Please review the CFPB data usage policy before commercial use.

---

## 🙌 Acknowledgements

* Consumer Financial Protection Bureau (CFPB)
* Hugging Face
* LangChain Community

---

**Author:** Zemicahel Abraham
**Project Type:** Retrieval-Augmented Generation (RAG) System

