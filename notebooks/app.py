import os
import streamlit as st
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# -------------------------------------------------
# 1. CONFIGURATION & SESSION STATE
# -------------------------------------------------
st.set_page_config(page_title="Consumer Complaint Assistant", page_icon="🤖")
load_dotenv()

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# -------------------------------------------------
# 2. LOAD RESOURCES (CACHED)
# -------------------------------------------------
@st.cache_resource
def load_rag_system():
    # Load Environment Token
    HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    
    # Load Vector DB
    persist_dir = "./vector_store/complaints_db"
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    
    # Setup LLM
    llm_endpoint = HuggingFaceEndpoint(
        repo_id="HuggingFaceH4/zephyr-7b-beta",
        task="conversational",
        max_new_tokens=512,
        temperature=0.1,
        huggingfacehub_api_token=HF_TOKEN,
    )
    llm = ChatHuggingFace(llm=llm_endpoint)
    
    return vector_db, llm

try:
    vector_db, llm = load_rag_system()
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})
except Exception as e:
    st.error(f"Failed to load system: {e}")
    st.stop()

# -------------------------------------------------
# 3. HELPER FUNCTIONS
# -------------------------------------------------
def get_response(question):
    # 1. Retrieve relevant documents
    docs = retriever.invoke(question)
    context = "\n\n".join([d.page_content for d in docs])
    
    # 2. Build the prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Answer the question using ONLY the provided context. If the answer is not in the context, say 'Information not found'."),
        ("user", "Context:\n{context}\n\nQuestion:\n{question}")
    ])
    
    # 3. Run the chain
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({"context": context, "question": question})
    
    return response, docs

# -------------------------------------------------
# 4. USER INTERFACE
# -------------------------------------------------
st.title("🤖 Consumer Complaint RAG Chatbot")
st.markdown("Ask questions about unauthorized credit card charges and consumer disputes.")

# Sidebar with "Clear Chat" button
with st.sidebar:
    st.header("Settings")
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
    st.info("This system uses the Zephyr-7B model and a Chroma vector database.")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message:
            with st.expander("View Sources"):
                for i, doc in enumerate(message["sources"]):
                    st.caption(f"Source {i+1}:")
                    st.write(doc.page_content)

# Chat Input
if prompt_input := st.chat_input("Type your question here..."):
    # Add user message to state
    st.session_state.messages.append({"role": "user", "content": prompt_input})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt_input)

    # Generate Response
    with st.chat_message("assistant"):
        with st.spinner("Searching database and thinking..."):
            answer, source_docs = get_response(prompt_input)
            st.markdown(answer)
            
            # Display sources in an expander
            with st.expander("View Sources"):
                for i, doc in enumerate(source_docs):
                    st.caption(f"Source {i+1}:")
                    st.write(doc.page_content)
        
    # Add assistant response to state
    st.session_state.messages.append({
        "role": "assistant", 
        "content": answer, 
        "sources": source_docs
    })