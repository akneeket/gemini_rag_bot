import os
import tempfile
import pickle
import time
import json
import asyncio
import streamlit as st
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader

import google.generativeai as genai

# ========== CONFIG ==========
GEMINI_API_KEY = "place your api key here"  # 🔐 Replace this
MODEL_NAME = "models/gemini-1.5-flash"
PICKLE_PATH = "faiss_vectorstore.pkl"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ========== SETUP ==========
genai.configure(api_key=GEMINI_API_KEY)
embedding = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

# ========== STREAMLIT PAGE ==========
st.set_page_config(page_title="📚 Gemini RAG Bot", layout="centered")
st.title("🤖 Gemini-Powered RAG Chatbot")
st.markdown("Upload files (PDF/CSV/JSON) or provide a webpage to build your own **knowledge base**.")

# ========== SIDEBAR ==========
with st.sidebar:
    st.header("📁 Upload Your Knowledge Source")
    uploaded_file = st.file_uploader("Upload PDF, CSV, or JSON", type=["pdf", "csv", "json"])
    url = st.text_input("🌐 Or enter a webpage URL")

# ========== USER INPUT ==========
with st.form("question_form"):
    question = st.text_input("💬 Ask a question:", placeholder="e.g., Summarize the content")
    submit_clicked = st.form_submit_button("🔍 Ask")

# ========== HELPER FUNCTIONS ==========
def ask_gemini(context, question):
    prompt = f"""You are a helpful assistant. Use ONLY the provided context to answer the question.

Context:
{context}

Question: {question}
Answer:"""
    model = genai.GenerativeModel(MODEL_NAME)

    async def call_model():
        return await model.generate_content_async(prompt)

    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    try:
        response = loop.run_until_complete(call_model())
        return response.text.strip()
    except Exception as e:
        return f"❌ Error while calling Gemini: {str(e)}"

def load_json_as_docs(file) -> list:
    content = json.load(file)
    docs = []
    if isinstance(content, list):
        for item in content:
            docs.append(Document(page_content=json.dumps(item, indent=2)))
    else:
        docs.append(Document(page_content=json.dumps(content, indent=2)))
    return docs

def load_csv_as_docs(file) -> list:
    df = pd.read_csv(file)
    return [Document(page_content=df.to_string(index=False))]

# ========== INIT STATE ==========
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "docs" not in st.session_state:
    st.session_state.docs = None

# ========== LOAD FROM PICKLE ==========
if os.path.exists(PICKLE_PATH):
    with open(PICKLE_PATH, "rb") as f:
        st.session_state.vectorstore = pickle.load(f)
    st.toast("🔁 Loaded vectorstore from previous session")

# ========== PROCESS FILE OR URL ==========
if uploaded_file or url:
    with st.spinner("⏳ Processing document..."):
        documents = []
        if uploaded_file:
            suffix = os.path.splitext(uploaded_file.name)[1].lower()
            if suffix == ".pdf":
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.read())
                    loader = PyPDFLoader(tmp.name)
                    documents = loader.load()
            elif suffix == ".json":
                documents = load_json_as_docs(uploaded_file)
            elif suffix == ".csv":
                documents = load_csv_as_docs(uploaded_file)
        elif url:
            loader = WebBaseLoader(url)
            documents = loader.load()

        if documents:
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            docs = splitter.split_documents(documents)
            st.info("🔎 Embedding and indexing your data...")
            vectorstore = FAISS.from_documents(docs, embedding)

            with open(PICKLE_PATH, "wb") as f:
                pickle.dump(vectorstore, f)

            st.session_state.vectorstore = vectorstore
            st.session_state.docs = docs
            st.success("✅ Vectorstore created and saved!")

# ========== HANDLE QUESTIONS ==========
if submit_clicked and question:
    if st.session_state.vectorstore:
        with st.spinner("🧠 Thinking..."):
            retriever = st.session_state.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
            relevant_docs = retriever.get_relevant_documents(question)
            context = "\n\n".join([doc.page_content for doc in relevant_docs])
            answer = ask_gemini(context, question)
        st.success("🎯 Answer:")
        st.markdown(answer)
    else:
        st.warning("⚠️ Please upload a document or enter a URL first.")

# ========== FOOTER ==========
st.markdown("---")
st.caption("Built with 💙 using LangChain, Gemini, FAISS, and Streamlit")
