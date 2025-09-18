import streamlit as st
import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI

# Path to documents folder
documents_folder = "documents"

# Load documents
def load_documents(folder_path):
    docs = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            docs.extend(loader.load())
        elif filename.endswith(".docx"):
            loader = Docx2txtLoader(file_path)
            docs.extend(loader.load())
    return docs

# Split documents into chunks
def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(docs)

# Embed and store in vector DB
def create_vector_db(chunks):
    # Explicitly set model_name to avoid deprecation warning
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_documents(chunks, embeddings)
    return db


# Streamlit UI
st.set_page_config(page_title="Gemini RAG Chatbot", layout="wide")
st.title("Gemini RAG Chatbot with Document QA")
st.write("Upload your PDF or DOCX files to the 'documents' folder and ask questions based on their content.")

# Sidebar for Gemini API Key
with st.sidebar:
    st.subheader("Settings")
    gemini_api_key = st.text_input("Google Gemini API Key", type="password")
    reset_button = st.button("Reset Conversation", help="Clear all messages and start fresh")

# API key check
if not gemini_api_key:
    st.info("Please add your Google Gemini API key in the sidebar to start chatting.", icon="🗝️")
    st.stop()

# Streamlit UI
st.set_page_config(page_title="Gemini RAG Chatbot", layout="wide")
st.title("Gemini RAG Chatbot with Document QA")
st.write("Upload your PDF or DOCX files to the 'documents' folder and ask questions based on their content.")

# Load and process documents
with st.spinner("Loading and processing documents..."):
    docs = load_documents(documents_folder)
    if docs:
        chunks = split_documents(docs)
        vector_db = create_vector_db(chunks)
        retriever = vector_db.as_retriever()
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=gemini_api_key,
            temperature=0.2
        )
        qa_chain = ConversationalRetrievalChain.from_llm(llm, retriever, memory=memory)
    else:
        st.warning("No documents found in the 'documents' folder.")
        qa_chain = None

# Chat interface
if qa_chain:
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    user_input = st.text_input("Ask a question about your documents:")
    if user_input:
        with st.spinner("Gemini is thinking..."):
            result = qa_chain({"question": user_input, "chat_history": st.session_state["chat_history"]})
            st.session_state["chat_history"].append((user_input, result["answer"]))
            st.markdown(f"**You:** {user_input}")
            st.markdown(f"**Gemini:** {result['answer']}")
    if st.session_state["chat_history"]:
        st.markdown("---")
        st.markdown("### Chat History")
        for q, a in st.session_state["chat_history"]:
            st.markdown(f"**You:** {q}")
            st.markdown(f"**Gemini:** {a}")
