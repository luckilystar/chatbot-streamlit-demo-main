import os
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. Load environment variables
load_dotenv()
if "OPENAI_API_KEY" not in os.environ:
    st.error("Please set your OPENAI_API_KEY in a .env file.")
    st.stop()

# 2. Document Loading and Processing
@st.cache_resource
def load_and_process_documents(folder_path="documents"):
    """
    Loads PDF and DOCX files from the specified folder, splits them into chunks,
    and creates a vector store for RAG.
    """
    st.info("Loading documents...")
    
    # Load all PDF files using DirectoryLoader
    pdf_loader = DirectoryLoader(
        folder_path,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader
    )
    # Load all DOCX files using DirectoryLoader and UnstructuredWordDocumentLoader
    docx_loader = DirectoryLoader(
        folder_path,
        glob="**/*.docx",
        loader_cls=UnstructuredWordDocumentLoader
    )
    
    # Load and combine all documents
    docs = []
    docs.extend(pdf_loader.load())
    docs.extend(docx_loader.load())

    if not docs:
        st.error("No PDF or DOCX files found in the 'documents' folder.")
        st.stop()

    # Split documents into chunks
    st.info(f"Processing {len(docs)} documents.")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    
    # Create and persist the vector store
    embedding_model = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(documents=splits, embedding=embedding_model, persist_directory="./chroma_db")
    st.success(f"Processed {len(splits)} document chunks and created vector store.")
    return vectorstore

# 3. RAG Chain Setup
@st.cache_resource
def setup_rag_chain(vectorstore):
    """Sets up the RAG chain with a retriever, prompt, and LLM."""
    retriever = vectorstore.as_retriever()
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    
    template = """
    You are an assistant for question-answering tasks. Use the following pieces of retrieved context
    to answer the question. If you don't know the answer, just say that you don't know.
    
    Question: {question}
    Context: {context}
    Answer:
    """
    prompt = ChatPromptTemplate.from_template(template)
    
    # Create the RAG chain using LangChain Expression Language (LCEL)
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain

# 4. Streamlit UI and Logic
def main():
    st.set_page_config(page_title="RAG Chat with Documents", layout="wide")
    st.title("📄 Chat with Your Documents (PDF & DOCX)")
    st.markdown("Ask questions about the files in your `documents` folder.")

    # Load and process documents, and set up the RAG chain
    vectorstore = load_and_process_documents()
    rag_chain = setup_rag_chain(vectorstore)

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("Ask a question about your documents:"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            response_stream = rag_chain.stream(prompt)
            # Write the streaming response to the UI
            full_response = st.write_stream(response_stream)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})

if __name__ == "__main__":
    main()

