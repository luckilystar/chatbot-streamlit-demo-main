# Import the necessary libraries
import streamlit as st  # For creating the web app interface
import os
from langchain_google_genai import ChatGoogleGenerativeAI  # For interacting with Google Gemini via LangChain
from langgraph.prebuilt import create_react_agent  # For creating a ReAct agent
from langchain_core.messages import HumanMessage, AIMessage  # For message formatting
from langchain_core.tools import tool  # For creating tools
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

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


# --- 1. Page Configuration and Title ---
# Set the title and a caption for the web page
st.title("💬 Document Assistant")
st.caption("A chatbot that can answer questions about document data using company policy")

# --- 2. Sidebar for Settings ---

# Create a sidebar section for app settings using 'with st.sidebar:'
with st.sidebar:
    # Add a subheader to organize the settings
    st.subheader("Settings")
    
    # Create a text input field for the Google AI API Key.
    # 'type="password"' hides the key as the user types it.
    google_api_key = st.text_input("Google AI API Key", type="password")

# --- 3. API Key and Agent Initialization ---

# Check if the user has provided an API key.
# If not, display an informational message and stop the app from running further.
if not google_api_key:
    st.info("Please add your Google AI API key in the sidebar to start chatting.", icon="🗝️")
    st.stop()

# This block of code handles the creation of the LangGraph agent.
# It's designed to be efficient: it only creates a new agent if one doesn't exist
# or if the user has changed the API key in the sidebar.
if ("agent" not in st.session_state) or (getattr(st.session_state, "_last_key", None) != google_api_key):
    # Load and process documents
    with st.spinner("Loading and processing documents..."):
        docs = load_documents(documents_folder)
    if docs:
        chunks = split_documents(docs)
        vector_db = create_vector_db(chunks)
        retriever = vector_db.as_retriever()
        memory = ConversationBufferMemory(memory_key="messages", return_messages=True)
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=google_api_key,
            temperature=0.2
        )
        qa_chain = ConversationalRetrievalChain.from_llm(llm, retriever, memory=memory)
    else:
        st.warning("No documents found in the 'documents' folder.")
        qa_chain = None


    try:
        # Create a ReAct agent for document QA using RAG
        st.session_state.agent = create_react_agent(
            model=llm,
            tools=[],
            prompt="""You are a helpful assistant that can answer questions about the uploaded documents using company policy and document content.\n\nIMPORTANT: When a user asks a question, use only the information found in the documents.\nIf you don't find an answer, do not make up an answer. Just say 'I don't know.'\nFormat your answer to be readable and clear.\nIf you encounter a question that cannot be answered with the documents or doesn't relate to the documents, politely inform the user that you can only answer questions related to the uploaded documents."""
        )
        st.session_state._last_key = google_api_key
        st.session_state.pop("messages", None)
    except Exception as e:
        st.error(f"Invalid API Key or configuration error: {e}")
        st.stop()

# --- 4. Chat History Management ---

# Initialize the message history (as a list) if it doesn't exist.
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 5. Display Past Messages ---

# Loop through every message currently stored in the session state.
for msg in st.session_state.messages:
    # For each message, create a chat message bubble with the appropriate role ("user" or "assistant").
    with st.chat_message(msg["role"]):
        # Display the content of the message using Markdown for nice formatting.
        st.markdown(msg["content"])

# --- 6. Handle User Input and Agent Communication ---

# Create a chat input box at the bottom of the page.
# The user's typed message will be stored in the 'prompt' variable.
prompt = st.chat_input("Ask a question about the document data...")

# Check if the user has entered a message.
if prompt:
    # 1. Add the user's message to our message history list.
    st.session_state.messages.append({"role": "user", "content": prompt})
    # 2. Display the user's message on the screen immediately for a responsive feel.
    with st.chat_message("user"):
        st.markdown(prompt)

    # 3. Get the assistant's response.
    # Use a 'try...except' block to gracefully handle potential errors (e.g., network issues, API errors).
    try:
        # Convert the message history to the format expected by the agent
        messages = []
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))
        
        # Show a spinner while waiting for the response
        with st.spinner("Thinking..."):
            # Send the user's prompt to the agent
            response = st.session_state.agent.invoke({"messages": messages})
            
            # Extract the answer from the response
            if "messages" in response and len(response["messages"]) > 0:
                answer = response["messages"][-1].content
            else:
                answer = "I'm sorry, I couldn't generate a response."

    except Exception as e:
        # If any error occurs, create an error message to display to the user.
        answer = f"An error occurred: {e}"

    # 4. Display the assistant's response.
    with st.chat_message("assistant"):
        st.markdown(answer)
    
    # 5. Add the assistant's response to the message history list.
    st.session_state.messages.append({"role": "assistant", "content": answer})