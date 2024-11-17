import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import torch

# Configure device for optimal performance on M1
device = "mps" if torch.backends.mps.is_available() else "cpu"

def initialize_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': device},
        encode_kwargs={'device': device, 'batch_size': 32}
    )

def get_vectorstore_from_url(url):
    try:
        loader = WebBaseLoader(url)
        document = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len
        )
        document_chunks = text_splitter.split_documents(document)
        # Initialize embeddings silently
        embeddings = initialize_embeddings()
        
        vector_store = FAISS.from_documents(
            document_chunks, 
            embeddings
        )
        return vector_store
    except Exception:
        return None

@st.cache_resource
def get_llm():
    try:
        return Ollama(
            model="llama3:latest",
            temperature=0.7,
            num_ctx=4096,
            num_gpu=1,
            num_thread=4
        )
    except Exception:
        return None

def get_context_retriever_chain(vector_store):
    llm = get_llm()
    if not llm:
        return None
        
    retriever = vector_store.as_retriever(
        search_kwargs={"k": 3}
    )
    
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the article you added, can you tell me more about it?"),
    ])
    
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    return retriever_chain

def get_conversational_rag_chain(retriever_chain):
    llm = get_llm()
    if not llm or not retriever_chain:
        return None
        
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI assistant analyzing articles. Provide clear, concise answers based on the article content.\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])
    
    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)

def process_query(user_query, vector_store, chat_history):
    try:
        retriever_chain = get_context_retriever_chain(vector_store)
        if not retriever_chain:
            return "I'm having trouble processing your request at the moment. Please try again later."
            
        conversational_rag_chain = get_conversational_rag_chain(retriever_chain)
        if not conversational_rag_chain:
            return "I'm having trouble processing your request at the moment. Please try again later."
            
        response = conversational_rag_chain.invoke({
            "chat_history": chat_history,
            "input": user_query,
        })
        return response['answer']
    except Exception:
        return "I'm having trouble generating a response. Please try again later."

# App configuration
st.set_page_config(
    page_title="NewsK AI Advisor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply dark mode styling
st.markdown(
    """
    <style>
    .main {
        background-color: #1E1E1E;
        color: #FFFFFF;
    }
    .stSidebar {
        background-color: #121212;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("NewsK AI Advisor 🎯")

# App sidebar for user input
with st.sidebar:
    st.header("Settings")
    website_url = st.text_input("Article URL")

# Process and interaction handling
if website_url:
    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content="Hello! I'm your NewsK AI Advisor. How can I help you today? 🤖"),
        ]
    
    # Initialize or retrieve vector store
    if "vector_store" not in st.session_state:
        with st.spinner("Processing article... This may take a moment."):
            st.session_state.vector_store = get_vectorstore_from_url(website_url)
            if st.session_state.vector_store:
                st.success("Article processed successfully!")
            else:
                st.error("Failed to process article. Please check the URL and try again.")
    
    # Handle user queries
    if st.session_state.get("vector_store"):
        user_query = st.chat_input("Ask me anything about the article...")
        if user_query:
            with st.spinner("Thinking..."):
                response = process_query(
                    user_query, 
                    st.session_state.vector_store, 
                    st.session_state.chat_history
                )
            st.session_state.chat_history.append(HumanMessage(content=user_query))
            st.session_state.chat_history.append(AIMessage(content=response))
            
        # Display conversation history
        for message in st.session_state.chat_history:
            with st.chat_message("AI" if isinstance(message, AIMessage) else "Human"):
                st.write(message.content)
else:
    st.info("Please enter a valid article URL to begin.")
