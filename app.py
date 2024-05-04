import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

def get_vectorstore_from_url(url, api_key):
    # Load web document using the specified URL
    loader = WebBaseLoader(url)
    document = loader.load()
    
    # Split the document into chunks using Recursive Character Text Splitter
    text_splitter = RecursiveCharacterTextSplitter()
    document_chunks = text_splitter.split_documents(document)
    
    # Create vector store for each chunk using Chroma and OpenAI embeddings
    vector_store = Chroma.from_documents(document_chunks, OpenAIEmbeddings(api_key=api_key))
    return vector_store

def get_context_retriever_chain(vector_store):
    # Set up the LLM (Large Language Model) and the retriever with the given vector store
    llm = ChatOpenAI()
    retriever = vector_store.as_retriever()
    # Define prompt for the chat interaction
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the article you added, can you tell me more about it?"),
    ])
    # Create and return a history-aware retriever chain
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    return retriever_chain

def get_conversational_rag_chain(retriever_chain):
    # Reuse LLM and define a new prompt for document-based questions
    llm = ChatOpenAI()
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the following questions based on the article added:\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])
    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)
    
    # Create and return the full retrieval chain integrating document retrieval
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)

def get_response(user_query):
    # Retrieve response using the established conversational chain
    retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
    conversational_rag_chain = get_conversational_rag_chain(retriever_chain)
    response = conversational_rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_query,
    })
    return response['answer']

# App configuration
st.set_page_config(page_title="Marketing AI Advisor", page_icon="📈")
st.title("Marketing AI Advisor 🎯")

# App sidebar for user input
with st.sidebar:
    st.header("Settings")
    website_url = st.text_input("Article URL")
    api_key = st.text_input("Enter your OpenAI API key", type='password')

# Process and interaction handling
if website_url and api_key:
    # Initialize chat history if not already done
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content="Hello! I'm your Marketing AI Advisor. How can I help you today? 🤖"),
        ]
    # Retrieve or create vector store for the URL provided
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = get_vectorstore_from_url(website_url, api_key)
        
    # Handle user queries about the article
    user_query = st.chat_input("Ask me anything about the article that you added...")
    if user_query:
        response = get_response(user_query)
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=response))
        
    # Display the conversation history
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)
else:
    # Prompt user to input required settings if missing
    st.info("Please enter both a valid article URL and an OpenAI API key.")