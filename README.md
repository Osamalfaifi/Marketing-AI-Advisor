# Marketing AI Advisor 🎯

This Streamlit-based web application leverages advanced NLP and AI to provide insightful analysis and responses based on marketing-related articles. Using the power of LangChain and OpenAI's technologies, it creates an interactive AI assistant that can dissect and discuss the contents of any marketing article you provide.

## Features

- **Article Analysis**: Input the URL of a marketing article, and the app will analyze its content to understand the context and key points.
- **Interactive AI Conversation**: Engage in a dynamic conversation with the AI about the article's content, asking any questions to deepen your understanding or gain additional insights.
- **Contextual Understanding**: The AI retains context from the article and your questions, allowing for a coherent and contextually aware dialogue.
- **Vector Store Creation**: The app breaks down the article into manageable chunks and creates a vector store using `Chroma` for efficient retrieval and analysis.
- **Retrieval Chains**: Utilizes LangChain's retrieval chains to fetch relevant information and generate responses based on the article's context and user queries.

## How It Works

1. **Article Processing**: Upon entering the URL of a marketing article, the application uses `WebBaseLoader` to fetch and process the document.
2. **Document Splitting and Vectorization**: The document is split into smaller chunks using `RecursiveCharacterTextSplitter`, and each chunk is vectorized to create a searchable vector store.
3. **Interactive Dialogue**: Users can engage in a dialogue with the AI, powered by `ChatOpenAI`, to ask questions and receive contextually relevant answers based on the article's content.
4. **Session State Management**: The application maintains a conversation history and the vector store in the session state, allowing for continuity and depth in the conversation.

## Installation

To run this application locally, clone the repository and install the required dependencies:

- ```git clone Osamalfaifi/Marketing-AI-Advisor```
- ```cd your-repository-directory```
- ```pip install -r requirements.txt```
- ```streamlit run app.py```
