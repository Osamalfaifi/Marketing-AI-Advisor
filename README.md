# NewsK AI Advisor 🎯

This Streamlit-based web application leverages advanced NLP, LLMs, and AI-driven contextual retrieval to analyze and provide insightful responses based on articles, enhancing user engagement and content exploration. Using the LangChain framework, Hugging Face embeddings, FAISS vector stores, and the Ollama LLM, it enables a conversational experience centered around article content, perfect for personalized recommendations and in-depth analysis.

## Features

- **Article Analysis**: Input the URL of an article, and the app will fetch, process, and analyze its content to create context-aware responses.
- **Interactive AI Conversation**: Engage in dynamic conversations with an AI advisor about the article's content, asking questions to gain further insights and understanding.
- **Contextual Memory**: The AI retains the context of your queries, allowing for a coherent dialogue that builds on previous interactions.
- **Vector Store Creation with FAISS**: Efficiently stores and retrieves text embeddings using FAISS, enabling semantic search and rapid responses.
- **Retrieval Chains**: Utilizes LangChain's retrieval chains to fetch relevant information and generate insightful, context-aware answers.
- **Ollama LLM Integration**: Leverages the capabilities of the Ollama LLM to generate high-quality, contextually aware responses, tailored to user queries.
![Solution](docs/NewsK%20AI%20Advisor%20-%20Solution.png)


## How It Works

1. **Article Retrieval**:
   - The app uses `WebBaseLoader` from LangChain to scrape data from the provided article URL, extracting and preparing it for analysis.

2. **Text Splitting**:
   - The content is segmented into smaller chunks using the `RecursiveCharacterTextSplitter` for efficient processing and retrieval.

3. **Embeddings and Vectorization**:
   - Text chunks are converted into embeddings using the Hugging Face model `sentence-transformers/all-MiniLM-L6-v2`. This step converts text into numerical representations that capture semantic meaning.

4. **Vector Store Creation**:
   - The embeddings are stored in a FAISS vector store, optimized for high-speed similarity searches during user queries.

5. **Interactive Conversations**:
   - User inputs are vectorized and compared with stored embeddings to find relevant information. The LangChain retrieval chains, coupled with context-aware prompts and the Ollama LLM, generate responses.

6. **Conversational History**:
   - The AI maintains conversational history to provide consistent, contextually relevant answers and engage in meaningful dialogue.

7. **Ollama LLM Configuration**:
   - The application integrates the Ollama LLM, configured with customizable parameters like temperature, context length, and multi-threading, to optimize response generation.

8. **Customization Options**:
   - Offers adjustable parameters like model temperature and context length to tailor the interaction experience.

![Solution Architecture](docs/NewsK%20AI%20Advisor%20-%20Architecture.png)
[Solution Website](https://newsK-ai-advisor.streamlit.app/)

## Application Configuration

- **Page Setup**: The app has a sleek UI with dark mode styling, providing an optimal user experience.
- **Device Optimization**: It utilizes `torch` with M1 chip optimization, ensuring high performance on compatible Apple devices.

## Installation and Setup

### Prerequisites

- Python 3.7 or higher
- Streamlit
- Required libraries specified in `requirements.txt`

### Installation Steps

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Osamalfaifi/NewsK-AI-Advisor
   cd NewsK-AI-Advisor