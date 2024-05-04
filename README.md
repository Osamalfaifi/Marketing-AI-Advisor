# Marketing AI Advisor 🎯

This Streamlit-based web application leverages advanced NLP and AI to provide insightful analysis and responses based on marketing-related articles. Using the power of LangChain and OpenAI's technologies, it creates an interactive AI assistant that can dissect and discuss the contents of any marketing article you provide.

## Features

- **Article Analysis**: Input the URL of a marketing article, and the app will analyze its content to understand the context and key points.
- **Interactive AI Conversation**: Engage in a dynamic conversation with the AI about the article's content, asking any questions to deepen your understanding or gain additional insights.
- **Contextual Understanding**: The AI retains context from the article and your questions, allowing for a coherent and contextually aware dialogue.
- **Vector Store Creation**: The app breaks down the article into manageable chunks and creates a vector store using `Chroma` for efficient retrieval and analysis.
- **Retrieval Chains**: Utilizes LangChain's retrieval chains to fetch relevant information and generate responses based on the article's context and user queries.

## How It Works

# Article Selection and Analysis Workflow

## Article Selection
The process begins with scraping article data from a website. We utilize WebBaseLoader, a tool from Langchain, to extract data from HTML and XML files. It parses the raw HTML content of the website into a more manageable form.

## Text-Splitting
After scraping, the text data is divided into smaller chunks or documents using the Recursive Character Text Splitter. This segmentation is crucial for managing the data more efficiently and is a critical step in Natural Language Processing (NLP).

## Vectorization
Each chunk of text is transformed into a numerical format known as embeddings. These vector representations capture the semantic meaning of the text, enabling further processing.

## Vector Database
The generated embeddings are stored in a vector database. This database is specifically optimized for high-speed vector searches, crucial for the semantic search phase.

## Question Embedding
In the interface, a question like "What is this article mostly about?" is embedded using a similar vectorization process as the text data. This converts the question into a vector, allowing it to be compared with other vectors in the database.

## Semantic Search
The embedded question is then used to perform a semantic search within the vector database. This search identifies the vectors (text chunks) that are most similar or relevant to the question vector.

## Retrieval of Information
### Ranked Results
The results from the semantic search are ranked based on their relevance to the query. This ranking is pivotal in determining which chunks of text are most likely to contain the answer to the user's question.

## Answer Generation
A Large Language Model uses the ranked results to generate a coherent and contextually appropriate answer.

## Conclusion
The final output is the answer to the query about the website, formulated based on the most relevant text chunks retrieved and processed by the language model.


![Solution Architecture](docs/Marketing Project Report.png)

## Installation
To run this application locally, clone the repository and install the required dependencies:
- ```get your openAI key: https://help.openai.com/en/articles/4936850-where-do-i-find-my-openai-api-key```
- ```git clone Osamalfaifi/Marketing-AI-Advisor```
- ```cd your-repository-directory```
- ```pip install -r requirements.txt```
- ```streamlit run app.py```