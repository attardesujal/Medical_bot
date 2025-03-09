# Medical_bot
Here's a structured README description for your GitHub repository:  

---

# Medico - AI Medical Assistance Chatbot  

## Overview  
Medico is an AI-powered medical assistance chatbot designed to provide health suggestions and medicine recommendations. It leverages state-of-the-art NLP models to process user queries and generate relevant medical responses. The chatbot is built using Hugging Face models and integrates FAISS for efficient document retrieval.  

## FAISS Vector Database Creation  
To enable fast and accurate retrieval of medical information, the project processes medical PDFs and converts them into vector embeddings. The key steps include:  
1. **PDF Data Extraction:** The system loads medical documents from a specified directory.  
2. **Text Chunking:** The extracted text is divided into smaller, overlapping chunks for better embedding representation.  
3. **Embedding Generation:** Sentence-transformers from Hugging Face are used to create dense vector embeddings.  
4. **FAISS Indexing:** The embeddings are stored in a FAISS vector database for efficient similarity-based retrieval.  

To access the FAISS and pickle files required for chatbot functionality, visit:  
🔗 [FAISS & Pickle Files](https://huggingface.co/sujalattarde/medical_chatbot/tree/main)  

## Chatbot Implementation  
The chatbot utilizes a retrieval-augmented generation (RAG) pipeline:  
- **LLM Integration:** The chatbot is powered by Mistral-7B-Instruct, hosted on Hugging Face.  
- **Retrieval Mechanism:** FAISS is used to fetch relevant text chunks as context for question answering.  
- **Custom Prompting:** A structured prompt ensures that the chatbot provides precise, context-aware responses.  

## GUI Interface  
A user-friendly graphical interface (GUI) is implemented using Tkinter. The interface allows users to input queries, receive AI-generated responses, and clear the chat history.  

## Installation & Usage  
1. Clone the repository and install dependencies.  
2. Download the FAISS vector database from the provided link.  
3. Set up your Hugging Face API token in an `.env` file.  
4. Run the chatbot application to start interacting with Medico.  

## Future Enhancements  
- Expand the medical document database for broader knowledge coverage.  
- Improve model fine-tuning for better response accuracy.  
- Deploy the chatbot as a web application for wider accessibility.  
