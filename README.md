# PDF-Question-Answering-System

This project demonstrates a fully functional Retrieval Augmented Generation (RAG) pipeline built in Python. It allows users to upload PDF documents and then ask questions about their content. The system retrieves relevant text chunks from the PDFs and uses a Large Language Model (LLM) via the Groq API to generate answers.

The pipeline features semantic retrieval using sentence embeddings and a FAISS vector store, along with dynamic management of the PDF index to reflect changes in the source documents.

## Features

*   **PDF Ingestion:** Loads PDF documents from a local directory.
*   **Document Processing:** Splits PDFs into manageable, overlapping text chunks.
*   **Semantic Embedding:** Generates dense vector embeddings for text chunks using Hugging Face's `BAAI/bge-small-en-v1.5` model.
*   **Vector Storage & Search:** Utilizes FAISS (cpu) for efficient storage and similarity search of document embeddings.
*   **LLM Integration:** Leverages the Groq API for fast inference with LLMs (e.g., `llama3-8b-8192`) to generate answers based on retrieved context.
*   **Prompt Engineering:** Uses a custom prompt template to guide the LLM in synthesizing answers from provided context.
*   **Dynamic Index Management:**
    *   Automatically detects new, modified, or removed PDFs in the `pdfs/` directory.
    *   Updates the FAISS vector store incrementally for new files.
    *   Performs a full rebuild of the index if files are modified/removed or if the embedding model changes, ensuring data integrity.
    *   Tracks indexed files and their metadata (modification time, size) in `index_metadata.json`.
*   **Interactive CLI:** Allows users to ask questions, rebuild the index, and check the status of indexed documents.
*   **Source Citation:** Provides source document and page numbers for the context used in generating answers.

## Tech Stack

*   **Core Language:** Python 3.8+
*   **Orchestration:** Langchain
*   **LLM Provider:** Groq API (`langchain-groq`)
*   **Embedding Model:** Hugging Face `BAAI/bge-small-en-v1.5` (`langchain-huggingface`, `sentence-transformers`)
*   **Vector Store:** FAISS (`faiss-cpu`, `langchain-community`)
*   **PDF Loading:** `PyPDFLoader` (`pypdf`)
*   **Configuration:** `python-dotenv`
