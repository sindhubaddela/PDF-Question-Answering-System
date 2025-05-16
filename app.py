import os
import shutil
import json
import time 
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA


load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Model and Embedding Configuration
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
LLM_MODEL_NAME = "llama3-8b-8192"

PDFS_PATH = "./pdfs"
VECTORSTORE_BASE_PATH = "faiss_index" # Directory to hold FAISS index and metadata
METADATA_FILE = os.path.join(VECTORSTORE_BASE_PATH, "index_metadata.json")



def get_embedding_model(model_name=EMBEDDING_MODEL_NAME):
    print(f"Loading embedding model: {model_name}")
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': True}
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    print("Embedding model loaded.")
    return embeddings

def load_metadata():
    if os.path.exists(METADATA_FILE):
        try:
            with open(METADATA_FILE, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"Warning: Metadata file {METADATA_FILE} is corrupted. Will be treated as empty.")
    return {"indexed_files": {}, "embedding_model": ""}

def save_metadata(indexed_files_details, embedding_model_name):
    os.makedirs(VECTORSTORE_BASE_PATH, exist_ok=True)
    metadata = {
        "indexed_files": indexed_files_details,
        "embedding_model": embedding_model_name
    }
    with open(METADATA_FILE, 'w') as f:
        json.dump(metadata, f, indent=4)
    print("Metadata saved.")

def get_pdf_file_details(folder_path):
    pdf_details = {}
    if not os.path.isdir(folder_path):
        print(f"Warning: PDFs path '{folder_path}' does not exist or is not a directory.")
        os.makedirs(folder_path, exist_ok=True) # Create if not exists
        return pdf_details
        
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".pdf"):
            filepath = os.path.join(folder_path, filename)
            try:
                stat = os.stat(filepath)
                pdf_details[filepath] = {
                    "mtime": stat.st_mtime,
                    "size": stat.st_size
                }
            except OSError as e:
                print(f"Warning: Could not stat file {filepath}: {e}")
    return pdf_details

def load_pdfs_and_create_chunks(list_of_pdf_paths):
    """Loads specific PDFs, splits them into chunks."""
    docs_for_processing = []
    for pdf_path in list_of_pdf_paths:
        try:
            print(f"Loading PDF: {os.path.basename(pdf_path)}")
            loader = PyPDFLoader(pdf_path)
            docs_for_processing.extend(loader.load())
        except Exception as e:
            print(f"Error loading PDF {pdf_path}: {e}")
            continue # Skip problematic PDF

    if not docs_for_processing:
        print("No documents loaded from the provided PDF paths.")
        return None
    
    print(f"Loaded {len(docs_for_processing)} pages from {len(list_of_pdf_paths)} PDF(s).")
    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    doc_chunks = text_splitter.split_documents(docs_for_processing)
    print(f"Split into {len(doc_chunks)} chunks.")
    return doc_chunks


def initialize_or_update_vectorstore(embeddings, force_rebuild=False):
    """
    Manages the FAISS vector store and PDF metadata.
    - Loads existing store if metadata matches and no rebuild is forced.
    - Adds new PDFs incrementally if possible.
    - Triggers a full rebuild if PDFs were removed/changed or if forced.
    """
    print("\n--- Managing Vector Store & Documents ---")
    current_pdf_details_on_disk = get_pdf_file_details(PDFS_PATH)
    metadata = load_metadata()
    
    vectorstore = None
    perform_full_rebuild = force_rebuild
    rebuild_reason = "User forced rebuild." if force_rebuild else ""

    # 1. Check for embedding model mismatch
    if metadata.get("embedding_model") and metadata["embedding_model"] != EMBEDDING_MODEL_NAME:
        perform_full_rebuild = True
        rebuild_reason = f"Embedding model changed (was '{metadata['embedding_model']}', now '{EMBEDDING_MODEL_NAME}')."
    
    # 2. Analyze file changes if not already decided to rebuild
    if not perform_full_rebuild:
        new_files_to_process_paths = []
        changed_files_detected = False
        removed_files_detected = False

        # Check for new or changed files
        for filepath, current_details in current_pdf_details_on_disk.items():
            if filepath not in metadata.get("indexed_files", {}):
                new_files_to_process_paths.append(filepath)
                print(f"Detected new file: {os.path.basename(filepath)}")
            elif (metadata["indexed_files"][filepath]['mtime'] != current_details['mtime'] or
                  metadata["indexed_files"][filepath]['size'] != current_details['size']):
                changed_files_detected = True
                rebuild_reason = f"File '{os.path.basename(filepath)}' changed."
                print(rebuild_reason)
                break # One changed file means full rebuild

        if changed_files_detected:
            perform_full_rebuild = True
        else:
            # Check for removed files
            for indexed_filepath in metadata.get("indexed_files", {}).keys():
                if indexed_filepath not in current_pdf_details_on_disk:
                    removed_files_detected = True
                    rebuild_reason = f"File '{os.path.basename(indexed_filepath)}' removed."
                    print(rebuild_reason)
                    break # One removed file means full rebuild
            if removed_files_detected:
                perform_full_rebuild = True
    
    # 3. Perform full rebuild if needed
    if perform_full_rebuild:
        print(f"Full rebuild of vector store initiated. Reason: {rebuild_reason}")
        if os.path.exists(VECTORSTORE_BASE_PATH):
            print(f"Removing existing vector store at {VECTORSTORE_BASE_PATH}...")
            shutil.rmtree(VECTORSTORE_BASE_PATH)
        
        all_pdf_paths_on_disk = list(current_pdf_details_on_disk.keys())
        if not all_pdf_paths_on_disk:
            print("No PDF files found in ./pdfs folder to build the index.")
            save_metadata({}, EMBEDDING_MODEL_NAME) # Save empty metadata
            return None

        print("Processing all PDFs for rebuild...")
        doc_chunks = load_pdfs_and_create_chunks(all_pdf_paths_on_disk)
        
        if doc_chunks:
            print("Creating new FAISS index from all PDFs...")
            vectorstore = FAISS.from_documents(doc_chunks, embeddings)
            os.makedirs(VECTORSTORE_BASE_PATH, exist_ok=True)
            vectorstore.save_local(VECTORSTORE_BASE_PATH)
            save_metadata(current_pdf_details_on_disk, EMBEDDING_MODEL_NAME)
            print("Vector store rebuilt and saved.")
        else:
            print("No documents to build after processing PDFs. Vector store is empty.")
            save_metadata({}, EMBEDDING_MODEL_NAME)
            vectorstore = None
        
        print("--- Vector Store Management Complete (Rebuild) ---")
        return vectorstore

    # 4. Load existing store or handle incremental additions
    else:
        # Try to load existing vector store
        faiss_index_file = os.path.join(VECTORSTORE_BASE_PATH, "index.faiss")
        if os.path.exists(faiss_index_file):
            try:
                print(f"Loading existing vector store from {VECTORSTORE_BASE_PATH}...")
                vectorstore = FAISS.load_local(
                    VECTORSTORE_BASE_PATH,
                    embeddings,
                    allow_dangerous_deserialization=True
                )
                print("Existing vector store loaded.")
            except Exception as e:
                print(f"Error loading existing vector store: {e}. Will attempt to rebuild.")
                # Trigger a rebuild by calling self with force_rebuild
                return initialize_or_update_vectorstore(embeddings, force_rebuild=True)
        
        # Add new files if any (and store exists or can be created)
        if new_files_to_process_paths:
            print(f"Processing {len(new_files_to_process_paths)} new PDF(s)...")
            new_doc_chunks = load_pdfs_and_create_chunks(new_files_to_process_paths)

            if new_doc_chunks:
                if vectorstore:
                    print(f"Adding {len(new_doc_chunks)} new chunks to existing vector store.")
                    vectorstore.add_documents(new_doc_chunks)
                else: # No existing store, create new one with these new files
                    print(f"Creating new vector store with {len(new_doc_chunks)} chunks from new files.")
                    vectorstore = FAISS.from_documents(new_doc_chunks, embeddings)
                
                os.makedirs(VECTORSTORE_BASE_PATH, exist_ok=True)
                vectorstore.save_local(VECTORSTORE_BASE_PATH)
                
                # Update metadata: add new file details to existing
                updated_metadata_files = metadata.get("indexed_files", {}).copy()
                for fp in new_files_to_process_paths:
                    if fp in current_pdf_details_on_disk: # Ensure it's still there
                         updated_metadata_files[fp] = current_pdf_details_on_disk[fp]
                save_metadata(updated_metadata_files, EMBEDDING_MODEL_NAME)
                print("New PDFs processed and vector store updated.")
            else:
                print("No valid content found in new PDFs to add.")
        
        elif not vectorstore and not current_pdf_details_on_disk:
            print("No PDFs in ./pdfs folder and no existing index. Vector store remains empty.")
            save_metadata({}, EMBEDDING_MODEL_NAME) # Ensure metadata is empty
        
        elif not vectorstore and current_pdf_details_on_disk: # No store, but PDFs exist (first run after deleting index manually)
            print("No existing vector store, but PDFs found. Building new index.")
            return initialize_or_update_vectorstore(embeddings, force_rebuild=True)
        
        elif not new_files_to_process_paths and vectorstore:
            print("No changes detected in PDF files. Using existing vector store.")

    if not vectorstore and current_pdf_details_on_disk:
        print("Warning: PDFs are present, but the vector store could not be initialized. Try 'rebuild index'.")
    elif not vectorstore and not current_pdf_details_on_disk:
        print("No PDFs found. Add PDF files to the './pdfs' directory and use 'rebuild index'.")

    print("--- Vector Store Management Complete ---")
    return vectorstore


def get_qa_chain(llm, vectorstore, prompt_template_str):
    if not vectorstore:
        print("Cannot create QA chain: Vector store is not available.")
        return None

    prompt = PromptTemplate(template=prompt_template_str, input_variables=["context", "question"])
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={'k': 4}), # Retrieve top 4
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    return qa_chain

# --- Main Application Logic ---
def main():
    if not GROQ_API_KEY:
        print("GROQ_API_KEY not found. Please set it in your .env file.")
        return

    embeddings = get_embedding_model()
    vectorstore = initialize_or_update_vectorstore(embeddings)

    print(f"Initializing LLM: {LLM_MODEL_NAME} with Groq...")
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name=LLM_MODEL_NAME,
        temperature=0.2
    )
    print("LLM initialized.")

    prompt_template_str = """
    You are an AI assistant for querying PDF documents.
    Use the following pieces of context from the documents to answer the question.
    If the answer is not found in the provided context, state that clearly. Do not make up an answer.
    Be concise and directly answer the question. If possible, cite the source document or page.

    Context:
    {context}

    Question: {question}

    Helpful Answer:
    """
    qa_chain = None
    if vectorstore:
        qa_chain = get_qa_chain(llm, vectorstore, prompt_template_str)
    else:
        print("Warning: Vector store is not initialized. Q&A functionality will be limited until PDFs are processed.")


    print("\n--- Chat with your PDFs! ---")
    print("Type 'exit' or 'quit' to end.")
    print("Type 'rebuild index' to re-process all PDFs in the './pdfs' folder.")
    print("Type 'status' to see current index information.\n")

    while True:
        user_input = input("Ask a question or type a command: ").strip()

        if user_input.lower() in ["exit", "quit"]:
            print("Exiting chat. Goodbye!")
            break
        
        if user_input.lower() == "rebuild index":
            print("Rebuilding index as requested by user...")
            vectorstore = initialize_or_update_vectorstore(embeddings, force_rebuild=True)
            if vectorstore:
                qa_chain = get_qa_chain(llm, vectorstore, prompt_template_str)
                print("Index rebuilt. You can now ask questions.")
            else:
                qa_chain = None
                print("Index rebuild resulted in an empty store. Add PDFs to './pdfs' and try again.")
            continue
        
        if user_input.lower() == "status":
            metadata = load_metadata()
            print("\n--- Index Status ---")
            print(f"Embedding Model: {metadata.get('embedding_model', 'N/A')}")
            indexed_files = metadata.get('indexed_files', {})
            if indexed_files:
                print(f"Number of indexed PDF files: {len(indexed_files)}")
                print("Indexed files:")
                for i, (filepath, details) in enumerate(indexed_files.items()):
                    filename = os.path.basename(filepath)
                    # Convert mtime to readable format
                    mtime_readable = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(details.get('mtime', 0)))
                    print(f"  {i+1}. {filename} (Size: {details.get('size', 0)} bytes, Last Modified: {mtime_readable})")
            else:
                print("No PDF files are currently indexed.")
            
            if vectorstore and hasattr(vectorstore, 'index') and vectorstore.index:
                 print(f"Vector store contains {vectorstore.index.ntotal} embedding vectors.")
            else:
                print("Vector store is not loaded or is empty.")
            print("---------------------\n")
            continue

        if not user_input:
            continue
        
        if not qa_chain:
            print("Cannot process query: The Q&A system is not ready (likely no PDFs indexed). Try 'rebuild index' after adding PDFs.")
            continue

        print("\nThinking...")
        try:
            result = qa_chain.invoke({"query": user_input})
            answer = result.get("result", "No answer found.")
            source_docs = result.get("source_documents")

            print("\n--- Answer ---")
            print(answer)

            if source_docs:
                print("\n--- Sources ---")
                # Consolidate sources by filename
                sources_summary = {}
                for doc in source_docs:
                    source_file = os.path.basename(doc.metadata.get('source', 'Unknown Source'))
                    page_number = doc.metadata.get('page', 'N/A')
                    if source_file not in sources_summary:
                        sources_summary[source_file] = set()
                    if page_number != 'N/A':
                         sources_summary[source_file].add(str(page_number + 1)) # PyPDFLoader is 0-indexed

                for file, pages in sources_summary.items():
                    if pages:
                        print(f"  - File: '{file}', Pages: {', '.join(sorted(list(pages), key=int))}")
                    else:
                        print(f"  - File: '{file}' (page N/A)")
            print("\n" + "-"*30 + "\n")

        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    # Ensure pdfs directory exists
    if not os.path.exists(PDFS_PATH):
        os.makedirs(PDFS_PATH)
        print(f"Created directory: {PDFS_PATH}. Please add your PDF files there.")
    main()


