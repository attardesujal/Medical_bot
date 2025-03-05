import time
import os
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

DATA_PATH = "D:/PyCharm 2024.3.1.1/Medico/books"


# Step 1: Load raw PDF(s)
def load_pdf_files(data):
    start_time = time.time()

    # Get a list of PDF files
    pdf_files = [f for f in os.listdir(data) if f.endswith('.pdf')]

    loader = DirectoryLoader(data, glob='*.pdf', loader_cls=PyPDFLoader)

    try:
        documents = loader.load()
    except Exception as e:
        print(f"Error loading PDFs: {e}")
        return []

    end_time = time.time()
    print(f"\n✅ Loaded {len(pdf_files)} PDFs in {end_time - start_time:.2f} seconds")

    for file in pdf_files:
        print(f"📂 Processing: {file}")

    return documents


start_total_time = time.time()
documents = load_pdf_files(DATA_PATH)

if not documents:
    print("❌ No valid documents loaded. Exiting...")
    exit()


# Step 2: Create Chunks
def create_chunks(extracted_data):
    start_time = time.time()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    text_chunks = text_splitter.split_documents(extracted_data)
    end_time = time.time()
    print(f"\n✅ Created {len(text_chunks)} text chunks in {end_time - start_time:.2f} seconds")
    return text_chunks


text_chunks = create_chunks(documents)


# Step 3: Create Vector Embeddings
def get_embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


embedding_model = get_embedding_model()

# Step 4: Store embeddings in FAISS
DB_FAISS_PATH = "vectorstore/db_faiss"
start_faiss_time = time.time()

db = FAISS.from_documents(text_chunks, embedding_model)
db.save_local(DB_FAISS_PATH)

end_faiss_time = time.time()
print(f"\n✅ FAISS database saved in {end_faiss_time - start_faiss_time:.2f} seconds")

end_total_time = time.time()
print(f"\n🚀 Total execution time: {end_total_time - start_total_time:.2f} seconds")
