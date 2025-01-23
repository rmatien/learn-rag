# Install library yang dibutuhkan
# pip install pypdf langchain langchain-community chromadb sentence-transformers

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# 1. Load dan Ekstrak PDF
pdf_path = "pdf/draf panduan SNBP web 2025.pdf"
loader = PyPDFLoader(pdf_path)
pages = loader.load_and_split()

# 2. Split Dokumen menjadi Chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,      # Ukuran chunk dalam karakter
    chunk_overlap=200,    # Overlap antar chunk
    length_function=len,
    is_separator_regex=False,
)
texts = text_splitter.split_documents(pages)

# 3. Inisialisasi Embedding Model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}  # Gunakan "cuda" jika ada GPU
)

# 4. Simpan ke ChromaDB
persist_directory = "db/chroma"  # Direktori penyimpanan database

# Membuat vector store baru
vector_db = Chroma.from_documents(
    documents=texts,
    embedding=embeddings,
    persist_directory=persist_directory
)

# Simpan secara permanen
vector_db.persist()

# 5. Verifikasi
print(f"Total dokumen tersimpan: {vector_db._collection.count()}")

# Contoh pencarian
query = "Apa definisi machine learning?"
docs = vector_db.similarity_search(query, k=3)
print("\nHasil pencarian:")
for doc in docs:
    print(f"- {doc.page_content[:100]}...")