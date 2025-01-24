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
    chunk_size=500,      # Ukuran chunk dalam karakter
    chunk_overlap=50,    # Overlap antar chunk
    length_function=len,
    separators=["\n\n", "\n", " ", ""]  # Optimalkan untuk bahasa Indonesia
)
docs = text_splitter.split_documents(pages)

# 3. Inisialisasi Sentence Transformer Model
embedding = HuggingFaceEmbeddings(
    model_name= "tomaarsen/static-similarity-mrl-multilingual-v1",
    # model_name= "intfloat/multilingual-e5-small",
    # model_name="HIT-TMG/KaLM-embedding-multilingual-mini-instruct-v1.5",
    model_kwargs={"device": "cpu"}  # Gunakan "cuda" jika ada GPU
)

# 4. Simpan ke ChromaDB dengan Embedding Kustom
persist_directory = "db/chroma"  # Direktori penyimpanan database
vector_db = Chroma.from_documents(
    documents=docs,
    embedding=embedding,  # Langsung gunakan fungsi embedding kustom
    persist_directory=persist_directory)

# 5. Verifikasi
print(f"Total dokumen tersimpan: {vector_db._collection.count()}")

# Contoh pencarian
query = "syarat pendaftaran snbp"
docs = vector_db.similarity_search(query, k=3)
print("\nHasil pencarian:")
for doc in docs:
    print(f"- {doc.page_content[:100]}...")