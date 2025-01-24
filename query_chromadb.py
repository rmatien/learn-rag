from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

embedding = HuggingFaceEmbeddings(
    model_name= "tomaarsen/static-similarity-mrl-multilingual-v1",
    # model_name= "intfloat/multilingual-e5-small",
    # model_name="HIT-TMG/KaLM-embedding-multilingual-mini-instruct-v1.5",
    model_kwargs={"device": "cpu"}  # Gunakan "cuda" jika ada GPU
)

persist_directory = "db/chroma"  # Direktori penyimpanan database
vector_db = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding
)

# 5. Verifikasi
print(f"Total dokumen tersimpan: {vector_db._collection.count()}")

# Contoh pencarian
query = "daya tampung prodi pendidikan dokter"
# docs_scores = vector_db.similarity_search_with_score(query, k=5)
docs_scores = sorted(vector_db.similarity_search_with_score(query, k=5), key=lambda x: x[1], reverse=True)
print("\nHasil pencarian:")
for doc, score in docs_scores:
    print(f"similiarity: {score}")
    print(doc.page_content)
    print("---")