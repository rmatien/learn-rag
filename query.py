from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# 1. Inisialisasi Model & Embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
llm = OllamaLLM(model="llama3.2")  # Ganti dengan model Ollama yang Anda gunakan

# 2. Load Chroma DB yang sudah ada
persist_directory = "db/chroma"
vector_db = Chroma(
    persist_directory=persist_directory,
    embedding_function=embeddings
)

# 3. Fungsi untuk Ambil Jawaban dengan Similarity Threshold
def get_answer_with_threshold(query, threshold=0.5):
    # Cari dokumen relevan dengan similarity score
    docs_and_scores = vector_db.similarity_search_with_score(query, k=1)
    
    # Ekstrak skor similarity tertinggi
    if not docs_and_scores:
        return llm.invoke(query), None, "Model"
    
    doc, score = docs_and_scores[0]
    
    # Jika skor melebihi threshold, gunakan RAG
    if score < threshold:  # Catatan: Chroma menggunakan distance (bukan similarity)
        similarity = 1 - score  # Konversi distance ke similarity (0-1)
        if similarity > threshold:
            # Buat prompt dengan konteks RAG
            prompt_template = """
            Jawablah pertanyaan ini berdasarkan konteks di bawah:
            [Konteks]: {context}
            [Pertanyaan]: {question}
            Jawaban:
            """
            prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )
            
            chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vector_db.as_retriever(),
                chain_type_kwargs={"prompt": prompt}
            )
            
            answer = chain.invoke({"query": query})["result"]
            return answer, similarity, "RAG"
    
    # Jika skor di bawah threshold, gunakan model langsung
    answer = llm.invoke(query)
    return answer, (1 - score) if docs_and_scores else 0, "Model"

# 4. Contoh Penggunaan
query = "daya tampung program studi ilmu hukum"
answer, similarity, source = get_answer_with_threshold(query, threshold=0.6)

print(f"Pertanyaan: {query}")
print(f"Sumber: {source}")
print(f"Similarity Score: {similarity}")
print(f"Jawaban:\n{answer}")
