from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from typing import Tuple

# 1. Inisialisasi Model & Embeddings

embeddings = HuggingFaceEmbeddings(
    model_name= "tomaarsen/static-similarity-mrl-multilingual-v1",
    # model_name= "intfloat/multilingual-e5-small",
    # model_name="HIT-TMG/KaLM-embedding-multilingual-mini-instruct-v1.5",
    model_kwargs={"device": "cpu"}  # Gunakan "cuda" jika ada GPU
)

llm = OllamaLLM(model="llama3.2")  # Ganti dengan model Ollama yang Anda gunakan

# 2. Load Chroma DB yang sudah ada
persist_directory = "db/chroma"
vector_db = Chroma(
    persist_directory=persist_directory,
    embedding_function=embeddings
)

# 3. Fungsi Utama dengan Hybrid Strategy dan Fallback
def hybrid_rag_answer(
    query: str, 
    rag_threshold: float = 0.6, 
    model_confidence_threshold: float = 0.3
) -> Tuple[str, float, str]:
    
    # Langkah 1: Cari dokumen di RAG
    docs_and_scores = vector_db.similarity_search_with_score(query, k=5)
    best_similarity = docs_and_scores[0][1] / 100000 if docs_and_scores else 0.0 #untuk model tomaarsen/static-similarity-mrl-multilingual-v1
    # best_similarity = docs_and_scores[0][1] if docs_and_scores else 0.0
    
    # Langkah 2: Hybrid Logic Berdasarkan Similarity Score
    if best_similarity >= rag_threshold:
        # Case 1: RAG-dominant Answer
        return get_rag_answer(query), best_similarity, "RAG"
    
    elif best_similarity >= 0.4:
        # Case 2: Hybrid Answer (RAG + Model)
        rag_context = "\n".join([doc.page_content for doc, _ in docs_and_scores])
        hybrid_prompt = f"""
        [Konteks RAG]: {rag_context}
        [Pengetahuan Umum]: (Tambahkan informasi umum jika diperlukan)
        [Pertanyaan]: {query}
        
        Jawablah dengan menggabungkan informasi dari konteks dan pengetahuan umum:
        """
        hybrid_answer = llm.invoke(hybrid_prompt)
        return hybrid_answer, best_similarity, "Hybrid"
    
    else:
        # Case 3: Model Answer dengan Fallback
        model_answer = llm.invoke(query)
        model_confidence = estimate_confidence(model_answer)  # Fungsi custom
        
        if model_confidence >= model_confidence_threshold:
            return model_answer, best_similarity, "Model"
        else:
            # Fallback Answer
            return get_fallback_answer(query), best_similarity, "Fallback"

# 4. Fungsi Pendukung
def get_rag_answer(query: str) -> str:
    prompt_template = """
    Jawablah HANYA berdasarkan konteks ini:
    {context}
    
    Pertanyaan: {question}
    Jawaban:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_db.as_retriever(),
        chain_type_kwargs={"prompt": prompt}
    )
    
    return chain.invoke({"query": query})["result"]

def estimate_confidence(answer: str) -> float:
    """Estimasi confidence berdasarkan panjang jawaban dan kata kunci (contoh sederhana)"""
    confidence_signals = [
        len(answer) > 50,  # Jawaban panjang dianggap lebih confident (PERLU DIPERHATIKAN !!!)
        any(keyword in answer.lower() for keyword in ["tahu", "yakin", "menurut data"]),
        "?" not in answer  # Hindari jawaban berbentuk pertanyaan
    ]
    return sum(confidence_signals) / 3  # Normalisasi ke 0-1

def get_fallback_answer(query: str) -> str:
    """Jawaban default jika sistem tidak yakin"""
    return (
        "Maaf, saya tidak menemukan informasi yang cukup untuk menjawab pertanyaan Anda. "
        "Silahkan coba formulasi ulang pertanyaan atau tanyakan hal lain."
    )

# 5. Contoh Penggunaan
query = "daya tampung prodi pendidikan dokter"
answer, similarity, source = hybrid_rag_answer(query)

print(f"Pertanyaan: {query}")
print(f"Sumber: {source}")
print(f"Similarity Score: {similarity}")
print(f"Jawaban:\n{answer}")
