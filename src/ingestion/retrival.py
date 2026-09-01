import faiss
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
import json
emb1 = np.load(r"C:\Users\HI\Desktop\MY-PROJECTS\RAG-SERVICE-BOT\src\ingestion\embedding_store.npy")
@st.cache_resource
def load_model(path):
    return SentenceTransformer(path)
d = 384                           # dimension
nb = 2804                    # database size
nq = 1                       # nb of queries
np.random.seed(1234)             # make reproducible
xb = emb1.astype('float32')
xb = np.array(xb).astype('float32')

# faiss.normalize_L2(nb)
# faiss.normalize_L2(nq)
index = faiss.IndexFlatIP(d)   
index.add(xb)                  


def search_query(query_embedding,k=4):

    xq = np.array(query_embedding).astype('float32')

    D, I = index.search(xq, k)
    
    return D[0], I[0]

def embed_query(query_question):
   
    local_model_path = r'C:\Users\HI\Desktop\MY-PROJECTS\RAG-SERVICE-BOT\src\ingestion\local_model'

    model = load_model(local_model_path)

    embedding = model.encode(query_question,show_progress_bar=True,normalize_embeddings=True)
    return embedding

def query_related_chunk_ids(embedding):
    distances, indices = search_query(embedding.reshape(1, -1))
    return indices

def print_related_chunks(indices):
    chunk_ids = np.load(r"C:\Users\HI\Desktop\MY-PROJECTS\RAG-SERVICE-BOT\src\ingestion\chunkids.npy")
    with open(r"C:\Users\HI\Desktop\MY-PROJECTS\RAG-SERVICE-BOT\data\processed\chunks.jsonl","r",encoding="utf-8") as f:
        all_chunks = [json.loads(line) for line in f]
        for idx in indices:
            target_id = chunk_ids[idx]
            match = next(c for c in all_chunks if c["chunk_id"] == target_id)
            print(match["content"])
            print("\n\n\n\n")

a=embed_query("""How can I allow only GET and POST requests from a specific frontend while allowing credentials?""")
b=query_related_chunk_ids(a)
print_related_chunks(b)