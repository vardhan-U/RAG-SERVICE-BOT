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


index = faiss.IndexFlatL2(d)   
index.add(xb)                  


def search_query(query_embedding,k=4):

    xq = np.array(query_embedding).astype('float32')

    D, I = index.search(xq, k)
    
    return D[0], I[0]


query = "what is an environment variable"
local_model_path = r'C:\Users\HI\Desktop\MY-PROJECTS\RAG-SERVICE-BOT\src\ingestion\local_model'

model = load_model(local_model_path)

embedding = model.encode(query,show_progress_bar=True,normalize_embeddings=True)

print(search_query(embedding.reshape(1,-1)))
chunk_ids = np.load(r"C:\Users\HI\Desktop\MY-PROJECTS\RAG-SERVICE-BOT\src\ingestion\chunkids.npy")

distances, indices = search_query(embedding.reshape(1, -1))


with open(r"C:\Users\HI\Desktop\MY-PROJECTS\RAG-SERVICE-BOT\data\processed\chunks.jsonl","r",encoding="utf-8") as f:
    all_chunks = [json.loads(line) for line in f]
    for idx in indices:
        target_id = chunk_ids[idx]
        match = next(c for c in all_chunks if c["chunk_id"] == target_id)
        print(match["heading"], "->", match["content"])