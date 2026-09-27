import faiss
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
import json
from openai import OpenAI
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


def search_query(query_embedding,k=2):

    xq = np.array(query_embedding).astype('float32')

    D, I = index.search(xq, k)
    print("query search done")
    return D[0], I[0]

def embed_query(query_question):
   
    local_model_path = r'C:\Users\HI\Desktop\MY-PROJECTS\RAG-SERVICE-BOT\src\ingestion\local_model'

    model = load_model(local_model_path)

    embedding = model.encode(query_question,show_progress_bar=True,normalize_embeddings=True)
    print("embedding qury done")
    return embedding

def query_related_chunk_ids(embedding):
    distances, indices = search_query(embedding.reshape(1, -1))
    print("ids captured")
    return indices

def related_chunks(indices):
    chunk_ids = np.load(r"C:\Users\HI\Desktop\MY-PROJECTS\RAG-SERVICE-BOT\src\ingestion\chunkids.npy")
    relAted_chunks_list =[] 
    chunk_heading=[]
    with open(r"C:\Users\HI\Desktop\MY-PROJECTS\RAG-SERVICE-BOT\data\processed\chunks.jsonl","r",encoding="utf-8") as f:
        all_chunks = [json.loads(line) for line in f]
        for idx in indices:
            target_id = chunk_ids[idx]
            match = next(c for c in all_chunks if c["chunk_id"] == target_id)
            relAted_chunks_list.append(match["content"])
            chunk_heading.append(match["heading"])
    print("Chunks gathered")
    return relAted_chunks_list, chunk_heading



def generate_responese(chunks,query):
    client = OpenAI(base_url="http://localhost:11434/v1",api_key="OLLAMA")
    chunks_text = "\n\n".join(chunks)
    
    system_inst = (
    "You are an expert FastAPI backend developer. Your job is to answer user queries using high-level, "
    "idiomatic framework features found in the context chunks. "
    "CRITICAL RULES:\n"
    "1. Never write low-level or manual bytes-parsing logic unless explicitly requested.\n"
    "2. Use native FastAPI classes (like Form, Body, Depends, Path, Query) to handle requests.\n"
    "3. Keep code blocks strictly valid, idiomatic Python.\n"
    "4. If the context chunks do not contain a clear framework solution, state 'I don't know'."
)



    messages = [{"role":"system","content":system_inst},{
                     "role":"user",
                     "content":f"Now this is the query:{query} and these are chunks:{chunks_text}"
                 }]
    # response = client.chat.completions.create(model="phi3:instruct",messages=messages)
    # print(response.choices[0].message.content)  
    # return response.choices[0].message.content
    try:
        response = client.chat.completions.create(model="phi3:instruct", messages=messages)
        answer = response.choices[0].message.content
        print(answer)
        return answer
    except Exception as e:
        print("LLM CALL FAILED:", repr(e))
        return None



