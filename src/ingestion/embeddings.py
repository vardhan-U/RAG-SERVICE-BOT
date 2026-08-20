import os,json,pathlib 
from sentence_transformers import SentenceTransformer
import numpy as np
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

input_path = r"C:\Users\HI\Desktop\MY-PROJECTS\RAG-SERVICE-BOT\data\processed\chunks.jsonl"
output_path = r"C:\Users\HI\Desktop\MY-PROJECTS\RAG-SERVICE-BOT\src\ingestion\embedding"





local_model_path = r'C:\Users\HI\Desktop\MY-PROJECTS\RAG-SERVICE-BOT\src\ingestion\local_model'

model = SentenceTransformer(local_model_path)
print("loaded model")

text_to_embed =[]
print("opemimg file")
chunk_ids=[]
with open(input_path,"r",encoding="utf-8") as file:
    for line in file:
        chunk = json.loads(line.strip())
        text_to_embed.append(str(chunk["heading"])+"\n"+str(chunk["content"]))
        chunk_ids.append(str(chunk["chunk_id"]))
embeddings = model.encode(text_to_embed,show_progress_bar=True)

np.save(r"C:\Users\HI\Desktop\MY-PROJECTS\RAG-SERVICE-BOT\src\ingestion\embedding_store.npy",embeddings)
np.save(r"C:\Users\HI\Desktop\MY-PROJECTS\RAG-SERVICE-BOT\src\ingestion\chunkids.npy",chunk_ids)