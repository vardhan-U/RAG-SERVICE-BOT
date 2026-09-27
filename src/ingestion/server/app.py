from fastapi import FastAPI
from pydantic import BaseModel

from ..retrival import embed_query, query_related_chunk_ids, related_chunks, generate_responese

app = FastAPI()


class QueryRequest(BaseModel):
    question: str


@app.post("/query")
def query(request: QueryRequest):
    embedding = embed_query(request.question)
    indices = query_related_chunk_ids(embedding)
    chunks, headings = related_chunks(indices)

    if isinstance(chunks, str):
        chunks = [chunks]
    if isinstance(headings, str):
        headings = [headings]

    print("DEBUG chunks:", type(chunks), len(chunks))
    print("DEBUG headings:", type(headings), headings)

    answer = generate_responese(chunks=chunks, query=request.question)
    print("DEBUG answer:", answer)
    if answer is None:
        answer = "Couldn't generate an answer "

    sources = [{"heading": h} for h in headings]
    return {"answer": answer, "sources": sources}