import json
from pathlib import Path
from pydantic import BaseModel

MAX_CHARS = 800  # ~200 tokens, safety margin under the 256-token model limit

INPUT_PATH = r"C:\Users\HI\Desktop\MY-PROJECTS\RAG-SERVICE-BOT\data\processed\documents.jsonl"
OUTPUT_PATH = r"C:\Users\HI\Desktop\MY-PROJECTS\RAG-SERVICE-BOT\data\processed\chunks.jsonl"


class Chunk(BaseModel):
    chunk_id: str
    document_id: str
    heading: str
    content: str


def split_by_heading(content: str):
   
    lines = content.split("\n")
    sections = []
    current_heading = "intro"
    current_lines = []
    inside_code_block = False

    for line in lines:
        if line.strip().startswith("```"):
            inside_code_block = not inside_code_block

        is_heading = (
            (line.startswith("##") or line.startswith("###"))
            and not inside_code_block
        )

        if is_heading:
            sections.append((current_heading, "\n".join(current_lines)))
            current_heading = line.strip()
            current_lines = []
        else:
            current_lines.append(line)

    sections.append((current_heading, "\n".join(current_lines)))

    return [(h, t) for h, t in sections if t.strip()]


def split_oversized(text: str, max_chars: int = MAX_CHARS):

    if len(text) <= max_chars:
        return [text]

    pieces = []
    for i in range(0, len(text), max_chars):
        piece = text[i : i + max_chars]
        if piece.strip():
            pieces.append(piece)
    return pieces


def build_chunks_for_document(doc: dict) -> list[Chunk]:
    document_id = doc["source_path"]
    sections = split_by_heading(doc["content"])

    chunks = []
    counter = 0
    for heading, text in sections:
        for piece in split_oversized(text):
            chunk = Chunk(
                chunk_id=f"{document_id}#{counter}",
                document_id=document_id,
                heading=heading,
                content=piece.strip(),
            )
            chunks.append(chunk)
            counter += 1

    return chunks


def main():
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        documents = [json.loads(line) for line in f]

    all_chunks = []
    for doc in documents:
        all_chunks.extend(build_chunks_for_document(doc))

    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(chunk.model_dump_json() + "\n")

    print(f"{len(documents)} documents -> {len(all_chunks)} chunks")
    print(f"written to {OUTPUT_PATH}")

main()
