import os,glob,re
from pathlib import Path
from pydantic import BaseModel

class Document(BaseModel):
    id : str
    title : str
    content : str
    source_path : str
    section : str


filenames = glob.glob("C:/Users/HI/Desktop/MY-PROJECTS/RAG-SERVICE-BOT/fastapi/docs/en/docs/**/*.md")

files_metadata = []
for filename in filenames:
    pattern = r"(?<!#)#[^#\r\n][^\r\n]*"
    docs_root = Path("C:/Users/HI/Desktop/MY-PROJECTS/RAG-SERVICE-BOT/fastapi/docs/en/docs")
    rel_path = Path(filename).relative_to(docs_root)
    if len(rel_path.parts) > 1:
        section = rel_path.parts[0]
    else:
        section = ""
    with open(filename,"r",encoding="utf-8") as f:
        content = f.read()
        title_match = re.match(pattern, content)
        title =title_match.group().lstrip("#").strip() if title_match else str(Path(filename).stem)

        doc_scanning = Document(id=str(rel_path),
                          title=title,
                          content=content,
                          source_path=str(Path(filename)),
                          section=section)
        files_metadata.append(doc_scanning)
import os,glob,re
from pathlib import Path
from pydantic import BaseModel

class Document(BaseModel):
    id : str
    title : str
    content : str 
    source_path : str
    section : str


filenames = glob.glob("C:/Users/HI/Desktop/MY-PROJECTS/RAG-SERVICE-BOT/fastapi/docs/en/docs/**/*.md",recursive=True)

files_metadata = []
for filename in filenames:
    pattern = r"(?<!#)#[^#\r\n][^\r\n]*"
    docs_root = Path("C:/Users/HI/Desktop/MY-PROJECTS/RAG-SERVICE-BOT/fastapi/docs/en/docs")
    rel_path = Path(filename).relative_to(docs_root)
    if len(rel_path.parts) > 1:
        section = rel_path.parts[0]
    else:
        section = ""
    with open(filename,"r",encoding="utf-8") as f:
        content = f.read()
        title_match = re.match(pattern, content)
        title = title_match.group().lstrip("#").strip() if title_match else str(Path(filename).stem)
        doc_scanning = Document(id=str(rel_path),
                          title=title if title else str(Path(filename).stem),
                          content=content,
                          source_path=str(Path(filename)),
                          section=section)
        files_metadata.append(doc_scanning)
with open(Path(r"C:\Users\HI\Desktop\MY-PROJECTS\RAG-SERVICE-BOT\data\processed\documents.jsonl"),'w',encoding="utf-8") as f:
    for doc in files_metadata:
        line = doc.model_dump_json()
        f.write(line+"\n")