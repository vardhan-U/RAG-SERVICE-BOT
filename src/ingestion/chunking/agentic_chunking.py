from openai import Openai
OLLAMA_BASE_URL = "http://localhost:11434/v1"

ollama = Openai(base_url=OLLAMA_BASE_URL, api_key='ollama')
def agent(data):
    response = ollama.chat.completions.create(model="llama3.2",
                                           messages=[{"role": "system", 
                                             "content":"So now you will be provided json data with the params title,id,content"
                                             ",source path and section return only another json object in a list(donot explain any further) with params as"
                                             " chunk_id = source_path#1,document_id=parent source path,heading=next ## if" 
                                             "there exist ### then put it as headings remeber not to exceed the json object size "
                                             "of 256 tokens"},
                                             {"role":"user",
                                              "content":data}])

    return response.choices[0].message.content
