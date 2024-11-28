from typing import Union
from fastapi import FastAPI
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import pipeline
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate 
from langchain_community.llms.ollama import Ollama 

app = FastAPI()

# Define paths
PERSIST_DIR = "./chroma_storage"

@app.get("/")
def read_root():
    return "Hello world"


@app.get("/findsimilar") 
## Just return the similar documents ## 
def findSimilar(query:str, k=5, mediaSrc=None):               
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = Chroma(
        collection_name="news_articles",
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR,
    )                                                                         
    
    if mediaSrc: 
        base_retreiver = vector_store.as_retriever(
            search_kwargs = {'filter': {"media_source":{"$in":mediaSrc}}}         
        )
        
        results = base_retreiver.invoke(query)  
    
    else: 
        results = vector_store.similarity_search(query, k=k) 
        
    return results 
    