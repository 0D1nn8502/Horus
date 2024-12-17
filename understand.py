from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings 
from collections import Counter 
from scraper import similaritySearch
from sentence_transformers import SentenceTransformer  
from langchain.text_splitter import RecursiveCharacterTextSplitter


PERSIST_DIR = "./chroma_storage" 

query = "Tell me about the conflict between China and Taiwan" 

userText = "A motion to impeach South Korean President Yoon Suk Yeol has failed as National Assembly Speaker Woo Won-shik closed the session, which had stalled for hours after legislators from the governing party boycotted the vote." 


embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Create ChromaDB vector store 
vector_store = Chroma(
        collection_name="news_articles",  # Name of your collection
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR,
)


x = vector_store.similarity_search(
    
    userText,
    k = 7, 
    filter={"media_source" : "The Onion"}
)   


for y in x: 
    print(x) 
    


