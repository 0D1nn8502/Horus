from langchain.vectorstores import Chroma 
from langchain.embeddings import HuggingFaceEmbeddings

PERSIST_DIR = "./chroma_storage"


embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = Chroma(
        collection_name="news_articles",
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR,
)


print(vector_store.get(include=["metadatas"])) 

