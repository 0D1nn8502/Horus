from langchain.document_loaders import DirectoryLoader 
from langchain.embeddings import OpenAIEmbeddings 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document 
from langchain.vectorstores import Chroma
from pathlib import Path  
import json 
import os 
import openai 

openai.api_key = os.getenv("OPENAI_API_KEY") 

DATA_PATH = "articles/foxNews" 
DATA_PATH2 = "articles/dailyBeast"  

PERSIST_DIR = "./chroma_storage"

def load_metadata(folder_path):
    """Load metadata from the articles.json file."""
    json_path = Path(folder_path) / "articles.json"
    if json_path.exists():
        with open(json_path, "r") as file:
            return json.load(file)  # Returns a dictionary mapping filenames to URLs
    return {}



def load_documents():
    """Load documents and enrich with URL metadata."""
    # Load metadata for both folders
    fox_metadata = load_metadata(DATA_PATH)
    beast_metadata = load_metadata(DATA_PATH2)

    loader = DirectoryLoader(DATA_PATH, glob="*.txt") 
    loader2 = DirectoryLoader(DATA_PATH, glob="*.txt") 
    
    fox_documents = loader.load()
    beast_documents = loader2.load() 
    
    def enrich_with_metadata(documents, metadata):
        enriched_docs = []
        for doc in documents:
            file_name = Path(doc.metadata["source"]).name  # Extract the filename
            url = metadata.get(file_name, "Unknown")  # Get URL or default to "Unknown"
            doc.metadata["url"] = url  # Add URL to metadata
            enriched_docs.append(doc)
        return enriched_docs 
    
    
    enriched_fox_docs = enrich_with_metadata(fox_documents, fox_metadata) 
    enriched_beast_docs = enrich_with_metadata(beast_documents, beast_metadata) 
    
    
    return enriched_fox_docs + enriched_beast_docs  
    


def split_documents(documents):
    """Split documents into smaller chunks while retaining metadata."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,  # Customize chunk size
        chunk_overlap=50,  # Optional overlap between chunks
    )
    split_docs = []
    for doc in documents:
        # Split document into chunks
        chunks = text_splitter.split_text(doc.page_content)
        # Create new Document objects for each chunk, retaining metadata
        for chunk in chunks:
            split_docs.append(
                Document(page_content=chunk, metadata=doc.metadata)
            )
    return split_docs 


def store_embeddings_in_chromadb(split_docs):
    """Generate embeddings for chunks and store them in ChromaDB."""
    # Initialize the embedding model (replace OpenAIEmbeddings with your choice)
    embeddings = OpenAIEmbeddings()  # Or use another supported embedding model
    
    # Create ChromaDB vector store
    vector_store = Chroma(
        collection_name="news_articles",  # Name of your collection
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR, 
    )
    
    # Add split documents to the vector store
    vector_store.add_documents(split_docs)
    vector_store.persist() 
    
    print(f"ChromaDB has been persisted at: {PERSIST_DIR}")

    
    return vector_store


documents = load_documents()  # Load documents (with metadata)
split_docs = split_documents(documents)  # Split into chunks
vector_store = store_embeddings_in_chromadb(split_docs)  # Store in ChromaDB
