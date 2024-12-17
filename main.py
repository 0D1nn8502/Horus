from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document 
from pydantic import BaseModel 
import google.generativeai as genai 
import json 
import re 
from langchain.text_splitter import RecursiveCharacterTextSplitter 


app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (or specify your extension origin)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Define paths
PERSIST_DIR = "./chroma_storage" 

## Schema for article requests ## 
class ArticleRequest(BaseModel):
    
    """ Schema for article request """
    articleText: str
    mediaSrc: list[str] = None


class FindSimilarRequest(BaseModel): 
    
    """ Schema for vectordb queries """
    Supportive : list[str] | None 
    Contrasting : list[str] | None 
   

class FindSimilarwMediaSrc (BaseModel): 
    
    """ For querying with a specific media src """
    Supportive : list[str] 
    Contrasting : list[str] 
    MediaSrc : str 
    
    

class AddReqSchema(BaseModel): 
    
    """ Schema for adding text to vector db """
    articleText : str 
    sourceUrl : str  
    

class FindSimilarResponse(BaseModel): 
    
    """ Schema for findsimilar endpoint response """ 
    Support : list[str] 
    Contrast : list[str] 

 

@app.get("/")
def read_root():
    return {'content':"Hello world"} 



## Will receive something like : const testQueryobj = {'Supportive': ['Israeli soldiers West Bank Hebron Palestinian child death football'], 'Contrasting': ['Israeli military self-defense Hebron shooting investigation']};   

@app.post("/findsimilar")
def findSimilar(request: FindSimilarRequest):    
    """Given a couple of queries, return similar documents"""             

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = Chroma(
        collection_name="news_articles",
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR,
    )
    
    if request.Supportive and request.Contrasting : 
    
        sup = []
        cont = []
     
        unique_urls = set() 
    
        for x in request.Supportive:
            supRes = vector_store.similarity_search(str(x), k=3)   
            for j in supRes:
            
                urlz = j.metadata.get("url")   
                
                if urlz not in unique_urls:
                    unique_urls.add(urlz) 
                    sup.append(urlz)  
                   
        
        for y in request.Contrasting: 
            conRes = vector_store.similarity_search(str(y), k=3)  
            for i in conRes: 
            
                urlz = i.metadata.get("url")   
            
                if urlz not in unique_urls: 
                    unique_urls.add(urlz)  
                    cont.append(urlz)  
        

        return {"Supporting": sup, "Contrasting": cont}  
    
    return {"Error":"Invalid request format, give an object as defined in schema"} 



@app.post("/findsimilarmedia") 
def findmedia(request:FindSimilarwMediaSrc):
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = Chroma(
        collection_name="news_articles",
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR,
    )
     
    
    if request.Supportive and request.Contrasting and request.MediaSrc:  
    
        resultz = [] 
        unique_urls = set() 

        
        ## Filter based on media_source metadata ## 
        for x in request.Supportive:
            supRes = vector_store.similarity_search(
                str(x), 
                k=3,
                filter={"media_source": request.MediaSrc}   
            )   
            
            for j in supRes:
            
                urlz = j.metadata.get("url")   
                
                if urlz not in unique_urls:
                    unique_urls.add(urlz) 
                    resultz.append(urlz)  
                   
        
        for y in request.Contrasting: 
            conRes = vector_store.similarity_search(
                str(y), 
                k=3, 
                filter={"media_source": request.MediaSrc}     
            )  
            
            for i in conRes: 
            
                urlz = i.metadata.get("url")   
            
                if urlz not in unique_urls: 
                    unique_urls.add(urlz)  
                    resultz.append(urlz)  
        
        
        print(resultz) 
        return {"Results":resultz} 
         
         
    return {"Error":"Invalid request format, give an object as defined in schema"}  
    



## Given article text, generate suitable vector database queries ##
@app.post("/genquery") 
def convAndFind(request: ArticleRequest):
    articleText = request.articleText
    
    
    inputText = f"""
    Based on the events and people discussed in the given article, {articleText},
    generate two small queries for a vector database so that it returns results with diverse view points related to the given article.
    It should be in JSON format (with no other text or null output) as follows:
    {{
      "Query": [
        {{
          "supportive": "Query text", 
        }},
        {{
          "contrasting": "Second query text", 
          
        }}
      ]
    }}
    """
    
    genai.configure(api_key="AIzaSyDvQmkr-3i3V8a3jRG0YbqcYPb_pb3vDAk")
    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
    )
    
    geminiResponse = model.generate_content(inputText)
    
    # Extract the JSON content
    response_dict = geminiResponse.to_dict()
    
    # Extract the JSON content
    try:
        # Navigate to the nested JSON string
        json_string = response_dict['candidates'][0]['content']['parts'][0]['text']
        # Print the raw JSON string for debugging
        print("Raw JSON String:", json_string)
        
        # Clean up any code block markers or extraneous text
        json_string = re.sub(r'^```json|```$', '', json_string).strip()
        # Print the cleaned JSON string for debugging
        print("Cleaned JSON String:", json_string)
        
        # Parse the JSON string
        extracted_data = json.loads(json_string)
        
        queries = extracted_data.get("Query", [])

        supportive = [i.get('supportive') for i in queries if i.get('supportive') is not None]
        contrasting = [i.get('contrasting') for i in queries if i.get('contrasting') is not None]

        return  {
            "Supportive" : supportive, 
            "Contrasting" : contrasting
        }
        
    except (KeyError, ValueError, json.JSONDecodeError) as e:
        return {"Error": f"Failed to extract queries: {str(e)}"}
        
        
@app.post("/addarticle")   
def addArticle(articleAdd : AddReqSchema):   
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = Chroma(
        collection_name="news_articles",
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR,
    )
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=750,  # Customize chunk size
        chunk_overlap=100,  # Optional overlap between chunks
    ) 
    
    chunks = text_splitter.split_text(articleAdd.articleText) 
    
    doc_chunks = [] 
    for chunk in chunks: 
        doc_chunks.append(
            Document(page_content=chunk, metadata={"url" : articleAdd.sourceUrl})   
        )
           
   
    try:
        vector_store.add_documents(documents=doc_chunks)
        vector_store.persist()  # Save changes to the persistent directory
        
    except Exception as e:
        return {"success": False, "error": str(e)}
    
    return {"success": True, "message": "Article added successfully!"} 
    
    
    
