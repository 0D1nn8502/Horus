from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import pipeline
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate 
from langchain_community.llms.ollama import Ollama 


# Define paths
PERSIST_DIR = "./chroma_storage"

# Load the LM (local model)
def load_local_model():

    """Load a local HuggingFace model for reasoning."""
    reasoning_pipeline = pipeline("text-generation", model="meta-llama/LLaMA-2-7b-hf", max_new_tokens=600, truncation=True)   # Use a model like GPT-2
    llm = HuggingFacePipeline(pipeline=reasoning_pipeline)
    return llm 


## Just return the similar documents ## 
def findSimilar(query, k=5, mediaSrc=None):            
    
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


# Query and reasoning function # 
def query_with_reasoning(query, k=3):
    """Query ChromaDB, retrieve relevant documents, and generate reasoning."""
    # Initialize embeddings and vector store
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = Chroma(
        collection_name="news_articles",
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR,
    )
    
    # Perform similarity search
    results = vector_store.similarity_search(query, k=k)
    
    # Load the reasoning LM
    llm = Ollama(model="llama2") 
    
    
    document_texts = "\n".join(
        [f"Document {i+1}: {doc.page_content[:250]}... (URL: {doc.metadata.get('url', 'Unknown')})"
         for i, doc in enumerate(results)]
    )
    
    prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant that answers the {query} using only the relevant news article from the following: {document_text}.", 
        ),
        ("human", "{document_text}"), 
    ]
    )
    
    chain = prompt | llm 
    
    response = chain.invoke(
        
        {
            "query": query, 
            "document_text": document_texts, 
        }
    ) 
    
    return results, response


articleText = """ Former U.S. President and Republican Donald Trump made a forceful comeback as he won a second term in office, defeating Vice-President Kamala Harris, the Democratic nominee, to become the 47th President of the United States. Republicans took control of the Senate, increasing their tally to at least 52 of the chamber’s 100 seats.

That he had run a campaign of personal insults, misogynistic jibes, comments with racist overtones, committed felonies, instigated a mob which went on to attack the Capitol, and threatened allies abroad, was not enough to keep the majority of Americans from electing Mr. Trump their leader, again.

U.S. Elections 2024 results | LIVE updates

At the end of the day, data suggest that voters wanted a break with current circumstances, driven primarily by their concerns around inflation and the economy, as well as illegal migration — the focus of Mr. Trump’s campaign. The former President had promised to secure the border, and said he would conduct mass deportations and impose heavy import tariffs to fund tax cuts.

Mr. Trump, 78, is only one of two Presidents elected to a non-consecutive second term, and he will also become the oldest President at the time of entry into office. U.S. President Joe Biden, 82, was pressured by his party to not seek another term, owing to concerns around age-related cognitive issues.

Ms. Harris, who could have become the first Indian American and Black woman President — had she won — campaigned on a theme of unity as she reached out to Independents and traditional Republicans who did not back Mr. Trump. She also focused on the risks to women’s reproductive rights, especially restrictions on abortions, under a conservative government. Her campaign dwelt on the authoritarian tendencies and plans of Mr. Trump.

Ms. Harris was suddenly thrust onto the stage after Mr. Biden withdrew from the race in July and had difficulty in quickly defining her positions on the economy, defending her changing stance on illegal migration, and separating herself adequately from her predecessor’s administration.

It turned out, in the end, that she could not convince enough voters.

During the night on Tuesday Ms. Harris lost the most crucial prize of Pennsylvania by 2.3 points, with 97% of the votes counted. She also lost or looked poised to lose other crucial parts of the ‘Blue Wall’ (traditionally Democratic strongholds), with Wisconsin and Michigan turning red.

Mr. Trump also won Ohio, Iowa, and West Virginia — States that are home to large numbers of blue-collar workers and form the Rust Belt, along with Wisconsin, Pennsylvania, Michigan, and Ohio.

Having lost two crucial battleground Sun Belt States — North Carolina and Georgia — Ms. Harris’s chances of winning were next to impossible after she lost Pennsylvania.

Polls have been suggesting that Mr. Trump’s message has reached new audiences, such as Hispanic voters and younger Gen-Z voters. Exit poll results (Edison Research via Reuters) suggest Mr. Trump’s support from Hispanic voters went up 13 points since the last election (45% versus Ms. Harris’s 53%) and initial results suggest he managed to retain Black voter support at 2020 levels (12% versus Ms. Harris’s 86%). Mr. Trump did especially well among Hispanic men, while his support among white women fell by 3 points (52% versus Ms Harris’s 47%).

By Wednesday afternoon, the Associated Press had called 292 Electoral College votes for Mr. Trump and 224 for Ms. Harris. At least 270 votes, distributed unevenly across States, are required to win the Presidency.

Remarkably, Mr. Trump was already projected to win the popular vote by Wednesday morning. George W. Bush won the popular vote in 2004, the last Republican to do so until now.

Republicans also gained control of the Senate, increasing their tally to 52 of the 100 seats, even as the contest for the U.S. House of Representatives remained open as The Hindu went to press. If Republicans win the House, they would control the White House and U.S. Congress, enabling them to push through a conservative agenda in Washington.

“…We’re going to help our country heal,” Mr. Trump said, speaking in West Palm Breach, Florida, before he had reached the 270 mark.

“It needs help very badly. We’re going to fix our borders,” he added. Later in the speech, he indicated that immigrants could come in legally.

Mr. Trump, who has survived two recent assassination attempts, said he had been told by others that “God spared my [his] life for a reason”.

Mr. Trump spoke of the coming of a “golden age” for America, saying, “America has given us an unprecedented and powerful mandate. We have taken back control of the Senate.”

“It’s time to put the divisions of the past four years behind us,” said Mr. Trump, who himself ran a divisive campaign.

Speaking after Mr. Trump’s initial remarks, Vice-President-elect J.D. Vance called the results “the greatest political comeback” in American history.

“He’s turned out to be a good choice,” Mr. Trump said about Mr. Vance, to laughter from his supporters.

“I took a little heat at the beginning, but I knew the brain was a good one, as good as it gets,” Mr. Trump said. His choice of Mr. Vance, a Yale Law School graduate who grew up in a white working-class family, was questioned because of Mr. Vance’s gaffes (a reference to “childless cat ladies” was thought to cost Mr. Trump politically during the campaign).

Mr. Vance’s wife, Usha Vance, whose parents immigrated to the U.S. from India, is set to become the first Indian American Second Lady of the United States.

In Washington, Ms. Harris was scheduled to make a speech at her alma mater Howard University on Wednesday afternoon."""


query = "Why are some criticisms of Trump?"     

docs = findSimilar(query, 5) 

for x in docs:
    print(x.page_content) 
    print(x.metadata.get("url", "Unknown")) 


# results, reasoning = query_with_reasoning(query)

# Display results and reasoning
# for i, result in enumerate(results):
#     print(f"Result {i+1}:")
#     print(f"Metadata: {result.metadata}")
#     print("-" * 50)

# print("\nReasoning:\n", reasoning)