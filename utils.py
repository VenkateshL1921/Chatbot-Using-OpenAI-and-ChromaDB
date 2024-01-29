from langchain_community.vectorstores import Chroma
from sentence_transformers import SentenceTransformer
import chromadb
from dotenv import load_dotenv
import os
import warnings

from langchain_community.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain

warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')
load_dotenv()

# get enviroment variables
persist_directory = os.environ.get('PERSIST_DIRECTORY')
model = os.environ.get("MODEL")
collection = os.environ.get("COLLECTION_NAME")
os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_KEY")
openai_model_name = os.environ.get("OPENAI_MODEL")

# get models
embedding = SentenceTransformer(model)
llm = llm = OpenAI(model_name=openai_model_name)
chain = load_qa_chain(llm, chain_type="stuff")

# load the persisted chroma database 
client = chromadb.PersistentClient(path=persist_directory)
collection = client.get_collection(collection)

def get_results(query):
    input_emb = embedding.encode(query).tolist()
    results = collection.query(query_embeddings=[input_emb],
                                n_results=3)
    return results["documents"][0]

def get_answer(query):
    docs = get_results(query=query)
    answer =  chain.run(input_documents=docs, question=query)
    return answer



