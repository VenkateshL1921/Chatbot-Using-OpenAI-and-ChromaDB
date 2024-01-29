from langchain_community.vectorstores import Chroma
from sentence_transformers import SentenceTransformer
import chromadb
from dotenv import load_dotenv
import os
import warnings

warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')

load_dotenv()

persist_directory = os.environ.get('PERSIST_DIRECTORY')
model = os.environ.get("MODEL")
collection = os.environ.get("COLLECTION_NAME")

embedding = SentenceTransformer(model)

client = chromadb.PersistentClient(path=persist_directory)
collection = client.get_collection(collection)

def get_results(query):
    input_emb = embedding.encode(query).tolist()
    results = collection.query(query_embeddings=[input_emb],
                                n_results=3)
    return results["documents"][0]

query = "what is rivalry between india and australia?"
print(get_results(query=query))
