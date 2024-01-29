from langchain_community.vectorstores import Chroma
from sentence_transformers import SentenceTransformer
import chromadb


embedding = SentenceTransformer('paraphrase-MiniLM-L3-v2')

# get persisted database
persist_directory = "/home/user/Desktop/Main Projects/Chatbot-Using-OpenAI-and-ChromaDB/db"
#vectordb = Chroma(persist_directory=persist_directory,
#                 embedding_function=embedding)
#vectordb.get()

# access collection
client = chromadb.PersistentClient(path=persist_directory)
collection = client.get_collection("test")

# test with a query
query = "what is rivalry between india and australia?"
input_em = embedding.encode(query).tolist()
results = collection.query(
    query_embeddings=[input_em],
    n_results=1)

print(results)


