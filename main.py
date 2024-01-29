from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
import warnings

warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')

embedding = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# get persisted database
persist_directory = "/home/user/Desktop/Main Projects/Final bot/db"
vectordb = Chroma(persist_directory=persist_directory,
                 embedding_function=embedding)
vectordb.get()



# test with a query
query = "What is the rivalry between india and pakistan?"
matching_docs = vectordb.similarity_search(query, k=3)

print(matching_docs)


