from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from dotenv import load_dotenv
import os
import warnings

from langchain_community.llms import OpenAI

from langchain_community.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain

warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')
load_dotenv()

# get enviroment variables
persist_directory = os.environ.get('PERSIST_DIRECTORY')
model = os.environ.get("MODEL")
os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_KEY")
openai_model_name = os.environ.get("OPENAI_MODEL")

# get models
embedding = SentenceTransformerEmbeddings(model_name=model)
llm = llm = ChatOpenAI(model_name=openai_model_name)
chain = load_qa_chain(llm, chain_type="stuff")

# load the persisted chroma database 
vectordb = Chroma(persist_directory=persist_directory,
                 embedding_function=embedding)
vectordb.get()

def get_results(query):
    similar_docs = vectordb.similarity_search(query, k=3)
    return similar_docs

def get_answer(query):
    docs = get_results(query=query)
    answer =  chain.run(input_documents=docs, question=query)
    return answer


