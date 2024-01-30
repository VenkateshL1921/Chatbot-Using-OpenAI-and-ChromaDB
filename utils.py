from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from dotenv import load_dotenv
import os
import warnings

from langchain_community.llms import OpenAI

from langchain_community.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain

import openai
import streamlit as st

warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')
load_dotenv()

# get enviroment variables
persist_directory = os.environ.get('PERSIST_DIRECTORY')
model = os.environ.get("MODEL")
os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_KEY")
openai_model_name = os.environ.get("OPENAI_MODEL")
openai.api_key = os.environ.get("OPENAI_KEY")
refiner_model = os.environ.get("CHAT_MODEL")

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

def query_refiner(conversation, query):
    response = openai.Completion.create(
    model=refiner_model,
    prompt=f"Given the following user query and conversation log, formulate a question that would be the most relevant to provide the user with an answer from a knowledge base.\
            \n\nCONVERSATION LOG: \n{conversation}\n\nQuery: {query}\n\nRefined Query:",
    temperature=0.7,
    max_tokens=256,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
    )
    return response['choices'][0]['text']


def get_conversation_string():
    conversation_string = ""
    for i in range(len(st.session_state['responses'])-1):
        
        conversation_string += "Human: "+st.session_state['requests'][i] + "\n"
        conversation_string += "Bot: "+ st.session_state['responses'][i+1] + "\n"
    return conversation_string


