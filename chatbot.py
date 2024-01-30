from langchain_community.chat_models import ChatOpenAI
from langchain.chains import ConversationChain

from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.prompts import (SystemMessagePromptTemplate,HumanMessagePromptTemplate,
                                ChatPromptTemplate,MessagesPlaceholder)

import streamlit as st
from streamlit_chat import message

from utils import get_results, get_conversation_string, query_refiner
from dotenv import load_dotenv
import os

load_dotenv()

# Set the app title
st.set_page_config(page_title='🏏 Talk with the Chatbot about Cricket 🏏')
st.header('🏏 Talk with the Chatbot about Cricket 🏏')

# Session states
if 'responses' not in st.session_state:
    st.session_state['responses'] = ["Hi, Ask me about the cricket!"]

if 'requests' not in st.session_state:
    st.session_state['requests'] = []

if 'buffer_memory' not in st.session_state:
            st.session_state.buffer_memory=ConversationBufferWindowMemory(k=3,return_messages=True)

# LLM
openai_key = os.environ.get("OPENAI_KEY")
llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=openai_key)

# Templates
prompt = """Use the following pieces of information to answer the user's question truthfully.
If you cant find the answer in the text, just say 'I don't know', don't try to make up an answer"""
system_template = SystemMessagePromptTemplate.from_template(template=prompt)

human_template = HumanMessagePromptTemplate.from_template(template="{input}")

prompt_template = ChatPromptTemplate.from_messages([system_template,
                                                     MessagesPlaceholder(variable_name="history"),
                                                       human_template])

# Chain
conversation = ConversationChain(llm=llm, memory=st.session_state.buffer_memory, 
                                 prompt=prompt_template)

# Container
response_container = st.container()
textcontainer = st.container()

with textcontainer:
    query = st.text_input("Query: ", key="input")
    if query:
        with st.spinner("typing..."):
            conversation_string = get_conversation_string()
            refined_query = query_refiner(conversation_string, query)
            st.subheader("Refined Query:")
            st.write(refined_query)
            context = get_results(refined_query) 
            response = conversation.predict(input=f"Context:\n {context} \n\n Query:\n{query}")
        st.session_state.requests.append(query)
        st.session_state.responses.append(response) 

with response_container:
    if st.session_state['responses']:

        for i in range(len(st.session_state['responses'])):
            message(st.session_state['responses'][i],key=str(i))
            if i < len(st.session_state['requests']):
                message(st.session_state["requests"][i], is_user=True,key=str(i)+ '_user')