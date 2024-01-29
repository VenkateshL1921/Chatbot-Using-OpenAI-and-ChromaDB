import streamlit as st
import requests
from dotenv import load_dotenv
import os
import json

load_dotenv()
url = os.environ.get("URL")

def run(url):
    st.set_page_config(page_title='🏏 Ask the QA about Cricket 🏏')
    st.title('🏏 Ask the QA about Cricket 🏏')
    # Query text
    query_text = st.text_input('Enter your question: ',
                            placeholder = 'Ask your question to QA bot.')
    
    data = {"question": query_text}

    with st.spinner('Calculating...'):
        output = requests.post(url, json=data)
        st.success(output.text)

if __name__ == '__main__':
    run(url=url)

