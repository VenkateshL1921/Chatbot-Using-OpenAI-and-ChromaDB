# QA-bot-Using-OpenAI-and-ChromaDB
This repository consist of codebase of QA bot for cricket using LLM and vector database

## Architecture

![LLM (1)](https://github.com/VenkateshL1921/Chatbot-Using-OpenAI-and-ChromaDB/assets/108605062/998b1ae7-5e46-4d97-a334-7281723c6732)

## Tech stack
1. Python
2. Langchain
3. ChromaDB
4. OpenAI
5. FastAPI
6. Streamlit

## Result

![Screenshot from 2024-01-29 20-13-47](https://github.com/VenkateshL1921/QA-Bot-Using-OpenAI-and-ChromaDB/assets/108605062/7a140241-9449-4f1b-8f1a-f3138690420b)

## Approach

1. Created data source related to Cricket (i.e text files).
2. Extracted the data using Directory loader and created chunks of data
3. Created embeddings of chunks
4. Inserted the embedding in chromaDB and persisted the database
5. Loaded the persisted db from local disk and tested for similarity search
6. Added OpenAI gpt-3.5-turbo model for answering user questions
7. Created API endpoints using FastAPI
8. Created frontend for QA model using Streamlit and integrated it with FastAPI
 