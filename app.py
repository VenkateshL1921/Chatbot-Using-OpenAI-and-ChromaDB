from fastapi import FastAPI
from pydantic import BaseModel
from utils import get_results, get_answer

app = FastAPI()

class Query(BaseModel):
    question: str
    

@app.get("/")
async def greet(name:str="Cricket"):
   return f"Hello, This is a {name} QA bot"

@app.post("/get-similar-result}")
async def query_result(query:Query):
   result = get_results(query=query.question)
   return result

@app.post("/get-bot-answer")
async def bot_answer(query:Query):
   answer = get_answer(query=query.question)
   return answer
   