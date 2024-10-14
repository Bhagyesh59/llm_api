
from app.models import Summary
from app.models import ChatCompletionRequest
from fastapi import APIRouter
# from routers import ChatCompletionRequest, Message
from app.agents import summary
import uuid
from openai import OpenAI
import os

Client = OpenAI(base_url = "https://api.groq.com/openai/v1",api_key=os.getenv("GROQ_API_KEY"))

router = APIRouter()

from pymongo import MongoClient


client = MongoClient("mongodb+srv://LLM_User:User_llm01@clusterllm.saneb.mongodb.net/?retryWrites=true&w=majority&appName=ClusterLLM")
db = client["chat_database"]
collection = db["chat_history"]


@router.get("/health")
async def healthCheck():
    return {"message": "success"}


@router.post("/summary", response_model=Summary)
async def summary_agent(request: ChatCompletionRequest):
    data=request.data
    return summary.summary_responce(data)
    
    
@router.post("/getuid")
async def create_chat_session(user_input: str):
    # Create a new chat session
    session_id = uuid.uuid4()
    collection.insert_one({"session_id": session_id, "chat_history": []})
    print(session_id)
    # Respond with the session ID
    return {"session_id": session_id}

# @router.post("/chat/{session_id}")
# async def send_message(session_id: str, user_input: str):
#     # Retrieve the chat session
#     session = collection.find_one({"session_id": session_id})

#     # Append the user input to the chat history
#     session["chat_history"].append({"user_input": user_input})

#     # Call the OpenAI API to generate a response
#     response = Client.Completion.create(
#         prompt=user_input,
#         max_tokens=1024,
#         temperature=0.5
#     )

#     # Append the response to the chat history
#     session["chat_history"].append({"response": response.choices[0].text})

#     # Update the chat session
#     collection.update_one({"session_id": session_id}, {"$set": session})

#     # Respond with the updated chat history
#     return {"chat_history": session["chat_history"]}