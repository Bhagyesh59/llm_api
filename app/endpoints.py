import hashlib
from typing import Union
from fastapi import APIRouter
from app.models import Summary, Chat, Name, SendMessageEvent
from .util_function import retrieve_context_from_vector_store, send_to_llm, compare_chat_history, get_next_question, parse_question_answer
from app.agents import summary, chat
import uuid
from openai import OpenAI
import os
import datetime
from app import database_handler
from app import artifacts




router = APIRouter() 

Client = OpenAI(base_url = "https://api.groq.com/openai/v1",api_key=os.getenv("GROQ_API_KEY"))

router = APIRouter()







def deterministic_uuid(content: Union[str, bytes]) -> str:
    """Creates deterministic UUID on hash value of string or byte content.

    Args:
        content: String or byte representation of data.

    Returns:
        UUID of the content.
    """
    if isinstance(content, str):
        content_bytes = content.encode("utf-8")
    elif isinstance(content, bytes):
        content_bytes = content
    else:
        raise ValueError(f"Content type {type(content)} not supported !")

    hash_object = hashlib.sha256(content_bytes)
    hash_hex = hash_object.hexdigest()
    namespace = uuid.UUID("00000000-0000-0000-0000-000000000000")
    content_uuid = str(uuid.uuid5(namespace, hash_hex))

    return content_uuid





# Sync Endpoints
@router.get("/health")
async def healthCheck():
    return {"message": "success"}


@router.post("/summary")
async def summary_agent(request: Summary):
    data=request.data
    return summary.summary_responce(data)
    
@router.post("/chat")
async def chat_agent(request: Chat):
    userinput = request.user_p
    id_ = request.id_
    # Retrieve the chat session
    session = database_handler.collection.find_one({"id_": id_})
    # Append the user input to the chat history
    session["chat_history"].append({"user_input": userinput})


    # Append the response to the chat history
    session["chat_history"].append({"response": chat.chat_responce(userinput) })

    # Update the chat session
    database_handler.collection.update_one({"id_": id_}, {"$set": session})
    responce =  {"chat_history": session["chat_history"]}
    return responce  

# Async Endpoints

   
@router.get("/getuid")
async def create_chat_session(request: Name):
    # Create a new chat session
    id_ =deterministic_uuid(request.id_)
    database_handler.collection.insert_one({"id_": id_, "chat_history": []})
    print(id_)
    # Respond with the session ID
    return {"session_id": id_}

@router.post("/chat/{session_id}")
async def send_message(request: SendMessageEvent):
    id_= request.id_
    etype= request.etype
    platform = request.platform
    timestamp = request.timestamp
    data = request.data
    # print(data)

    # Retrieve and compare chat history
    previous_messages, new_messages = await compare_chat_history(id_, data)
    
    if new_messages:
        # Update the chat history only if there are new messages
        database_handler.collection.update_one(
            {"id_": id_},
            {"$push": {"data": {"$each": new_messages}}}
        )
        updated_chat_history = previous_messages + new_messages
    else:
        updated_chat_history = previous_messages
    
    # Get the most recent user message
    user_message = data[-1].message
    
    # Retrieve relevant context from the vector store
    context = await retrieve_context_from_vector_store(user_message)
    answer = ''
    question=get_next_question(artifacts.question_flow,artifacts.questionid,answer)
    print(question['next_question'])
    prompt=f'''
    Question flow:
    {question['next_question']}
    
    chat history:
    {updated_chat_history}
    
    context:    
    {context}
    
    user message: 
    {user_message}
    '''

    # Send updated chat history and context to the LLM for a response
    llm_response =  chat.chat_responce(prompt)
    
    
    if 'Answer' in llm_response:
        print(llm_response)
        # Extract the answer from the LLM response
        answer = llm_response
        question, answer=parse_question_answer(llm_response)
        next_question=get_next_question(artifacts.question_flow,artifacts.questionid+1,answer)
        prompt=f'''
        Question flow:
        {next_question["next_question"]}
        
        chat history:
        {updated_chat_history}
        
        context:    
        {context}
        
        user message: 
        {user_message}
        '''

       
        # Send updated chat history and context to the LLM for a response
        llm_response =  chat.chat_responce(prompt)
        print(llm_response)
        
        
        database_handler.collection.update_one(
            {"id_": id_},
            {"$push": 
                {"data":
                {"sender": "assistent",
                 "message": llm_response,
                 "timestamp":
                    int(datetime.datetime.now().timestamp())
                    }
                }
            }
        )
        
        
        responce={
            "session_id": id_,
            "type": etype,
            "platform": platform,
            "status": "success",
            "updated_chat_history": updated_chat_history,
            "llm_response": llm_response,
            "timestamp": timestamp,
        }
    else:
        # Update the chat history with the LLM response
        database_handler.collection.update_one(
            {"id_": id_},
            {"$push": 
                {"data":
                {"sender": "assistent",
                 "message": llm_response,
                 "timestamp":
                    int(datetime.datetime.now().timestamp())
                    }
                }
            }
        )
    
        responce={
            "session_id": id_,
            "type": etype,
            "platform": platform,
            "status": "success",
            "updated_chat_history": updated_chat_history,
            "llm_response": llm_response,
            "timestamp": timestamp,
        }
    # Respond with the updated chat history
    return responce