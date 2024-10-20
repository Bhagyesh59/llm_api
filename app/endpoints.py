import hashlib
from typing import Union
from fastapi import APIRouter
from app.models import Summary, Chat, Name, SendMessageEvent
from .util_function import retrieve_context_from_vector_store,compare_chat_history, get_next_question, parse_question_answer, parse_llm_output
from app.agents import summary, chat
import uuid
from openai import OpenAI
import os
import datetime
from app import database_handler
from app import artifacts
from app.agents import agents

import logging
# Configure the logger
logging.basicConfig(
    filename='chat_logs.log',  # File to log to
    level=logging.INFO,  # Logging level
    format='%(asctime)s - %(levelname)s - %(message)s',  # Log format
    datefmt='%Y-%m-%d %H:%M:%S'  # Date format
)


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
    database_handler.collection.insert_one({"id_": id_, "chat_history": [], "chat_summary":"","question_answered":[],"question_id":0})
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
    document=database_handler.collection.find_one({"id_": id_})
    chathistory=document['chat_history']
    # Retrieve and compare chat history
    previous_messages, new_messages = await compare_chat_history(id_, data, chathistory)
    print(new_messages)
    if new_messages:
        
        updated_chat_history = previous_messages + new_messages
    else:
        
        updated_chat_history = previous_messages
    
    # Get the most recent user message
    user_message = data[-1].message
    document["chat_history"].append({'user_input': user_message})
    # Retrieve relevant context from the vector store
    context = await retrieve_context_from_vector_store(user_message)
    answer = ''
    question=get_next_question(artifacts.question_flow,document['question_id'],answer)
    
    classify_task=agents.classify_task
    classify_task.description=classify_task.description.format(question=question['next_question'], usermessage=user_message)
    output=agents.classifier.execute_task(classify_task)
    parsed_output = parse_llm_output(output)

    if parsed_output['agent']=='QuestionAnswer':
        QandAtsak=agents.QandATask
        QandAtsak.description=QandAtsak.description.format(question=question['next_question'],usermessage=user_message)
        qaoutput=agents.classifier.execute_task(agents.QandATask)
        questionandanswer=parse_llm_output(qaoutput)
        next_question=get_next_question(artifacts.question_flow,document['question_id'],answer)

    
        ChatTask=agents.ChatTask
        ChatTask.description=ChatTask.description.format(question=next_question['next_question'],
                                                         updated_chat_history=updated_chat_history,
                                                         usermessage=user_message,
                                                         context=context)
        new_llm_response=agents.chatAgent.execute_task(ChatTask)
        
        document["chat_history"].append({"llm_response":new_llm_response})
        document["question_answered"].append(questionandanswer)
        document["question_id"]+=1
        
        # Update the chat history with the LLM response
        # database_handler.collection.update_one(
        #     {"id_": id_},
        #     {"$set": document}
        # )
        responce={
            "id_": id_,
            "type": etype,
            "platform": platform,
            "status": "success",
            "updated_chat_history": updated_chat_history,
            "llm_response": new_llm_response,
            "timestamp": timestamp,
            "temp":"0"
        }
    
    
    else:
    
        ChatTask=agents.ChatTask
        ChatTask.description=ChatTask.description.format(question=question['next_question'],usermessage=user_message,updated_chat_history=updated_chat_history, context=context)
        new_llm_response=agents.chatAgent.execute_task(ChatTask)
        document["chat_history"].append({"llm_response":new_llm_response})
        # Update the chat history with the LLM response
        # database_handler.collection.update_one(
        #     {"id_": id_},
        #     {"$set": document}
        # )
        responce={
            "id_": id_,
            "type": etype,
            "platform": platform,
            "status": "success",
            "updated_chat_history": updated_chat_history,
            "llm_response": new_llm_response,
            "timestamp": timestamp,
            "temp":"0"
        }
        
    # logging.info(f"Generated Prompt:\n{prompt}")
    # Send updated chat history and context to the LLM for a response
    
    # logging.info(f"LLM Response:\n{llm_response}")
        
    return responce