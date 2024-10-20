
import re
from typing import Any, List, Optional, Optional
from app.models import Message
from fastapi import  HTTPException
from app import database_handler



    

def extract_yaml(output_text):
    """
    Extract and return the YAML content from a string with additional text.

    :param output_text: The text containing the YAML output enclosed within ```
    :return: The YAML content as a string
    """
    # Use a regular expression to extract the YAML content between ``` markers
    yaml_content = re.search(r"```(.*?)```", output_text, re.DOTALL)
    
    if yaml_content:
        return yaml_content.group(1).strip()  # Extract the YAML and remove extra whitespace
    else:
        print("No YAML content found.")
        return None
    
async def update_chat_history(id_: str, new_data: List[Message]):
    # Retrieve chat history from MongoDB
    chat_history = database_handler.collection.find_one({"id_": id_})
    
    if not chat_history:
        return {"no chat history found"}

    else:
        # Extract previous messages and compare with new data
        previous_messages = chat_history.get("data", [])
        new_messages = []

        # Check for new messages
        for message in new_data:
            if message.dict() not in previous_messages:
                new_messages.append(message.dict())
        
        if new_messages:
            # Update chat history with new messages
            await database_handler.collection.update_one(
                {"id_": id_},
                {"$push": {"data": {"$each": new_messages}}}
            )
            previous_messages.extend(new_messages)
        
        return previous_messages

# Function to retrieve context from the vector store
async def retrieve_context_from_vector_store(message: str):
    # Example logic to query vector store based on message
    # You will need to replace this with actual vector similarity search logic
    context=  database_handler.test_vector_context()
    # context = await database_handler.vector_store.find_one({"$text": {"$search": message}})
    
    if context:
        return context.get("context")
    else:
        return "No relevant context found."

# Placeholder function to send data to an LLM and get a response
async def send_to_llm(chat_history, context):
    # Simulate LLM response generation using both chat history and retrieved context
    llm_response = f"Generated response based on {len(chat_history)} messages and context: {context}."
    return llm_response


# Function to retrieve the chat history and compare with incoming data
async def compare_chat_history(id_: str, data: List[Message],chathistory):
    # Retrieve chat history from MongoDB asynchronously
    chat_history =  chathistory
    
    if not chat_history:
        return [], []  # Return empty lists if no chat history is found
    
    previous_messages = chat_history
    if not previous_messages:
        previous_messages = []
    
    new_messages = []
    for message in data:
        if message.dict() not in previous_messages:
            new_messages.append(message.dict())
    
    return previous_messages, new_messages


def get_next_question(questions: list, current_question_id: int, answers: dict):
    # Check if current_question_id is out of range (conversation is complete)
    if current_question_id >= len(questions):
        return {
            "status": "complete",
            "message": "All questions have been asked.",
            "answers": answers
        }

    # Get the next question based on the current ID
    next_question = questions[current_question_id]
    
    return {
        "next_question_id": current_question_id + 1 if current_question_id else current_question_id,
        "next_question": next_question,
        "answers_so_far": answers
    }
    
def parse_question_answer(llm_response: str) -> dict:
    """
    Parses a question and answer from a given LLM response.
    
    Args:
        llm_response (str): The response text from the LLM containing a question and answer.
    
    Returns:
        dict: A dictionary containing the parsed 'question' and 'answer'.
    """
    # Use regex to find the question and answer in the response
    question_pattern = r'Question:\s*(.*?)(?:\n|$)'
    answer_pattern = r'Answer:\s*(.*?)(?:\n|$)'

    question_match = re.search(question_pattern, llm_response)
    answer_match = re.search(answer_pattern, llm_response)

    question = question_match.group(1).strip() if question_match else None
    answer = answer_match.group(1).strip() if answer_match else None

    return question, answer

import ast
import json

def parse_llm_output(output):
    try:
        # Try parsing using json.loads (assumes valid JSON format with double quotes)
        return json.loads(output)
    except json.JSONDecodeError:
        try:
            # If JSON parsing fails, try using ast.literal_eval for single quotes
            return ast.literal_eval(output)
        except (ValueError, SyntaxError) as e:
            # Return a message if parsing fails
            return f"Parsing failed: {e}"

