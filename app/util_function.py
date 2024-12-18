import re
from typing import Any, List, Optional, Optional
from app.models import Message
from fastapi import HTTPException, status
from app import database_handler
import ast
import json
from app.agents import agents
from pydantic import ValidationError
from app.models import IdealCandidateProfile
import asyncio


def extract_yaml(output_text):
    """
    Extract and return the YAML content from a string with additional text.

    :param output_text: The text containing the YAML output enclosed within ```
    :return: The YAML content as a string
    """
    # Use a regular expression to extract the YAML content between ``` markers
    yaml_content = re.search(r"```(.*?)```", output_text, re.DOTALL)

    if yaml_content:
        return yaml_content.group(
            1
        ).strip()  # Extract the YAML and remove extra whitespace
    else:
        print("No YAML content found.")
        return None


async def Retrieve_chat_history(id_: str, new_data: List[Message]):
    # Retrieve chat history from MongoDB
    try:
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
                database_handler.collection.update_one(
                    {"id_": id_}, {"$push": {"data": {"$each": new_messages}}}
                )
                previous_messages.extend(new_messages)

            return previous_messages
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database update failed: {e}")


# Function to retrieve context from the vector store
async def retrieve_context_from_vector_store(message: str):
    # Example logic to query vector store based on message
    # You will need to replace this with actual vector similarity search logic
    context = database_handler.test_vector_context()
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
async def compare_chat_history(id_: str, data: List[Message], chathistory):
    # Retrieve chat history from MongoDB asynchronously
    chat_history = chathistory

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
            "answers": answers,
        }

    # Get the next question based on the current ID
    next_question = questions[current_question_id]

    return {
        "next_question_id": (
            current_question_id + 1 if current_question_id else current_question_id
        ),
        "next_question": next_question,
        "answers_so_far": answers,
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
    question_pattern = r"Question:\s*(.*?)(?:\n|$)"
    answer_pattern = r"Answer:\s*(.*?)(?:\n|$)"

    question_match = re.search(question_pattern, llm_response)
    answer_match = re.search(answer_pattern, llm_response)

    question = question_match.group(1).strip() if question_match else None
    answer = answer_match.group(1).strip() if answer_match else None

    return question, answer


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


def prepare_chat_task(
    question: str, user_message: str, updated_chat_history: str, context: str
) -> str:
    """Helper function to prepare and execute the ChatTask."""
    ChatTask = agents.ChatTask
    ChatTask.description = ChatTask.description.format(
        question=question,
        updated_chat_history=updated_chat_history,
        usermessage=user_message,
        context=context,
    )
    return agents.chatAgent.execute_task(ChatTask)


async def update_new_chat_history(id_: str, updated_data: dict):
    """Helper function to update chat history in the database."""
    try:
        database_handler.collection.update_one({"id_": id_}, {"$set": updated_data})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database update failed: {e}")


def parse_candidate_profile(json_data: str) -> IdealCandidateProfile:
    try:
        # Load JSON data from string
        data = json.loads(json_data)

        # Parse and validate data using the Pydantic model
        candidate_profile = IdealCandidateProfile(**data)

        return candidate_profile
    except json.JSONDecodeError:
        print("Invalid JSON format.")
    except ValidationError as e:
        print("Validation error:", e)


async def Retrieve_candidate_chat_data(chatid):
    with open("candidatedoc.json", "r") as f:
        doc = json.load(f)
    # time delay of 0.5 sec
    await asyncio.sleep(0.5)

    return doc


def get_last_candidate_messages(messages):
    """
    Get all the last consecutive messages sent by the CANDIDATE.

    Args:
        messages (list): List of message dictionaries, each containing a 'sender' key.

    Returns:
        list: A list of the last consecutive messages from the CANDIDATE, in chronological order.
    """
    last_candidate_messages = []
    for message in reversed(messages):  # Iterate over messages in reverse order
        if message["sender"] == "CANDIDATE":
            last_candidate_messages.append(message)

        else:
            break  # Stop as soon as a message not from CANDIDATE is encountered
    return list(
        reversed(last_candidate_messages)
    )  # Reverse to maintain chronological order


def get_latestquestion():
    qs = [
        "Are you currently exploring new opportunities?",
        "Could you tell me more about your current role?",
        "Could you tell me about your experience?",
    ]

    return qs


def parse_llm_tool_response(tool_response):
    """
    Parse the response from an LLM tool and extract the tool name and user message.

    Args:
        tool_response (str): JSON string containing the tool response.

    Returns:
        dict: A dictionary with 'tool' and 'message' keys if successful, otherwise an error message.
    """
    try:
        # Parse the JSON string into a Python dictionary
        response_data = json.loads(tool_response)

        # Extract the tool name and message
        tool_name = response_data.get("tool")
        parameters = response_data.get("parameters", {})
        user_message = parameters.get("message")

        # Validate extracted data
        if not tool_name or not user_message:
            raise ValueError("Missing 'tool' or 'message' in the response.")

        # Return the extracted data
        return {"tool": tool_name, "message": user_message}

    except json.JSONDecodeError:
        return {"error": "Invalid JSON format."}
    except Exception as e:
        return {"error": str(e)}


def extract_messages(data):
    """
    Processes a list of message data and concatenates the 'content' field from each entry
    into a single string, separated by commas.

    Args:
        data (list): List of dictionaries containing message data.

    Returns:
        str: Concatenated messages.
    """
    # Extract the 'content' of each message and join them with commas
    messages = ", ".join(entry.get("content", "") for entry in data)
    return f"messages: {messages}"
def convert_to_sender_message_format(data, sender_filter=None):
    """
    Converts a list of message data into a list of dictionaries with 'sender' and 'message' keys.

    Args:
        data (list): List of dictionaries containing message data.
        sender_filter (str, optional): If specified, filters messages by sender.

    Returns:
        list: List of dictionaries in the format {"sender": <sender>, "message": <content>}.
    """
    # Filter data by sender if a filter is provided
    filtered_data = (
        [entry for entry in data if entry.get("sender") == sender_filter]
        if sender_filter
        else data
    )

    # Transform the data into the required format
    result = [{"sender": entry["sender"], "message": entry["content"]} for entry in filtered_data]
    return result
def getresponcetemp(chat_id,final_answer,timestamp,recresponce=False,errors=None, ):
    status_code=status.HTTP_200_OK
    return {
            "status_code":status_code,
            "message":"Generated llm responce",
            "data": {
                "id_": chat_id,
                "type": "llm_responce",
                "llm_response": final_answer,
                "timestamp": timestamp,
                "recruiterCallResponse": recresponce
            },
            "error": errors,
        }