from typing import List, Type, Optional
from crewai_tools import BaseTool
from pydantic import BaseModel
from crewai import Agent, Task, Crew, Process, LLM
from textwrap import dedent
from app.agents.agents import (
    tool_calling_agent,
    tool_calling_task,
    recruiter_agent,
    rag_task,
    ChatTask,
    chatAgent
)


llm = LLM(
    model="groq/llama3-70b-8192",
    api_key="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",  # remove this too after testing, really remove this!!!!!!!!!
    base_url="https://api.groq.com/openai/v1",
)


def OcCrew(usermessage, chathistory):

    tool_calling_task.description = tool_calling_task.description.format(
        usermessage=usermessage,
        chathistory=chathistory,
        sender="",
        receiver="",
    )
    result = tool_calling_agent.execute_task(tool_calling_task)
    # occrew = Crew(
    #     agents=[tool_calling_agent],
    #     tasks=[tool_calling_task],
    #     custom_llm_provider="groq",
    #     custom_llm_options={
    #         "prompt_on_error": True
    #     },  # Enable error prompts, remove this after testing.
    #     verbose=True,
    #     function_calling_llm="groq/llama3-groq-70b-8192-tool-use-preview",
    #     message_analyzer=True,
    #     llm=llm,
    # )

    # result = occrew.kickoff()
    return result


def RagCrew(message):
    rag_task.description = rag_task.description.format(message=message)

    responce = recruiter_agent.execute_task(rag_task)
    return responce

def SimpleChatCrew(candidate_message,question,chathistory ):
    ChatTask.description = ChatTask.description.format(
        usermessage=candidate_message,
        question=question,
        updated_chat_history=chathistory,
        context="",
        sender="",
        receiver="",
    )
    responce = chatAgent.execute_task(ChatTask)
    return responce
    