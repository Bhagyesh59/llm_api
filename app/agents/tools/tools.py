from typing import List, Type, Optional, Any
from crewai_tools import BaseTool
from pydantic import BaseModel
from crewai import Agent, Task, Crew, Process, LLM
from textwrap import dedent
from app.agents import agents


llm = LLM(
    model="groq/llama3-70b-8192",
    api_key="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",
    base_url="https://api.groq.com/openai/v1",
)


# Define the RAGTool
class RAGToolfordataInput(BaseModel):
    user_questions: str = "The candidate's question to query the vector database."


class RAGToolfordata(BaseTool):
    name: str = "RAGToolfordata"
    description: str = (
        "Queries the vector database of job descriptions to retrieve context related to the user question."
    )
    args_schema: Type[BaseModel] = RAGToolfordataInput

    def _run(self, user_questions: str) -> Optional[str]:
        """
        Simulates querying a vector database. Replace this logic with actual database querying.
        """
        # Mock database
        vector_database = {
            "what is the job title?": "senior software developer.",
            "What are the required skills for a data scientist?": "The required skills include Python, machine learning, and data visualization.",
        }

        # # Search for context in the database
        # response = vector_database.get(query, None)
        # if response:
        #     return response
        # else:
        #     return None  # No relevant data found in the database
        return {"tool": "RAGToolfordata", "parameters": {"message": user_questions}}


# this is the continues chat for unrelated or casual topics
class ContinueChatInput(BaseModel):
    question: str = "The user's casual or conversational question."


class ContinueChat(BaseTool):
    name: str = "ContinueChat"
    description: str = "Continues chatting with the user on casual topics."
    args_schema: Type[BaseModel] = ContinueChatInput

    def _run(self, question: str) -> dict:
        return {"tool": "ContinueChat", "parameters": {"message": question}}


# this is the Rag agent for job related topics
class RAGAgentInput(BaseModel):
    question: str = "The user's job-related question."


class RAGAgent(BaseTool):
    name: str = "RAGAgent"
    description: str = "Handles job-related queries by using a RAG database."
    args_schema: Type[BaseModel] = RAGAgentInput

    def _run(self, question: str) -> dict:
        # responce = agents.recruiter_agent.execute_task(agents.rag_task)
        return {"tool": "ContinueChat", "parameters": {"message": question}}


# this is the agent for unexpected question or conditional question to just notify the requrater for assistence
class RecruiterCallInput(BaseModel):
    message: str = "The user's question related to salary or specific recruiter tasks."


class RecruiterCall(BaseTool):
    name: str = "RecruiterCall"
    description: str = (
        "Handles messages related to salary, location, or specific recruiter-related questions."
    )
    args_schema: Type[BaseModel] = RecruiterCallInput

    def _run(self, message: str) -> dict:
        return {"tool": "RecruiterCall", "parameters": {"message": message}}
