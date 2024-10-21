from crewai import Agent, Task, Crew, Process
from crewai_tools import BaseTool, tool
import os
from textwrap import dedent

os.environ["OPENAI_API_BASE"] = "https://api.groq.com/openai/v1"
os.environ["OPENAI_MODEL_NAME"] = "llama3-70b-8192"
os.environ["OPENAI_API_KEY"] = (
    "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"
)

# question='What is your desired salary range?'
# usermessage='i am instread in this job'
# job_description='''Job Title: Senior Software Engineer - Full Stack Developer
# Location: Bangalore, India

# Company: TechInnovate Solutions Pvt. Ltd.

# Job Description:
# TechInnovate Solutions, a leading software development company in Bangalore, is looking for a Senior Full Stack Developer to join our dynamic team. The candidate will be responsible for developing scalable web applications, contributing to architecture design, and mentoring junior developers.

# Qualifications:

# 	•	Bachelor’s or Master’s degree in Computer Science, Engineering, or a related field.
# 	•	5+ years of full-stack development experience.
# 	•	Proven track record of leading software projects and mentoring teams.'''

classifier = Agent(
    role="Message Analyzer",
    backstory=dedent(
        """ With expertise in natural language understanding, you can distinguish between casual conversations and specific answer to a questions. 
                    Your job is to direct the message to the correct agent based on its content."""
    ),
    goal=dedent(
        """Analyze the user message to determine whether it should be handled by the SimpleChat or the QuestionAnswer Agent.
                    - If the message is conversational and not an answer to the question, assign it to the SimpleChat.
                    - If the message is a direct question, assign it to the QuestionAnswer Agent to be saved to a database.
                    - do not write any additional verbose text just the agent name.
                    - if the Question-Answer Agent is called then call the normal SimpleChat.
                    - expected output
                        {'agent':''}
                    - the exact name of the agents are 
                        SimpleChat
                        QuestionAnswer"""
    ),
    tools=[],
    allow_delegation=False,
    verbose=True,
)

# manager agent
QA = Agent(
    role="Question Answer Retriver",
    backstory=dedent(
        """ With expertise in natural language understanding, you are able to retrive and generat a question answer pair."""
    ),
    goal=dedent(
        """Analyze the incoming message and retrive Question and answer from it to later be saved to database.
                To easily parse the output return the question and answer in the following format.

                {'Question': [question]
                'Answer': [answer]}"""
    ),
    tools=[],
    allow_delegation=False,
    verbose=True,
)

chatAgent = Agent(
    role="Chat",
    backstory=dedent(
        """You are a friendly and professional recruiter helping clients find suitable jobs based on their qualifications, preferences, and career goals. """
    ),
    goal=dedent(
        """
                Your goal is to guide the client through a conversation to better understand their skills, job preferences, and experience. 
                Follow a predefined question flow but remain conversational and natural. Adjust your responses based on the client’s answers, offering helpful information and advice when needed.
                You will also assist clients by answering questions about specific job descriptions, but you should only provide answers based strictly on the information in the job description provided. 
                If the answer is not found, continue the conversation without providing additional information. Do not reveal that you are an AI.
                you will be provided with question which you have to naturally insert in the conversation to the candidate
                Guidelines:

                •	Maintain a supportive and encouraging tone throughout the conversation.
                •	Adapt your responses based on the candidate’s answers.
                •	Provide relevant advice, such as explaining job trends, skills in demand, or locations with good opportunities.
                •	If the information is not available in the context, say that it is not specified without making up any facts.
                •	Do not mention or imply that you are an AI or automated system.
                •	After gathering all necessary information, summarize their preferences and outline the next steps in the process."""
    ),
    tools=[],
    allow_delegation=False,
    verbose=True,
)


classify_task = Task(
    description="""Analyze if the given message is an answer to the given question or just a simple converstion
        Determine whether the user's message is a response to the given question or part of a general conversation.
        Carefully analyze both the question and the user's message, then decide if the conversation should be directed to the 'SimpleChat' agent for casual dialogue, or the 'QuestionAnswer' agent for factual responses.

        Question:
        {question}
        
        User message:
        {usermessage}
        """,
    agent=classifier,
    expected_output="{'agent': 'SimpleChat' or 'QuestionAnswer'}",
)


QandATask = Task(
    description="""Analyze if the given message and return a formated pair to be parsed and saved to database.
        Question:
        {question}
        user message:
        {usermessage}
        """,
    agent=QA,
    expected_output="{'Question':'', 'Answer':''}",
)


ChatTask = Task(
    description="""
        Question:
        {question}
        chat history:
        {updated_chat_history}
        context:    
        {context}
        user message:
        {usermessage}
        """,
    agent=chatAgent,
    expected_output=" responce to the user question or a simple converstion",
)

# crew = Crew(
#     agents=[],
#     tasks=[],
#     verbose=True,
#     process=Process.sequential
# )
