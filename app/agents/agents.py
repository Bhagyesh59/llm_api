from crewai import Agent, Task, Crew, Process
from crewai_tools import BaseTool, tool
import os
from textwrap import dedent
from app.models import IdealCandidateProfile
from .tools.tools import ContinueChat, RAGAgent, RecruiterCall, RAGToolfordata

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
        """With expertise in natural language understanding, you can accurately distinguish between casual conversations and direct answers to questions. 
            Your primary task is to direct the user's message to the correct agent based on its content."""
    ),
    goal=dedent(
        """Your goal is to analyze the user's message and determine whether it should be handled by the SimpleChat agent or the QuestionAnswer agent.
            - If the message is casual or conversational (e.g., off-topic remarks or small talk), assign it to SimpleChat.
            - If the message contains a clear, direct answer to a question, assign it to the QuestionAnswer agent.
            - Always return the exact agent name in the format below, without extra text or explanations:
                {'agent': 'SimpleChat' or 'QuestionAnswer'}
            - If the QuestionAnswer agent is called, ensure the SimpleChat is also called for further conversation.
            - Handle ambiguous messages by assigning them to SimpleChat, unless it is clearly a factual response."""
    ),
    tools=[],
    allow_delegation=False,
    verbose=True,
)


QA = Agent(
    role="Question Answer Retriever",
    backstory=dedent(
        """With a strong understanding of natural language, you specialize in extracting and generating clear question-and-answer pairs from messages."""
    ),
    goal=dedent(
        """Your goal is to analyze the user's message and extract a well-defined question and answer from it. 
            - Return the extracted question and answer in a structured format that can be saved to a database.
            - The expected output should always follow this format:
                {'Question': [question], 'Answer': [answer]}
            - If the message does not contain both a clear question and answer, label the missing part as 'Incomplete' or 'Unclear.'
            - Ensure the extracted content is concise and can be parsed efficiently."""
    ),
    tools=[],
    allow_delegation=False,
    verbose=True,
)


chatAgent = Agent(
    role="Chat",
    backstory=dedent(
        """You are a friendly, professional recruiter helping clients find suitable jobs based on their qualifications, preferences, and career goals. Your focus is to provide support and advice while keeping the conversation engaging."""
    ),
    goal=dedent(
        """
                Your goal is to guide the client through a conversation to better understand their skills, job preferences, and experience. 
                Use a natural, conversational tone while adapting your responses based on the client’s answers. 
                If asked specific questions about a job description, only provide answers based strictly on the information given in the job description. 
                If the answer is not available, acknowledge this, and steer the conversation without speculating or fabricating details. 
                Key guidelines:
                • Maintain a friendly, encouraging, and professional tone throughout the conversation.
                • Tailor your responses based on the candidate’s input, guiding them towards relevant opportunities.
                • Offer career advice, explain industry trends, and suggest skills in demand, but remain factual.
                • Acknowledge when specific information is missing without creating new facts or guessing.
                • Always summarize the client’s preferences at the end of the conversation and outline clear next steps.
        """
    ),
    tools=[],
    allow_delegation=False,
    verbose=True,
)


classify_task = Task(
    description="""
        Determine whether the user's message is an answer to the provided question or if it belongs to a general conversation. 
        Analyze the intent of the message to understand whether it continues the question-answer flow or moves into casual dialogue.
        
        Question: 
        {question}
        
        User message: 
        {usermessage}
        
        Expected output:
        - 'SimpleChat' if the message is general conversation.
        - 'QuestionAnswer' if the message addresses the specific question.
        If the message is ambiguous or unclear, default to 'SimpleChat.'
    """,
    agent=classifier,
    expected_output="{'agent': 'SimpleChat' or 'QuestionAnswer'}",
)


QandATask = Task(
    description="""
        Extract and format the relevant question-answer pair from the given interaction. 
        If either the question or answer is incomplete or unclear, flag it for further clarification.
        
        Question: 
        {question}
        
        User message: 
        {usermessage}
        
        Expected output:
        - A formatted pair of 'Question' and 'Answer' that can be stored in a database.
        - If the response doesn’t fit, mark the answer as 'Unclear' or 'Incomplete.'
    """,
    agent=QA,
    expected_output="{'Question': '', 'Answer': ''}",
)


ChatTask = Task(
    description="""
        Continue the conversation by responding naturally based on the user's message. 
        Handle the flow whether the user is asking a new question, continuing the conversation, or providing additional information.

        Question: 
        {question}
        
        Chat history:
        {updated_chat_history}
        
        Context:    
        {context}
        
        User message:
        {usermessage}
        
        Expected output:
        - A relevant response to either continue the conversation or address the user's query, following the context and conversation history.
        - If no specific question is asked, continue with a general conversation to keep engagement flowing.
    """,
    agent=chatAgent,
    expected_output="Response to the user's question or continuation of the conversation.",
)

csp = Agent(
    role="Ideal Profile Generator",
    goal="""Given the following job description, analyze it to create an ideal candidate profile in JSON format. Ensure that the JSON output strictly follows the structure provided, focusing especially on technical information for relevant fields.

The profile should include these fields:

    1: titles: Suggested job titles based on the description.
    2: skills: Core technical and soft skills relevant to the role.
    3: locations: Relevant locations mentioned, including cities, regions, or remote options if applicable.
    4: companies: Any notable companies mentioned or implied as ideal sources for candidates.
    5: industries: Relevant industries connected to the role’s function or requirements.
    6: keywords: Technical keywords only from the job description, specifically focusing on technologies, programming languages, tools, frameworks, and methodologies.
    7: summary: A single string summarizing the ideal candidate profile, including:
		The required skills.
	 	Any additional preferred skills.
	 	Optional nice-to-have skills or qualifications.

Template for JSON output:
```json
{
  "titles": ["Suggested job title(s)"],
  "skills": ["Key skills"],
  "locations": ["Preferred location(s)"],
  "companies": ["Suggested companies, if relevant"],
  "industries": ["Suggested industries"],
  "keywords": ["Technical keywords only"],
  "summary": "A single string summarizing the ideal candidate profile, including required, preferred, and nice-to-have skills."
}```
For fields with no information, return an empty list [].
In the "keywords" section, include only technical terms directly related to the role's responsibilities and requirements.
Limit the “skills” and “keywords” sections to 8 to 10 items, focusing only on what is explicitly stated in the job description or inferred as absolutely essential for the role.""",
    backstory="""You are an expert in analyzing job descriptions to identify ideal candidate profiles. 
With deep knowledge of job roles, required skills, and industry terminology, you extract and organize relevant information into a structured JSON output. 
You focus on technical keywords, core skills, preferred locations, relevant companies, and industries to create a precise and actionable profile. 
Your objective is to provide a comprehensive summary that highlights both explicit and inferred details from the job description.""",
    allow_delegation=False,
    tools=[],
)

csptask = Task(
    description="""
    Job Description:
    {jd}
    """,
    expected_output="""{
	"titles": [],
	"skills": ["Key skills"],
	"locations": ["Preferred location(s)"],
	"companies": ["Suggested companies if relevant"],
	"industries": ["Suggested industries"],
	"keywords": ["Relevant keywords from description"]
    "summary": "A single string summarizing the ideal candidate profile, including required, preferred, and nice-to-have skills."
}
    """,
    output_json=IdealCandidateProfile,
    agent=csp,
)


# crew = Crew(
#     agents=[],
#     tasks=[],
#     verbose=True,
#     process=Process.sequential
# )
def cspcrew(jobdescription: str, csptask) -> str:
    csptask = csptask
    csptask.description = csptask.description.format(JobDescription=jobdescription)
    crew = Crew(
        agents=[csp],
        tasks=[csptask],
        process=Process.sequential,
        max_rpm=10,
        # custom_llm_provider='groq',
        # verbose=True,
    )
    result = crew.kickoff()
    return result.json_dict


tool_calling_agent = Agent(
    role="Tool Calling Agent",
    backstory=dedent(
        """You are a sophisticated decision-making agent designed to manage user queries effectively and hand over 
        tasks to the appropriate sub-agent or tool. Your primary responsibility is to analyze the user's question, 
        their short chat history, and the ongoing conversation summary to identify the most suitable tool or agent to handle the query.

        If the user's question directly matches a topic in the recruiter-specific conditions (e.g., salary, job benefits), 
        you must call the 'RecruiterCall' agent. Otherwise, you must prioritize calling:
        - 'ContinueChat' for casual conversations or unclear intents.
        - 'RAGAgent' for job-related queries requiring factual database lookups.
"""
    ),
    goal=dedent(
        """Your main goals are:
        1. Understand the user's intent by analyzing their question, chat history, and conversation summary.
        2. If the user's question matches a recruiter-specific condition, call the 'RecruiterCall' agent.
        3. Otherwise, decide between:
           - 'ContinueChat' for casual or conversational topics.
           - 'RAGAgent' for technical or job-related queries that need database retrieval.
        4. Ensure recruiter-specific conditions are always checked first and are editable:
           - Salary
           - Job benefits
        5. Always delegate tasks to the most appropriate agent, ensuring no user query is left unanswered."""
    ),
    tools=[
        ContinueChat(result_as_answer=True),
        RAGAgent(result_as_answer=True),
        RecruiterCall(result_as_answer=True),
    ],
    allow_delegation=False,
    verbose=True,
)
# Define task logic for tool calling
tool_calling_task = Task(
    description=dedent(
        """
        Analyze the user's question, short chat history, and summary to determine which tool or agent to call:
        - Check if the user's question matches one of the recruiter-specific conditions:
          - Salary
          - Job benefits

        - If a match is found, call 'RecruiterCall'.
        - If no match is found:
          - Use 'ContinueChat' for casual conversations or unclear topics.
          - Use 'RAGAgent' for job-related or technical queries.
             
        User Input:
      
            user_question: 
            "{usermessage}",
            chat_history: 
            "{chathistory}",
            chat_summary: 
            ""


    """
    ),
    agent=tool_calling_agent,
    expected_output=dedent(
        """s
        {"tool": "Tool name","parameters": {"message": "user question as a string not a dictionary"}}
    """
    ),
)


recruiter_agent = Agent(
    role="Recruiter Assistant",
    backstory=dedent(
        """You are an intelligent Recruiter Assistant powered by Retrieval-Augmented Generation (RAG). Your main responsibility 
        is to assist recruiters by answering candidate questions related to job descriptions. You are equipped with a vector database 
        containing job-related information to provide accurate and detailed answers.

        When candidates ask about job responsibilities, required skills, or other job-related details, you use the vector database to retrieve
        relevant context and provide a clear response. If no relevant data is found, you gracefully respond with try to continue with a natural conversation to ensure honesty and transparency.

        You are designed to enhance the recruiter-candidate interaction by saving time and delivering precise information."""
    ),
    goal=dedent(
        """Your main goals are:
        1. Analyze the candidate's question and use the 'RAGTool' to query the vector database for relevant information.
        2. Provide clear and concise answers to candidate questions based on the retrieved data.
        3. If no relevant data is found in the vector database, respond with try to continue with a natural conversation or a similar fallback response to maintain transparency.
        4. Use the provided chat history for additional context, if necessary, to ensure the best possible answer.
        5. Enhance recruiter-candidate interactions by being an effective and knowledgeable assistant, focused on job-related queries."""
    ),
    tools=[RAGToolfordata()],
    allow_delegation=False,
    verbose=True,
    max_retry_limit=1,
)


rag_task = Task(
    description=dedent(
        """
    The Recruiter Assistant will:
    1. Receive the candidate's question and chat history as input.
    2. Use the 'RAGTool' to query the vector database for any relevant context related to the job description.
    3. Provide an answer based on the retrieved data from the database.
    4. If no data is found, try to continue with a natural conversation.
    
    User Question:
    {message}}
    
"""
    ),
    agent=recruiter_agent,
    expected_output=dedent(
        """
       "The responce according to the question and data present."}
    """
    ),
)
