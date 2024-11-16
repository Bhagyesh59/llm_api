from crewai import Agent, Task, Crew, Process
from crewai_tools import BaseTool, tool
import os
from textwrap import dedent
from app.models import IdealCandidateProfile

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
        Continue the conversation by responding naturally based on the user's message and the provided context. 
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
