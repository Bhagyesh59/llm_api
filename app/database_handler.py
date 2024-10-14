from fastapi import FastAPI

from pymongo import MongoClient

app = FastAPI()


client = MongoClient("mongodb+srv://LLM_User:User_llm01@clusterllm.saneb.mongodb.net/?retryWrites=true&w=majority&appName=ClusterLLM")
db = client['chat_db']
collection = db['chat_history']
vector_store = db['vector_store'] 



def test_vector_context():
    context = {"context":'''
    Job Title: Senior Software Engineer - Full Stack Developer
Location: Bangalore, India

Company: TechInnovate Solutions Pvt. Ltd.

Job Description:
TechInnovate Solutions, a leading software development company in Bangalore, is looking for a Senior Full Stack Developer to join our dynamic team. The candidate will be responsible for developing scalable web applications, contributing to architecture design, and mentoring junior developers.

Qualifications:

	•	Bachelor’s or Master’s degree in Computer Science, Engineering, or a related field.
	•	5+ years of full-stack development experience.
	•	Proven track record of leading software projects and mentoring teams.
    '''}
    return context