import os
from dotenv import load_dotenv

load_dotenv()

GROQ_LLM_URL = os.getenv("GROQ_LLM_URL")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
