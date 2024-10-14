from app.agents.prompts import prompt
from app import llmcall
from app.util_function import extract_yaml


def summary_responce(data:str):
    system_p=prompt.prompt_01
    user_p=data
    messages=[
    {"role": "system", "content": system_p},
    {"role": "user", "content": "Job Description:\n"+user_p}
    ]
    return extract_yaml(llmcall.not_streaming(messages))


