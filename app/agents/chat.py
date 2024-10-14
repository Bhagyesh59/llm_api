from app.agents.prompts import prompt
from app import llmcall
from app.util_function import extract_yaml


def chat_responce(user_p:str):
    system_p=prompt.prompt_02
    # system_p=systemprompt
    user_p=user_p
    messages=[
    {"role": "system", "content": system_p},
    {"role": "user", "content": user_p}
    ]
    return llmcall.not_streaming(messages)