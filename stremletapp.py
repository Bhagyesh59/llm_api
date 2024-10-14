import streamlit as st
from typing import Generator
from groq import Groq
import json

st.set_page_config(page_icon="💬", layout="wide",
                   page_title="Groq Goes Brrrrrrrr...")

st.subheader("Groq Chat Streamlit App", divider="rainbow", anchor=False)

client = Groq(
    api_key="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",
)

# Define system prompt
system_prompt = """
You are a friendly and professional recruiter chatbot helping clients find suitable jobs based on their qualifications, 
preferences, and career goals. Your goal is to guide the client through a conversation to better understand their skills, 
job preferences, and experience. Follow a question flow but remain conversational and natural. Adjust your responses based 
on the client’s answers, offering helpful information and advice when needed.You will also assist clients by answering questions 
about specific job descriptions, but you should only provide answers based strictly on the information in the job description provided.
Do not reveal that you are an AI.

Begin by asking a welcoming question, then move through the following flow:

    1.	Introduction:
	 	Greet the client and ask how you can assist them today.
	 	If they’re looking for a job, ask about their current or most recent position.
	2.	Qualification & Experience:
		Ask about their qualifications (education level, certifications).
		Inquire about their work experience, including industries and job titles.
	3.	Job Preferences:
		Ask what kind of role they’re seeking.
		Find out about their preferred industry, job title, and responsibilities.
		Ask about preferred location (remote or specific places).
	4.	Skills:
		Ask which skills or technologies they are proficient in (e.g., software, languages).
		Find out if they have any specialized skills related to the job they are looking for.
	5.	Work Environment & Culture:
		Ask about their preferred work environment (team collaboration, remote work, flexible hours).
		Inquire if they value specific company cultures or benefits.
	6.	Salary Expectations:
		Ask about their salary expectations, considering their experience and industry standards.
	7.	Additional Information:
		Ask if they have any additional preferences or information they want to share.

Throughout the conversation, adapt your tone to match the client’s responses. Be supportive, encouraging, and helpful. 
Provide relevant advice when needed, such as explaining certain job trends, skills in demand, or locations with good opportunities.
Do not mention or imply that you are an AI or automated system. 
After gathering all the necessary information, wrap up by summarizing their preferences and next steps. 
 
Help answer questions about the following job description:
Senior Python Developer: 5+ years of experience in Python, Django, and Flask. Familiar with cloud platforms like AWS and Azure. 
Must have strong communication skills and experience working in agile teams. Remote work available.
"""

# Initialize chat history and selected model
if "messages" not in st.session_state:
    st.session_state.messages = []

if "selected_model" not in st.session_state:
    st.session_state.selected_model = None

# Define model details
models = {
    "gemma-7b-it": {"name": "Gemma-7b-it", "tokens": 8192, "developer": "Google"},
    "llama3-70b-8192": {"name": "LLaMA3-70b-8192", "tokens": 8192, "developer": "Meta"},
    "llama3-8b-8192": {"name": "LLaMA3-8b-8192", "tokens": 8192, "developer": "Meta"},
    "mixtral-8x7b-32768": {"name": "Mixtral-8x7b-Instruct-v0.1", "tokens": 32768, "developer": "Mistral"},
}

# Layout for model selection and max_tokens slider
col1, col2 = st.columns(2)

with col1:
    model_option = st.selectbox(
        "Choose a model:",
        options=list(models.keys()),
        format_func=lambda x: models[x]["name"],
        index=2 # Default to mixtral
    )

# Detect model change and clear chat history if model has changed
if st.session_state.selected_model != model_option:
    st.session_state.messages = []
    st.session_state.selected_model = model_option

max_tokens_range = models[model_option]["tokens"]

with col2:
    # Adjust max_tokens slider dynamically based on the selected model
    max_tokens = st.slider(
        "Max Tokens:",
        min_value=512,  # Minimum value to allow some flexibility
        max_value=max_tokens_range,
        # Default value or max allowed if less
        value=min(32768, max_tokens_range),
        step=512,
        help=f"Adjust the maximum number of tokens (words) for the model's response. Max for selected model: {max_tokens_range}"
    )

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    avatar = '🤖' if message["role"] == "assistant" else '👨‍💻'
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])


def generate_chat_responses(chat_completion) -> Generator[str, None, None]:
    """Yield chat response content from the Groq API response."""
    for chunk in chat_completion:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content

if prompt := st.chat_input("Enter your prompt here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user", avatar='👨‍💻'):
        st.markdown(prompt)

    # Fetch response from Groq API
    try:
        # Include the system prompt at the start of the conversation
        chat_completion = client.chat.completions.create(
            model=model_option,
            messages=[
                {"role": "system", "content": system_prompt},  # Add system prompt
                *[
                    {
                        "role": m["role"],
                        "content": m["content"]
                    }
                    for m in st.session_state.messages
                ]
            ],
            max_tokens=max_tokens,
            stream=True
        )

        # Use the generator function to stream responses
        full_response = ""
        with st.chat_message("assistant", avatar="🤖"):
            response_placeholder = st.empty()  # Placeholder for updating response
            for response_chunk in generate_chat_responses(chat_completion):
                full_response += response_chunk
                response_placeholder.markdown(full_response)  # Update response in real-time
                    
    except Exception as e:
        st.error(e, icon="🚨")

    # Append the full response to session_state.messages
    st.session_state.messages.append(
        {"role": "assistant", "content": full_response})