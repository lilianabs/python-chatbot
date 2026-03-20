import os
import tiktoken
from openai import OpenAI

API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = "gpt-4.1-nano-2025-04-14"
TEMPERATURE = 0.7
MAX_TOKENS = 100
SYSTEM_PROMT = "You are a fed up and sassy assistant who hates answering questions."
MESSAGES = [{"role": "system", "content": SYSTEM_PROMT}]
client = OpenAI(api_key=API_KEY)

def chat(user_input):
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMT},
            {"role": "user", "content": user_input}
        ],
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS
    )

    response_message = response.choices[0].message.content
    MESSAGES.append({"role": "user", "content": user_input})
    MESSAGES.append({"role": "assistant", "content": response_message})
    
    return response_message

while True:
    user_input = input("You: ")
    if user_input.strip().lower() in {"exit", "quit"}:
        break
    answer = chat(user_input)
    print("Assistant:", answer)
    
print("Chat history: ", MESSAGES)

