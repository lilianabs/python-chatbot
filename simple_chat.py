import os
import tiktoken
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = "gpt-4.1-nano-2025-04-14"
TEMPERATURE = 0.7
MAX_TOKENS = 100
SYSTEM_PROMT = "You are a fed up and sassy assistant who hates answering questions."
MESSAGES = [{"role": "system", "content": SYSTEM_PROMT}]
client = OpenAI(api_key=API_KEY)

def get_encoding(model_name):
    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except KeyError:
        print("Model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    return encoding

ENCODING = get_encoding(MODEL_NAME)

def count_tokens(text):
    return len(ENCODING.encode(text))

def total_tokens_used(messages):
    try:
        return sum(count_tokens(msg["content"]) for msg in messages)
    except Exception as e:
        print(f"[token count error]: {e}")
        return 0

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
    print("Current tokens:", total_tokens_used(MESSAGES))
    
print("Chat history: ", MESSAGES)
print("Total tokens: ", total_tokens_used(MESSAGES))

