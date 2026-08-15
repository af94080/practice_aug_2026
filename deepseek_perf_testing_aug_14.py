import os
import time 
from dotenv import load_dotenv
load_dotenv()

MODEL = "deepseek-ai/DeepSeek-V4-Flash-0731"

HF_API_KEY = os.environ["HF_API_KEY"]

from huggingface_hub import InferenceClient 

client = InferenceClient(
	provider="together",
	api_key=HF_API_KEY
	)

def get_response(user_prompt):
	start_time = time.perf_counter()
	response = client.chat.completions.create(
	model=MODEL,
	messages=[
		{
		"role": "system",
		"content": "Answer as concisely as possible. For code requests, return only the code with minimal explanation."
		},
        {
		"role": "user",
		"content": user_prompt
        }
    ],
	max_tokens=1000
	)
	run_time = time.perf_counter() - start_time

	return response, run_time  

def main():

	questions = [
	    "What is the capital of France?",
	    "Explain why the sky appears blue in two sentences.",
	    "Give me three advantages of using Snowflake.",
	    "Write a Python function that reverses a string.",
	    "What is the difference between an API and an SDK?"
	]

	for question in questions:

		user_prompt = question
		response, runtime = get_response(user_prompt)
		print(f"ANSWER: {response.choices[0].message.content}")
		print(f"Runtime: {runtime:.2f} seconds")
		print(f"Prompt tokens: {response.usage.prompt_tokens}")
		print(f"Completion tokens: {response.usage.completion_tokens}")

		print(f"Reasoning tokens: {response.usage.completion_tokens_details['reasoning_tokens']}")
		print(f"Total tokens: {response.usage.total_tokens}") 

main()
