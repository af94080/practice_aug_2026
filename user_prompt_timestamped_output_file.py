import os 
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

system_prompt = "Act as an expert financial analyst"
user_prompt = "Offer insights into retirement plan for individuals approaching retirement age"

def get_response(system_prompt, user_prompt):
	messages = [
		{"role": "system", "content": system_prompt},
		{"role": "user", "content": user_prompt}
	]

	response = client.chat.completions.create(
		model="gpt-4o-mini",
		messages = messages,
		temperature = 0
		)
	return response.choices[0].message.content

def create_output_filename():
    script_name = os.path.splitext(os.path.basename(__file__))[0]
    timestamp = datetime.now().strftime("%m%d%y_%H%M")
    return f"{script_name}_{timestamp}_out.txt"

def write_output_file(output_file, result):
	with open (output_file, "w", encoding="utf-8") as f:
		f.write(result)
	print(f"output written to {output_file}")

if __name__ == "__main__":
    output_file = create_output_filename()
    result = get_response(system_prompt, user_prompt)
    write_output_file(output_file, result)
    
