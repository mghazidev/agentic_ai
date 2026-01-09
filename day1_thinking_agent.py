import requests

OLLAMA_URL = "http://localhost:11434/api/generate"

def call_llama(prompt):
    payload = {
        "model": "llama3",
        "prompt": prompt,
        "stream": False
    }
    response = requests.post(OLLAMA_URL, json=payload)
    return response.json()["response"]

goal = input("Enter your goal: ")

agent_state = f"""
You are an autonomous AI agent.

Your goal:
{goal}

Rules:
- Think step by step
- Decide only ONE next action
- Do NOT execute the action
- Explain reasoning clearly
"""

for step in range(5):
    print(f"\n🧠 STEP {step + 1}")
    
    response = call_llama(agent_state)
    print(response)

    agent_state += f"\nPrevious decision:\n{response}\nWhat should I do next?"
