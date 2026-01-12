import requests
import json

OLLAMA_URL = "http://localhost:11434/api/generate"

def call_llama(prompt):
    payload = {
        "model": "llama3",
        "prompt": prompt,
        "stream": False
    }
    response = requests.post(OLLAMA_URL, json=payload)
    return response.json()["response"]


# =========================
# 1️⃣ USER GOAL
# =========================
goal = input("Enter your goal: ")


# =========================
# 2️⃣ AGENT MEMORY (STATE)
# =========================
agent_state = {
    "goal": goal,
    "completed_steps": [],
    "current_step": None,
    "confidence": 0
}


# =========================
# 3️⃣ AGENT SYSTEM PROMPT
# =========================
def build_prompt(state):
    return f"""
You are an autonomous AI agent.

GOAL:
{state["goal"]}

COMPLETED STEPS:
{state["completed_steps"]}

RULES:
- Decide ONLY ONE next action
- Do NOT repeat previous steps
- Output MUST be valid JSON
- Include confidence score (0-100)
- If confidence < 70, refine the action
- If goal is complete, say "DONE"

OUTPUT FORMAT (STRICT JSON):
{{
  "next_action": "...",
  "reasoning": "...",
  "confidence": number,
  "is_goal_complete": true/false
}}
"""


# =========================
# 4️⃣ AGENT LOOP
# =========================
MAX_STEPS = 10

for step in range(MAX_STEPS):
    print(f"\n🧠 STEP {step + 1}")

    prompt = build_prompt(agent_state)
    raw_response = call_llama(prompt)

    try:
        decision = json.loads(raw_response)
    except:
        print("❌ Invalid JSON returned. Stopping.")
        break

    print(json.dumps(decision, indent=2))

    # =========================
    # 5️⃣ TERMINATION CHECK
    # =========================
    if decision["is_goal_complete"]:
        print("\n✅ GOAL ACHIEVED. AGENT STOPPED.")
        break

    # =========================
    # 6️⃣ CONFIDENCE CHECK
    # =========================
    if decision["confidence"] < 70:
        print("⚠️ Low confidence. Refining next step...")
        continue

    # =========================
    # 7️⃣ UPDATE STATE (MEMORY)
    # =========================
    agent_state["completed_steps"].append(decision["next_action"])
    agent_state["current_step"] = decision["next_action"]
    agent_state["confidence"] = decision["confidence"]

else:
    print("\n⛔ MAX STEPS REACHED. AGENT STOPPED.")
