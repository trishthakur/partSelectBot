"""
evaluation.py
Evaluate the Orchestrator on a given eval set using DeepSeek as a judge.
"""

import os
import json
from orchestrator import Orchestrator
from dotenv import load_dotenv
import requests

# Load .env variables
load_dotenv()  

# === CONFIG ===
EVAL_FILE = os.path.join("eval", "eval_set.json")
DB_PATH = os.path.join("database")
MAX_TURNS = 3

# Initialize orchestrator
orchestrator = Orchestrator()
print("Orchestrator initialized and ready!")

# === DEEPSEEK CLIENT FOR JUDGING ===
class DeepSeekJudge:
    def __init__(self):
        self.api_key = os.getenv("DEEPSEEK_API_KEY")
        self.model = os.getenv("DEEPSEEK_MODEL2", "chat-large")  # different model for judging
        self.base_url = "https://api.deepseek.com/v1"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

    def judge(self, user_msg, assistant_msg, criteria):
        prompt = f"""
You are an evaluator. Given a user message and assistant response, score them on a scale 0-1 for each criterion.

User message:
{user_msg}

Assistant response:
{assistant_msg}

Criteria:
- must_contain: {criteria.get('must_contain', [])}
- must_include_url: {criteria.get('must_include_url', False)}
- context_retention: {criteria.get('context_retention', True)}

Return ONLY JSON like:
{{
    "must_contain_score": 0.0,
    "must_include_url_score": 0.0,
    "context_retention_score": 0.0
}}
"""
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1000,
            "temperature": 0
        }

        try:
            resp = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=60
            )
            resp.raise_for_status()
            result = resp.json()
            content = result["choices"][0]["message"]["content"]
            return json.loads(content)
        except Exception as e:
            print(f"Error parsing judge response: {e}")
            return {
                "must_contain_score": 0.0,
                "must_include_url_score": 0.0,
                "context_retention_score": 0.0
            }

# Instantiate judge
judge_client = DeepSeekJudge()

# === LOAD EVAL SET ===
with open(EVAL_FILE, "r", encoding="utf-8") as f:
    eval_set = json.load(f)

if not eval_set:
    print("Eval set is empty!")
    exit()

# === RUN EVALUATION ===
all_scores = []

for example in eval_set:
    print("="*60)
    print(f"Eval ID: {example['id']} | Category: {example.get('category')}")
    
    orchestrator.reset_conversation()
    criteria = example.get("criteria", {})
    
    for turn in example.get("turns", []):
        user_msg = turn.get("user")
        if not user_msg:
            continue
        
        response, products, media = orchestrator.process_query(user_msg)
        
        print(f"\nUser: {user_msg}")
        print(f"Assistant: {response}\n")
        
        judge_scores = judge_client.judge(user_msg, response, criteria)
        print(f"Judge Scores: {judge_scores}\n")
        all_scores.append(judge_scores)

# === SUMMARY ===
summary = {
    "must_contain_avg": sum(s["must_contain_score"] for s in all_scores)/len(all_scores),
    "must_include_url_avg": sum(s["must_include_url_score"] for s in all_scores)/len(all_scores),
    "context_retention_avg": sum(s["context_retention_score"] for s in all_scores)/len(all_scores)
}

print("="*60)
print("Evaluation Summary:")
print(json.dumps(summary, indent=2))
