"""
DeepSeek Client using requests
"""
import os
import requests
import json

class DeepSeekClient:
    """DeepSeek API client using requests"""
    
    def __init__(self):
        self.api_key = os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError("DEEPSEEK_API_KEY not found")
        
        self.base_url = "https://api.deepseek.com/v1"
        self.model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
        
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
    
    def chat_completion(self, messages, tools=None, tool_choice="auto", max_tokens=2000):
        """Make a chat completion request"""
        
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.7
        }
        
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice
        
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            
            result = response.json()
            
            return result
        
        except requests.exceptions.RequestException as e:
            print(f" Request error: {str(e)}")
            return {"error": str(e)}
        except Exception as e:
            print(f" Unexpected error: {str(e)}")
            return {"error": str(e)}
