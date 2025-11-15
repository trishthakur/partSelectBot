
"""
Summary Agent 
"""

from typing import List, Dict
from utils.llm import DeepSeekClient
import json
import re


class SummaryAgent:
    """Conversation context manager with dynamic LLM extraction"""
    
    def __init__(self):
        self.client = DeepSeekClient()
        self.recent_messages = []
        self.max_recent = 6  # last N messages kept fully
        self.summary = ""     # older messages summarized
        self.context = {
            "appliance_type": None,
            "brand": None,
            "current_issue": None
        }
        print("Summary Agent initialized")
    
    def add_message(self, role: str, content: str):
        """Add a message and update context via LLM"""
        self.recent_messages.append({"role": role, "content": content})
        
        # Dynamically update context for user messages
        if role == "user":
            self._update_context_via_llm()
        
        # Summarize and trim if needed
        if len(self.recent_messages) > self.max_recent:
            self._summarize_and_trim()
    
    def get_context_string(self) -> str:
        """Get full context string for agents"""
        parts = []
        if self.summary:
            parts.append(f"Previous conversation summary:\n{self.summary}\n")
        if self.context["appliance_type"]:
            parts.append(f"Appliance: {self.context['appliance_type']}")
        if self.context["brand"]:
            parts.append(f"Brand: {self.context['brand']}")
        if self.context["current_issue"]:
            parts.append(f"Current Issue: {self.context['current_issue']}")
        
        if self.recent_messages:
            parts.append("\nRecent conversation:")
            for msg in self.recent_messages[-4:]:  # last 4 messages
                role_name = "User" if msg["role"] == "user" else "Assistant"
                content = msg["content"][:150]
                parts.append(f"{role_name}: {content}...")
        
        return "\n".join(parts)
    
    def get_context_for_display(self) -> str:
        """Short context for UI"""
        parts = []
        if self.context["appliance_type"]:
            parts.append(f"Appliance: {self.context['appliance_type']}")
        if self.context["brand"]:
            parts.append(f"Brand: {self.context['brand']}")
        if self.context["current_issue"]:
            parts.append(f"Issue: {self.context['current_issue']}")
        return " | ".join(parts) if parts else "No context yet"
    
    def _update_context_via_llm(self):
        """Extract context from conversation using LLM"""
        convo_text = "\n".join([f"{m['role']}: {m['content']}" for m in self.recent_messages[-self.max_recent:]])

        prompt = f"""Analyze this conversation and extract the current context.

Previous context:
- Appliance: {self.context.get('appliance_type') or 'unknown'}
- Brand: {self.context.get('brand') or 'unknown'}
- Issue: {self.context.get('current_issue') or 'unknown'}

Recent conversation:
{convo_text}

Based on the LATEST user message, determine:
1. Is this a NEW topic (completely different appliance/issue) or CONTINUING the previous topic?
2. Extract: appliance type, brand, and current issue

Return ONLY valid JSON:
{{
    "is_new_topic": false,
    "appliance_type": "refrigerator",
    "brand": "Whirlpool",
    "current_issue": "not cooling"
}}

Use "unknown" for any field you cannot determine. Do not include any text outside the JSON."""

        try:
            response = self.client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200
            )
            content = response["choices"][0]["message"]["content"].strip()

            # Extract JSON from response (handle markdown code blocks)
            content = re.sub(r'```json\s*', '', content)
            content = re.sub(r'```\s*', '', content)
            content = content.strip()
            
            # Find JSON object
            match = re.search(r'\{[^{}]*\}', content)
            if not match:
                print(f"No JSON found in response: {content[:100]}")
                return
            
            extracted = json.loads(match.group())
            
            # Update context based on whether it's a new topic
            is_new_topic = extracted.get("is_new_topic", False)
            
            if is_new_topic:
                # Reset context for new topic
                self.context = {
                    "appliance_type": None,
                    "brand": None,
                    "current_issue": None
                }
            
            # Update with new values (only if not "unknown")
            for key in ["appliance_type", "brand", "current_issue"]:
                value = extracted.get(key)
                if value and value.lower() != "unknown":
                    self.context[key] = value
            
            print(f"Context updated: {self.context}")

        except json.JSONDecodeError as e:
            print(f"JSON parsing failed: {e}")
        except Exception as e:
            print(f"Context extraction error: {e}")
    
    def _summarize_and_trim(self):
        """Summarize older messages using LLM and trim recent"""
        # Only summarize older messages beyond max_recent
        to_summarize = self.recent_messages[:-self.max_recent]
        if not to_summarize:
            return
        
        text = "\n".join([f"{m['role']}: {m['content']}" for m in to_summarize])
        
        prompt = f"""Summarize this conversation in 2-3 concise sentences.
Focus on: appliance type, brand, issues discussed, and solutions mentioned.

Conversation:
{text}

Provide only the summary, no extra text."""

        try:
            response = self.client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150
            )
            summary_text = response["choices"][0]["message"]["content"].strip()
            
            # Keep summary under control (max 500 chars)
            if len(self.summary) > 500:
                self.summary = summary_text
            else:
                self.summary = f"{self.summary}\n{summary_text}" if self.summary else summary_text
            
            print(f"Conversation summarized")
        except Exception as e:
            print(f"Summarization error: {e}")
        
        # Trim messages
        self.recent_messages = self.recent_messages[-self.max_recent:]
    
    def reset(self):
        """Reset conversation"""
        self.recent_messages = []
        self.summary = ""
        self.context = {
            "appliance_type": None,
            "brand": None,
            "current_issue": None
        }
        print("Conversation reset")