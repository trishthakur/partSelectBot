"""
Agent 1 - Router Agent
Decides which tools to call based on user query
"""

from typing import List, Dict
from utils.llm import DeepSeekClient


class RouterAgent:
    """Determines which databases to search"""
    
    def __init__(self):
        self.client = DeepSeekClient()
        
        # Available tools
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "search_products",
                    "description": "Search for appliance parts and products. Use when user asks about specific parts, part numbers, pricing, or wants to buy something.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "search_query": {
                                "type": "string",
                                "description": "What to search for (e.g., 'Whirlpool door latch', 'water filter')"
                            },
                            "brand": {
                                "type": "string",
                                "description": "Brand name if specified (e.g., 'Whirlpool', 'Samsung')"
                            },
                            "appliance_type": {
                                "type": "string",
                                "description": "Type of appliance (e.g., 'refrigerator', 'dishwasher')"
                            }
                        },
                        "required": ["search_query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "search_repairs",
                    "description": "Search for repair guides and troubleshooting. Use when user has a problem, issue, or needs repair instructions.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "issue_description": {
                                "type": "string",
                                "description": "The problem or symptom (e.g., 'not cooling', 'leaking', 'won't start')"
                            },
                            "appliance_type": {
                                "type": "string",
                                "description": "Type of appliance having issues"
                            }
                        },
                        "required": ["issue_description"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "search_blogs",
                    "description": "Search for articles, guides, tips, and general information. Use when user asks 'how to', wants advice, or general knowledge.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "topic": {
                                "type": "string",
                                "description": "The topic to search for"
                            }
                        },
                        "required": ["topic"]
                    }
                }
            }
        ]
        
        print("🎯 Router Agent initialized")
    
    def route_query(self, query: str, conversation_context: str = "") -> List[Dict]:
        """
        Determine which tools to call based on query
        Returns list of tool calls with parameters
        """
        
        system_prompt = """You are a routing assistant for an appliance parts assistant.
Analyze the user's query and decide which tools to call.

You can call MULTIPLE tools if needed. For example:
- "My fridge is leaking, what parts do I need?" → search_repairs AND search_products
- "Compare Whirlpool door latches" → search_products only
- "How do I fix a noisy dishwasher?" → search_repairs (and optionally search_blogs)

Consider the conversation context when routing."""

        messages = [
            {"role": "system", "content": system_prompt}
        ]
        
        if conversation_context:
            messages.append({
                "role": "user", 
                "content": f"Context from previous conversation:\n{conversation_context}"
            })
        
        messages.append({
            "role": "user",
            "content": f"User query: {query}"
        })
        
        try:
            print(f"\n🎯 Router Agent analyzing: '{query}'")
            
            response = self.client.chat_completion(
                messages=messages,
                tools=self.tools,
                tool_choice="auto",
                max_tokens=500
            )
            
            message = response["choices"][0]["message"]
            
            # Check if tools were called
            if message.get("tool_calls"):
                tool_calls = []
                for tool_call in message["tool_calls"]:
                    tool_name = tool_call["function"]["name"]
                    tool_args = tool_call["function"]["arguments"]
                    
                    # Parse arguments if string
                    if isinstance(tool_args, str):
                        import json
                        tool_args = json.loads(tool_args)
                    
                    tool_calls.append({
                        "tool": tool_name,
                        "arguments": tool_args
                    })
                    
                    print(f"  ✅ Tool: {tool_name}")
                    print(f"     Args: {tool_args}")
                
                return tool_calls
            else:
                # Fallback: treat as product search
                print("  ⚠️ No tools called")
                return []
        
        except Exception as e:
            print(f"❌ Router error: {e}")
            # Fallback
            return [{
                "tool": "search_products",
                "arguments": {"search_query": query}
            }]