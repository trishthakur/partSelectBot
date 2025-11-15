"""
Orchestrator - Coordinates the 3-agent system
"""

from typing import Tuple, List, Dict
from agent_router import RouterAgent
from data_retriever import DataRetriever
from response_agent import ResponseAgent
from summary_agent import SummaryAgent


class Orchestrator:
    """
    Simplified orchestration:
    1. Router decides which tools to call
    2. Data Retriever fetches top 5 JSONs from each tool
    3. Response Agent gets FULL data and answers ANY question
    4. Summary Agent maintains session context
    """
    
    def __init__(self):
        self.router = RouterAgent()              # Agent 1
        self.retriever = DataRetriever()         # Tools
        self.response_agent = ResponseAgent()    # Agent 2
        self.summary = SummaryAgent()      # Context Manager
        
        print("Orchestrator initialized")
        print("Ready to process queries!\n")
    
    def process_query(self, query: str) -> Tuple[str, List[Dict], Dict]:
        """
        Main pipeline:
        Returns: (response_text, products_for_ui, media_dict)
        """
        
        print(f"\n{'='*60}")
        print(f" Query: {query}")
        print(f" Context: {self.summary.get_context_for_display()}")
        print(f"{'='*60}")
        
        # Add user message to summary
        self.summary.add_message("user", query)
        
        # Get conversation context
        conversation_context = self.summary.get_context_string()
        
        # Step 1: Router decides which tools to call
        tool_calls = self.router.route_query(query, conversation_context)
        
        # Step 2: Retrieve data from each tool
        all_products = []
        all_repairs = []
        all_blogs = []
        
        for tool_call in tool_calls:
            tool_name = tool_call["tool"]
            args = tool_call["arguments"]
            
            if tool_name == "search_products":
                products = self.retriever.search_products(
                    search_query=args.get("search_query"),
                    brand=args.get("brand"),
                    appliance_type=args.get("appliance_type")
                )
                all_products.extend(products)
            
            elif tool_name == "search_repairs":
                repairs = self.retriever.search_repairs(
                    issue_description=args.get("issue_description"),
                    appliance_type=args.get("appliance_type")
                )
                all_repairs.extend(repairs)
            
            elif tool_name == "search_blogs":
                blogs = self.retriever.search_blogs(
                    topic=args.get("topic")
                )
                all_blogs.extend(blogs)
        
        # Check if we found anything
        if not all_products and not all_repairs and not all_blogs:
            response = "Hello! I couldn't find any relevant information. Could you rephrase your question?"
            self.summary.add_message("assistant", response)
            return response, [], {"images": [], "videos": []}
        
        print(f"\n Data Retrieved:")
        print(f"   Products: {len(all_products)}")
        print(f"   Repairs: {len(all_repairs)}")
        print(f"   Blogs: {len(all_blogs)}")
        
        # Step 3: Generate response with FULL context
        response = self.response_agent.generate_response(
            user_query=query,
            products=all_products,
            repairs=all_repairs,
            blogs=all_blogs,
            conversation_summary=conversation_context
        )
        
        # Add response to summary
        self.summary.add_message("assistant", response)
        
        # Step 4: Extract media and format for UI
        media = self.response_agent.extract_media(all_products, all_repairs)
        products_for_ui = self.response_agent.format_products_for_ui(all_products)
        
        print(f" Response generated!")
        print(f"   UI Products: {len(products_for_ui)}")
        print(f"   Images: {len(media['images'])}")
        print(f"   Videos: {len(media['videos'])}\n")
        
        return response, products_for_ui, media
    
    def reset_conversation(self):
        """Reset for new conversation"""
        self.summary.reset()
        print(" Conversation reset")
    
    def get_context_display(self) -> str:
        """Get context for UI display"""
        return self.summary.get_context_for_display()


# Example usage
if __name__ == "__main__":
    orchestrator = Orchestrator()
    
    # Test query
    response, products, media = orchestrator.process_query(
        "Compare Whirlpool door latches and recommend the best one"
    )
    
    print("\n" + "="*60)
    print("RESPONSE:")
    print(response)
    print("\n" + "="*60)
    print(f"Products for UI: {len(products)}")
    print(f"Media: {len(media['images'])} images, {len(media['videos'])} videos")