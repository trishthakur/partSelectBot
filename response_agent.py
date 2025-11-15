"""
Agent 2 - Response Generator
Has FULL context from retrieved data 
"""

from typing import List, Dict
from utils.llm import DeepSeekClient
import json


class ResponseAgent:
    """Generates responses with full context"""
    
    def __init__(self):
        self.client = DeepSeekClient()
        print(" Response Agent initialized")
    
    def generate_response(self, 
                         user_query: str,
                         products: List[Dict] = None,
                         repairs: List[Dict] = None,
                         blogs: List[Dict] = None,
                         conversation_summary: str = "") -> str:
        """
        Generate response with FULL context
        Can answer most detail because we have complete JSON data
        """
        
        # Build comprehensive context
        context = self._build_context(products, repairs, blogs)
        
        # Build system prompt
        system_prompt = """You are a helpful appliance parts assistant.

You have access to COMPLETE information about products, repair guides, and articles.
Answer the user's question thoroughly using the provided data.

Guidelines:
- For comparisons: Compare features, prices, compatibility, and recommend the best option
- For repairs: Explain the issue, steps, parts needed, and include video links
- For product details: Provide ALL relevant information (price, compatibility, specs)
- Be conversational but informative
- ALWAYS include URLs when available
- If data is missing, acknowledge it

IMPORTANT: 
- Never make up information not in the provided data
- Always cite sources with URLs when available
"""

        # Build user message with context
        user_message = f"""User Query: {user_query}

{conversation_summary}

{context}

Please provide a helpful, detailed response to the user's query."""

        try:
            print(f"\n Generating response for: '{user_query}'")
            
            response = self.client.chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                max_tokens=1000
            )
            
            generated_text = response["choices"][0]["message"]["content"]
            
            print(" Response generated")
            return generated_text
        
        except Exception as e:
            print(f" Response generation error: {e}")
            return "I encountered an error generating a response. Please try again."
    
    def _build_context(self, products: List[Dict], repairs: List[Dict], 
                      blogs: List[Dict]) -> str:
        """Build comprehensive context from all data sources"""
        
        context = ""
        
        # Products context
        if products:
            context += "\n=== PRODUCTS FOUND ===\n\n"
            for i, p in enumerate(products, 1):
                context += f"Product {i}: {p.get('name')}\n"
                context += f"  - Part Number: {p.get('part_number')}\n"
                context += f"  - Price: ${p.get('price')}\n"
                context += f"  - Brand: {p.get('brand')}\n"
                context += f"  - URL: {p.get('url')}\n"
                
                if p.get('description'):
                    context += f"  - Description: {p.get('description')[:300]}\n"
                
                if p.get('availability'):
                    context += f"  - Availability: {p.get('availability')}\n"
                
                if p.get('manufacturer_part'):
                    context += f"  - Manufacturer Part: {p.get('manufacturer_part')}\n"
                
                if p.get('fits_models'):
                    models = ', '.join(p.get('fits_models')[:3])
                    context += f"  - Fits Models: {models}\n"
                
                if p.get('symptoms'):
                    symptoms = ', '.join(p.get('symptoms')[:3])
                    context += f"  - Fixes Issues: {symptoms}\n"
                
                if p.get('specifications'):
                    specs = p.get('specifications')
                    if isinstance(specs, dict):
                        spec_str = ', '.join([f"{k}: {v}" for k, v in list(specs.items())[:3]])
                        context += f"  - Specs: {spec_str}\n"
                
                context += "\n"
        
        # Repairs context
        if repairs:
            context += "\n=== REPAIR GUIDES FOUND ===\n\n"
            for i, r in enumerate(repairs, 1):
                context += f"Repair Guide {i}: {r.get('title')}\n"
                context += f"  - Symptom: {r.get('symptom')}\n"
                context += f"  - Difficulty: {r.get('difficulty')}\n"
                context += f"  - Time: {r.get('time_required')}\n"
                context += f"  - URL: {r.get('url')}\n"
                
                if r.get('main_video_url'):
                    context += f"  - Video: {r.get('main_video_url')}\n"
                    context += f"    Title: {r.get('main_video_title')}\n"
                
                if r.get('tools_needed'):
                    tools = ', '.join(r.get('tools_needed')[:5])
                    context += f"  - Tools: {tools}\n"
                
                if r.get('common_parts'):
                    parts = [p.get('part_name') for p in r.get('common_parts')[:3] 
                            if p.get('part_name') != "More Repair Parts"]
                    if parts:
                        context += f"  - Common Parts: {', '.join(parts)}\n"
                
                if r.get('repair_steps'):
                    context += f"  - Total Steps: {len(r.get('repair_steps'))}\n"
                    # Include first 2 steps
                    for step_num, step in enumerate(r.get('repair_steps')[:2], 1):
                        step_title = step.get('step_title', f'Step {step_num}')
                        context += f"    {step_num}. {step_title}\n"
                
                context += "\n"
        
        # Blogs context
        if blogs:
            context += "\n=== ARTICLES FOUND ===\n\n"
            for i, b in enumerate(blogs, 1):
                context += f"Article {i}: {b.get('title', b.get('filename', 'Untitled'))}\n"
                context += f"  - Filename: {b.get('filename')}\n"
                context += f"  - URL: {b.get('url', 'N/A')}\n"
                context += f"  - Subtitle: {b.get('subtitle', '')}\n"
                context += f"  - Meta Description: {b.get('meta_description', '')}\n"
                context += f"  - Appliance Type: {b.get('appliance_type', 'Unknown')}\n"

                # Symptoms and parts mentioned
                if b.get('symptoms'):
                    context += f"  - Symptoms: {', '.join(b['symptoms'])}\n"
                if b.get('parts_mentioned'):
                    context += f"  - Parts Mentioned: {', '.join(b['parts_mentioned'])}\n"

                # Content preview
                if b.get('content'):
                    # Flatten paragraphs if content is a list of dicts
                    if isinstance(b['content'], list):
                        full_text = " ".join(
                            p.get('text', '') for section in b['content'] for p in (section.get('content', []) if isinstance(section, dict) else [])
                        )
                    else:
                        full_text = str(b['content'])
                    context += f"  - Content Preview: {full_text[:300]}...\n"

                # Headings
                if b.get('headings'):
                    context += f"  - Headings: {', '.join(b['headings'])}\n"

                context += "\n"

        return context
            
    def extract_media(self, products: List[Dict] = None, 
                     repairs: List[Dict] = None) -> Dict:
        """Extract images and videos for UI display"""
        
        media = {
            "images": [],
            "videos": [],
            "image_urls": {}  # Map image to product URL
        }
        
        # Extract product images
        if products:
            for product in products:
                if product.get('images'):
                    for img_url in product['images'][:2]:  # Max 2 per product
                        media['images'].append(img_url)
                        media['image_urls'][img_url] = product.get('url')
        
        # Extract repair videos
        if repairs:
            for repair in repairs:
                if repair.get('main_video_url'):
                    media['videos'].append({
                        "url": repair['main_video_url'],
                        "title": repair.get('main_video_title', ''),
                        "thumbnail": repair.get('main_video_thumbnail', '')
                    })
        
        # Deduplicate
        media['images'] = list(dict.fromkeys(media['images']))[:6]
        
        seen_video_urls = set()
        unique_videos = []
        for video in media['videos']:
            if video['url'] not in seen_video_urls:
                seen_video_urls.add(video['url'])
                unique_videos.append(video)
        media['videos'] = unique_videos[:3]
        
        return media
    
    def format_products_for_ui(self, products: List[Dict]) -> List[Dict]:
        """Format products for Streamlit UI display"""
        
        return [
            {
                "id": p.get('part_number'),
                "name": p.get('name'),
                "price": p.get('price'),
                "brand": p.get('brand'),
                "url": p.get('url'),
                "images": p.get('images', [])
            }
            for p in products
        ]