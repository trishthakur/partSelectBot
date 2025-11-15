"""
Data Retriever - Hybrid Search (Exact Match + Embeddings)
Fast, accurate, and scalable for large datasets
"""

import os
import json
import csv
import pickle
import re
from typing import List, Dict, Optional
import numpy as np


class DataRetriever:
    """Fetches relevant data using hybrid search (exact + semantic)"""
    
    def __init__(self, db_base: str = "database"):
        self.db_base = db_base
        self.model = None
        
        # Load indexes
        self.products_index = self._load_csv("product_index.csv")
        self.repairs_index = self._load_csv("repairs_index.csv")
        self.blogs_index = self._load_csv("blogs_index.csv")
        
        print(f" Data Retriever loaded:")
        print(f"   {len(self.products_index)} products")
        print(f"   {len(self.repairs_index)} repairs")
        print(f"   {len(self.blogs_index)} blogs")
        
        # Load or create embeddings
        self._load_or_create_embeddings()
    
    def _load_csv(self, filename: str) -> List[Dict]:
        """Load CSV index"""
        csv_path = os.path.join(self.db_base, filename)
        
        if not os.path.exists(csv_path):
            print(f" {filename} not found")
            return []
        
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                return list(csv.DictReader(f))
        except Exception as e:
            print(f" Error loading {filename}: {e}")
            return []
    
    def _get_embedding_model(self):
        """Lazy load embedding model"""
        if self.model is None:
            try:
                from sentence_transformers import SentenceTransformer
                print(" Loading embedding model (one-time)...")
                # Fast, efficient model - good balance of speed and quality
                self.model = SentenceTransformer('all-MiniLM-L6-v2')
                print(" Embedding model loaded")
            except ImportError:
                print(" sentence-transformers not installed. Run: pip install sentence-transformers")
                raise
        return self.model
    
    def _load_or_create_embeddings(self):
        """Load pre-computed embeddings or create them"""
        embeddings_dir = os.path.join(self.db_base, "embeddings")
        os.makedirs(embeddings_dir, exist_ok=True)
        
        # Products
        self.products_embeddings = self._load_or_compute_embeddings(
            data=self.products_index,
            text_func=lambda x: x.get('name', ''),  
            cache_path=os.path.join(embeddings_dir, "products_embeddings.pkl")
        )

        # Repairs
        self.repairs_embeddings = self._load_or_compute_embeddings(
            data=self.repairs_index,
            text_func=lambda x: x.get('page_title', ''), 
            cache_path=os.path.join(embeddings_dir, "repairs_embeddings.pkl")
        )

        # Blogs
        self.blogs_embeddings = self._load_or_compute_embeddings(
            data=self.blogs_index,
            text_func=lambda x: x.get('meta_description', ''),  
            cache_path=os.path.join(embeddings_dir, "blogs_embeddings.pkl")
        )
    
    def _load_or_compute_embeddings(self, data: List[Dict], text_func, cache_path: str):
        """Load cached embeddings or compute new ones"""
        
        # Try to load cached
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    embeddings = pickle.load(f)
                print(f" Loaded cached embeddings from {os.path.basename(cache_path)}")
                return embeddings
            except Exception as e:
                print(f" Error loading cache: {e}, recomputing...")
        
        # Compute embeddings
        if not data:
            return None
        
        print(f" Computing embeddings for {len(data)} items...")
        model = self._get_embedding_model()
        
        texts = [text_func(item) for item in data]
        embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        
        # Cache for future use
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(embeddings, f)
            print(f" Cached embeddings to {os.path.basename(cache_path)}")
        except Exception as e:
            print(f" Could not cache embeddings: {e}")
        
        return embeddings
    
    def _is_part_number(self, query: str) -> bool:
        """Check if query looks like a part number"""
        query = query.strip().upper()
        
        # Common part number patterns from your data
        patterns = [
            r'^PS\d+$',                    # PS11754317
            r'^WP[A-Z0-9]+$',              # WPW10404412, WP2179243
            r'^WR[0-9A-Z]+$',              # WR71X10683, WR78X30953
            r'^DA[0-9\-]+[A-Z]?$',         # DA97-08067B, DA63-05506A
            r'^\d{10}$',                   # 5304522259, 0064056500
            r'^[A-Z]{2,3}[0-9]{2}X[0-9]+$', # WR71X10683
            r'^[A-Z]{3}\d+[A-Z]\d+$',      # AJP73334413, ACQ85428622
            r'^EBR\d+$',                   # EBR41531310
            r'^AGB\d+$',                   # AGB73093001
            r'^W\d+[A-Z]?\d*$',            # W10921674, W11229977
        ]
        
        for pattern in patterns:
            if re.match(pattern, query):
                return True
        
        return False
    
    def _exact_match_products(self, query: str) -> List[str]:
        """Find exact part number matches"""
        query = query.strip().upper()
        matches = []
        
        for product in self.products_index:
            part_num = product.get('part_number', '').upper()
            name = product.get('name', '').upper()
            
            # Check part number or if it appears in name
            if query == part_num or query in name:
                matches.append(product.get('part_number'))
        
        return matches
    
    def _semantic_search(self, query: str, embeddings, candidates: List[Dict], 
                        top_k: int, id_field: str) -> List[str]:
        """Perform semantic search using cosine similarity"""
        
        if embeddings is None or not candidates:
            return []
        
        # Encode query
        model = self._get_embedding_model()
        query_embedding = model.encode(query, convert_to_numpy=True)
        
        # Compute cosine similarity
        similarities = np.dot(embeddings, query_embedding) / (
            np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Return IDs
        return [candidates[idx].get(id_field) for idx in top_indices if idx < len(candidates)]
    
    def search_products(self, search_query: str, brand: str = None, 
                       appliance_type: str = None, top_k: int = 5) -> List[Dict]:
        """Search for products using hybrid search (exact + semantic)"""
        
        print(f"\n🔍 Searching products: '{search_query}'")
        
        ranked_ids = []
        
        # Step 1: Try exact match for part numbers
        if self._is_part_number(search_query):
            print("   Detected part number - using exact match")
            exact_matches = self._exact_match_products(search_query)
            
            if exact_matches:
                print(f"   Found {len(exact_matches)} exact match(es)")
                ranked_ids = exact_matches[:top_k]
            else:
                print("   No exact match found, falling back to semantic search")
        
        # Step 2: If no exact matches, use semantic search
        if not ranked_ids:
            print("   Using semantic search")
            
            # Filter candidates
            candidates = self.products_index
            candidate_embeddings = self.products_embeddings
            
            if brand:
                indices = [i for i, c in enumerate(candidates) 
                          if brand.lower() in c.get('brand', '').lower()]
                candidates = [candidates[i] for i in indices]
                if candidate_embeddings is not None:
                    candidate_embeddings = candidate_embeddings[indices]
            
            if appliance_type:
                indices = [i for i, c in enumerate(candidates) 
                          if appliance_type.lower() in c.get('category', '').lower()]
                candidates = [candidates[i] for i in indices]
                if candidate_embeddings is not None:
                    candidate_embeddings = candidate_embeddings[indices]
            
            if not candidates:
                print(" No products found")
                return []
            
            # Semantic search
            ranked_ids = self._semantic_search(
                query=search_query,
                embeddings=candidate_embeddings,
                candidates=candidates,
                top_k=top_k,
                id_field='part_number'
            )
        
        # Step 3: Load full JSONs
        products = []
        for part_num in ranked_ids:
            if not part_num:
                continue
                
            json_path = os.path.join(self.db_base, "product_json_db", f"{part_num}.json")
            
            if os.path.exists(json_path):
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        product_data = json.load(f)
                    
                    products.append({
                        "part_number": product_data.get('partselect_number'),
                        "name": product_data.get('title'),
                        "price": product_data.get('price'),
                        "brand": product_data.get('brand'),
                        "description": product_data.get('description'),
                        "url": product_data.get('url'),
                        "images": product_data.get('images', []),
                        "availability": product_data.get('availability'),
                        "manufacturer_part": product_data.get('manufacturer_part_number'),
                        "fits_models": product_data.get('fits_models', [])[:5],
                        "symptoms": product_data.get('symptoms_fixed', [])[:5],
                        "specifications": product_data.get('specifications', {})
                    })
                    print(f"   Loaded: {part_num}")
                except Exception as e:
                    print(f"   Error loading {part_num}: {e}")
        
        return products
    
    def search_repairs(self, issue_description: str, appliance_type: str = None, 
                      top_k: int = 5) -> List[Dict]:
        """Search for repair guides using semantic search"""
        
        print(f"\n🔧 Searching repairs: '{issue_description}'")
        
        # Filter candidates
        candidates = self.repairs_index
        candidate_embeddings = self.repairs_embeddings
        
        if appliance_type:
            indices = [i for i, c in enumerate(candidates) 
                      if appliance_type.lower() in c.get('filename', '').lower()]
            candidates = [candidates[i] for i in indices]
            if candidate_embeddings is not None:
                candidate_embeddings = candidate_embeddings[indices]
        
        if not candidates:
            print(" No repair guides found")
            return []
        
        # Semantic search
        ranked_files = self._semantic_search(
            query=issue_description,
            embeddings=candidate_embeddings,
            candidates=candidates,
            top_k=top_k,
            id_field='filename'
        )
        
        # Load full JSONs
        repairs = []
        for filename in ranked_files:
            if not filename:
                continue
                
            json_path = os.path.join(self.db_base, "repairs_json_db", filename)
            
            if os.path.exists(json_path):
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        repair_data = json.load(f)
                    
                    repairs.append({
                        "filename": filename,
                        "title": repair_data.get('page_title'),
                        "symptom": repair_data.get('symptom'),
                        "url": repair_data.get('url'),
                        "difficulty": repair_data.get('difficulty_rating'),
                        "time_required": repair_data.get('time_required'),
                        "tools_needed": repair_data.get('tools_needed', []),
                        "main_video_url": repair_data.get('main_video_url'),
                        "main_video_title": repair_data.get('main_video_title'),
                        "repair_steps": repair_data.get('repair_steps', []),
                        "common_parts": repair_data.get('common_failing_parts', [])[:5],
                        "quick_nav": repair_data.get('quick_navigation', []),
                        "total_videos": repair_data.get('total_videos')
                    })
                    print(f"   Loaded: {filename}")
                except Exception as e:
                    print(f"   Error loading {filename}: {e}")
        
        return repairs
    
    def search_blogs(self, topic: str, top_k: int = 5) -> List[Dict]:
        """Search for blog articles using semantic search"""
        
        print(f"\n Searching blogs: '{topic}'")
        
        if not self.blogs_index:
            return []
        
        # Semantic search
        ranked_files = self._semantic_search(
            query=topic,
            embeddings=self.blogs_embeddings,
            candidates=self.blogs_index,
            top_k=top_k,
            id_field='filename'
        )
        
        # Load full JSONs
        blogs = []
        for filename in ranked_files:
            if not filename:
                continue
                
            json_path = os.path.join(self.db_base, "blogs_json_db", filename)
            
            if os.path.exists(json_path):
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        blog_data = json.load(f)
                    
                    blogs.append({
                        "filename": filename,
                        "url": blog_data.get('url'),
                        "title": blog_data.get('title'),
                        "subtitle": blog_data.get('subtitle'),
                        "meta_description": blog_data.get('meta_description'),
                        "content": blog_data.get('content'),
                        "sections": blog_data.get('sections', []),
                        "headings": blog_data.get('headings', []),
                        "appliance_type": blog_data.get('appliance_type'),
                        "symptoms": blog_data.get('symptoms', []),
                        "parts_mentioned": blog_data.get('parts_mentioned', [])
                    })
                    print(f"   Loaded: {filename}")
                except Exception as e:
                    print(f"   Error loading {filename}: {e}")
        
        return blogs
    
    def rebuild_embeddings(self):
        """Force rebuild all embeddings (call after updating CSVs)"""
        print(" Rebuilding all embeddings...")
        
        embeddings_dir = os.path.join(self.db_base, "embeddings")
        
        # Remove cached files
        for cache_file in ["products_embeddings.pkl", "repairs_embeddings.pkl", "blogs_embeddings.pkl"]:
            cache_path = os.path.join(embeddings_dir, cache_file)
            if os.path.exists(cache_path):
                os.remove(cache_path)
        
        # Rebuild
        self._load_or_create_embeddings()
        print(" All embeddings rebuilt")


if __name__ == "__main__":
    # Test the retriever
    retriever = DataRetriever()
    
    print("\n" + "="*60)
    print("TEST 1: Exact part number match")
    print("="*60)
    products = retriever.search_products("PS11754317")
    print(f"Found {len(products)} products")
    if products:
        print(f"  - {products[0]['name']}")
    
    print("\n" + "="*60)
    print("TEST 2: Semantic search")
    print("="*60)
    products = retriever.search_products("dishwasher door latch")
    print(f"Found {len(products)} products")
    for p in products[:3]:
        print(f"  - {p['name']}")
    
    print("\n" + "="*60)
    print("TEST 3: Repair search")
    print("="*60)
    repairs = retriever.search_repairs("dishwasher won't start")
    print(f"Found {len(repairs)} repairs")
    for r in repairs[:3]:
        print(f"  - {r['title']}")