import os
import aiohttp
import asyncio
import ssl
import certifi
from typing import List, Dict
from dotenv import load_dotenv
import re
import yaml
from LLMSetup import initialize_model

# Load environment variables
load_dotenv()

class ImageSearcher:
    def __init__(self, use_llm=True):
        self.api_key = os.environ.get("Custom_Search_API_KEY")
        self.custom_search_id = os.environ.get("GOOGLE_CUSTOM_SEARCH_ID")
        self.use_llm = use_llm
        
        # Load prompts from YAML file
        try:
            with open('prompts.yml', 'r') as file:
                self.prompts = yaml.safe_load(file)
                print("Successfully loaded prompts from prompts.yml")
        except Exception as e:
            print(f"Error loading prompts from YAML: {e}")
            self.prompts = {}
        
        # Add detailed debugging for environment variables
        if not self.api_key:
            print("Warning: Custom_Search_API_KEY environment variable is not set")
        if not self.custom_search_id:
            print("Warning: GOOGLE_CUSTOM_SEARCH_ID environment variable is not set")
            
        if not self.api_key or not self.custom_search_id:
            raise ValueError("Missing required environment variables: Custom_Search_API_KEY or GOOGLE_CUSTOM_SEARCH_ID")
        
        # Initialize LLM if enabled
        if self.use_llm:
            try:
                self.llm_config = initialize_model("gemini")
                self.llm = self.llm_config["model"]
                print(f"LLM initialized for query optimization using {self.llm_config['model_name']}")
            except Exception as e:
                print(f"Failed to initialize LLM: {e}. Falling back to heuristic query optimization.")
                self.use_llm = False
        
        print(f"ImageSearcher initialized successfully")
        print(f"API Key length: {len(self.api_key)} characters")
        print(f"Custom Search ID length: {len(self.custom_search_id)} characters")
        print(f"Using LLM for query optimization: {self.use_llm}")
        
        self.base_url = "https://www.googleapis.com/customsearch/v1"
    
    async def _optimize_query_with_llm(self, query: str) -> str:
        """
        Use Gemini to generate an optimized image search query from article text.
        """
        try:
            # Get the prompt template from prompts.yml
            prompt_template = self.prompts.get('image_search_prompt', '')
            if not prompt_template:
                raise ValueError("image_search_prompt not found in prompts.yml - cannot proceed with LLM query optimization")
            
            # Format the prompt with article text
            # Limit article text to avoid token limits
            truncated_query = query[:1500]
            prompt = prompt_template.replace("{article_text}", truncated_query)
            
            # Generate response from Gemini
            response = await self.llm.generate_content_async(prompt)
            response_text = response.text
            
            # Extract the search query from the response
            if "[Search Query]:" in response_text:
                search_query = response_text.split("[Search Query]:")[1].strip()
            else:
                search_query = response_text.strip()
                
            # Clean up any remaining markdown or formatting
            search_query = search_query.replace('`', '').replace('"', '').strip()
            
            # Simplify the query by taking only the first 4-5 words if it's too long
            words = search_query.split()
            if len(words) > 5:
                # Keep the first few words (likely the most important entities)
                search_query = " ".join(words[:5])
                
            print(f"LLM generated search query: {search_query}")
            return search_query  # Return the raw search query with no additional directives
        except Exception as e:
            print(f"Error generating query with LLM: {e}")
            # Fall back to heuristic method if there's an issue with the LLM
            # but not if the error was about missing prompt
            if "image_search_prompt not found" in str(e):
                raise
            return self._optimize_query_heuristic(query)
    
    def _optimize_query_heuristic(self, query: str) -> str:
        """
        Optimizes the search query for better image results using heuristics.
        This is a fallback if LLM optimization fails.
        """
        # Remove HTML tags if present
        query = re.sub(r'<[^>]+>', '', query)
        
        # Clean and normalize the query
        query = query.strip()
        
        # If query is too long, extract main nouns and entities
        if len(query.split()) > 10:
            # Extract noun phrases and named entities first
            nouns_and_entities = self._extract_key_entities(query)
            
            # If we have enough entities, use them
            if len(nouns_and_entities) >= 3:
                base_query = " ".join(nouns_and_entities[:6])  # Use up to 6 key entities
            else:
                # Otherwise use first part of the query (more reliable than complex filtering)
                base_query = " ".join(query.split()[:6])
        else:
            # Short queries are kept intact
            base_query = query
        
        return base_query  # Return the simplified query without directives
    
    def _extract_key_entities(self, text: str) -> List[str]:
        """Extract important terms from text using basic heuristics."""
        # Split into words
        words = text.split()
        
        # Identify potential entities (capitalized words not at start of sentence)
        entities = set()
        for i, word in enumerate(words):
            clean_word = word.strip(',.!?:;()[]{}"\'-')
            if not clean_word:
                continue
                
            # Likely a proper noun if capitalized (not at sentence start)
            is_sentence_start = i == 0 or words[i-1][-1] in '.!?'
            if clean_word[0].isupper() and not is_sentence_start:
                entities.add(clean_word)
        
        # Get numeric values (years, statistics, etc.)
        numbers = [w.strip(',.!?:;()[]{}"\'-') for w in words if w.strip(',.!?:;()[]{}"\'-').isdigit()]
        
        # Extract two-word phrases (likely more meaningful)
        phrases = []
        for i in range(len(words) - 1):
            w1 = words[i].strip(',.!?:;()[]{}"\'-')
            w2 = words[i+1].strip(',.!?:;()[]{}"\'-')
            if len(w1) > 2 and len(w2) > 2:  # Both words are substantial
                phrases.append(f"{w1} {w2}")
        
        # Combine entities, numbers, and some phrases
        key_terms = list(entities) + numbers
        if len(key_terms) < 3 and phrases:
            key_terms.extend(phrases[:3])  # Add up to 3 phrases
            
        # If we still don't have enough terms, add longer individual words
        if len(key_terms) < 4:
            # Add longer words (likely more meaningful)
            long_words = [w.strip(',.!?:;()[]{}"\'-') for w in words 
                        if len(w.strip(',.!?:;()[]{}"\'-')) > 6]
            key_terms.extend(long_words[:5])
            
        return list(dict.fromkeys(key_terms))  # Remove duplicates but preserve order

    async def search_images(self, query: str, num_images: int = 3) -> List[Dict[str, str]]:
        """
        Search for images using Google Custom Search API
        
        Args:
            query (str): The search query
            num_images (int): Number of images to return (max 10)
            
        Returns:
            List[Dict[str, str]]: List of image information including URL and title
        """
        # Choose optimization method based on configuration
        if self.use_llm:
            optimized_query = await self._optimize_query_with_llm(query)
        else:
            optimized_query = self._optimize_query_heuristic(query)
            
        print(f"Original query: {query[:100]}...")
        print(f"Optimized query: {optimized_query}")
        
        # Improved search parameters for more consistent results - removed restrictive filters
        params = {
            'key': self.api_key,
            'cx': self.custom_search_id,
            'q': optimized_query,
            'searchType': 'image',
            'num': min(num_images, 10),
            'safe': 'active',
            # Removed imgSize and imgType to get more results
        }
        
        ssl_context = ssl.create_default_context(cafile=certifi.where())
        connector = aiohttp.TCPConnector(ssl=ssl_context)
        
        async with aiohttp.ClientSession(connector=connector) as session:
            try:
                async with session.get(self.base_url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        print(f"API Response status: {response.status}")
                        
                        # Debug response content
                        if 'error' in data:
                            print(f"API error: {data['error'].get('message', 'Unknown error')}")
                            return []
                            
                        if 'items' in data:
                            images = [{
                                'url': item['link'],
                                'title': item['title'],
                                'thumbnailUrl': item.get('image', {}).get('thumbnailLink', ''),
                                'width': item.get('image', {}).get('width', 0),
                                'height': item.get('image', {}).get('height', 0)
                            } for item in data['items']]
                            print(f"Found {len(images)} images")
                            return images
                        else:
                            error_msg = data.get('error', {}).get('message', 'No error message provided')
                            queries_info = data.get('searchInformation', {})
                            print(f"No items found in response. Error: {error_msg}")
                            print(f"Search information: {queries_info}")
                            return []
                    else:
                        print(f"API request failed with status: {response.status}")
                        response_text = await response.text()
                        print(f"Response content: {response_text[:200]}")  # Show limited content
                        return []
            except Exception as e:
                print(f"Error searching for images: {str(e)}")
                import traceback
                print(f"Full traceback: {traceback.format_exc()}")
                return []