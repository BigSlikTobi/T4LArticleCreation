import os
import aiohttp
import asyncio
import ssl
import certifi
from typing import List, Dict, Tuple, Optional
from dotenv import load_dotenv
import re
import yaml
import json
import time
import hashlib
from pathlib import Path
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
        
        # Rate limiting settings
        self.requests_per_day = 100  # Google's default limit
        self.requests_remaining = self.requests_per_day
        self.last_reset_time = time.time()
        self.reset_period = 24 * 60 * 60  # 24 hours in seconds
        self.min_wait_time = 1  # Minimum seconds between requests
        self.last_request_time = 0
        
        self.base_url = "https://www.googleapis.com/customsearch/v1"
    
    def _check_rate_limit(self):
        """Check and update rate limiting, reset if needed"""
        current_time = time.time()
        
        # Reset count if a new 24-hour period has started
        if current_time - self.last_reset_time > self.reset_period:
            self.requests_remaining = self.requests_per_day
            self.last_reset_time = current_time
            print(f"Rate limit reset. {self.requests_remaining} requests available for today.")
            
        # Enforce minimum wait time between requests
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_wait_time:
            sleep_time = self.min_wait_time - time_since_last
            print(f"Waiting {sleep_time:.2f}s between requests...")
            time.sleep(sleep_time)
            
        self.last_request_time = time.time()
            
        # Check if we have requests remaining
        if self.requests_remaining <= 0:
            seconds_until_reset = self.reset_period - (current_time - self.last_reset_time)
            print(f"API quota exceeded. Reset in {seconds_until_reset/60:.1f} minutes.")
            return False
            
        return True
    
    def _use_request(self):
        """Mark that we've used a request from our quota"""
        self.requests_remaining -= 1
        print(f"API request used. {self.requests_remaining} requests remaining for today.")

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

    async def validate_image_url(self, session: aiohttp.ClientSession, image_data: Dict[str, str]) -> bool:
        """
        Validate if a URL actually points to a valid image by making a HEAD request.
        
        Args:
            session (aiohttp.ClientSession): The active client session
            image_data (Dict[str, str]): Dictionary containing image data including URL
            
        Returns:
            bool: True if the URL is a valid image, False otherwise
        """
        url = image_data['url']
        try:
            # Try HEAD request first (more efficient)
            async with session.head(url, allow_redirects=True, timeout=5) as response:
                if response.status != 200:
                    print(f"Invalid image URL (status {response.status}): {url}")
                    return False
                
                # Check content type
                content_type = response.headers.get('Content-Type', '')
                if not content_type.startswith('image/'):
                    print(f"Invalid content type ({content_type}): {url}")
                    return False
                
                return True
        except Exception as e:
            # If HEAD fails, try GET as fallback (some servers don't support HEAD)
            try:
                async with session.get(url, timeout=5) as response:
                    if response.status != 200:
                        print(f"Invalid image URL (status {response.status}): {url}")
                        return False
                    
                    content_type = response.headers.get('Content-Type', '')
                    if not content_type.startswith('image/'):
                        print(f"Invalid content type ({content_type}): {url}")
                        return False
                    
                    return True
            except Exception as e:
                print(f"Error validating image URL {url}: {str(e)}")
                return False
            
    def rank_image_by_relevance(self, image: Dict[str, str], content: str, query: str) -> float:
        """
        Calculate a relevance score for an image based on its metadata and content.
        
        Args:
            image (Dict[str, str]): The image data including title and URL
            content (str): The article content
            query (str): The optimized search query
            
        Returns:
            float: A relevance score (higher is more relevant)
        """
        score = 0.0
        
        # Extract key terms from content to compare with image metadata
        content_words = set(re.findall(r'\b\w+\b', content.lower()))
        query_words = set(query.lower().split())
        
        # Clean image title for comparison
        image_title = image['title'].lower()
        title_words = set(re.findall(r'\b\w+\b', image_title))
        
        # Score based on title word matches with content
        content_match_count = len(title_words.intersection(content_words))
        score += content_match_count * 1.0
        
        # Score based on title word matches with query (weighted higher)
        query_match_count = len(title_words.intersection(query_words))
        score += query_match_count * 2.0
        
        # Bonus points for each key term from query that appears in title
        for term in query_words:
            if term.lower() in image_title:
                score += 0.5
        
        # Image quality factors - prefer larger images but not excessively
        width = image.get('width', 0)
        height = image.get('height', 0)
        
        # Bonus for images with good resolution (but not excessive)
        if width > 0 and height > 0:
            aspect_ratio = width / max(height, 1)
            # Prefer images with reasonable aspect ratios (not too narrow or wide)
            if 0.5 <= aspect_ratio <= 2.0:
                score += 0.5
            
            # Prefer images with higher resolution (but not excessively large)
            resolution = width * height
            if 100000 <= resolution <= 2000000:  # Reasonable size range
                score += 0.5
        
        return score
    
    async def rank_images(self, images: List[Dict[str, str]], content: str, query: str) -> List[Dict[str, str]]:
        """
        Rank images by relevance to the content and query.
        
        Args:
            images (List[Dict[str, str]]): List of validated images
            content (str): The article content
            query (str): The optimized search query
            
        Returns:
            List[Dict[str, str]]: List of images sorted by relevance (most relevant first)
        """
        if not images:
            return []
        
        # Calculate relevance score for each image
        scored_images = [(image, self.rank_image_by_relevance(image, content, query)) for image in images]
        
        # Sort images by score in descending order (highest score first)
        scored_images.sort(key=lambda x: x[1], reverse=True)
        
        # Print ranking information for debugging
        print("\nImage ranking results:")
        for i, (image, score) in enumerate(scored_images, 1):
            print(f"Rank {i}: '{image['title']}' - Score: {score:.2f}")
        
        # Return sorted images without scores
        return [image for image, _ in scored_images]
    
    async def _api_request_with_backoff(self, session: aiohttp.ClientSession, params: Dict, max_retries: int = 3) -> Optional[Dict]:
        """
        Make API request with exponential backoff for retries
        
        Args:
            session (aiohttp.ClientSession): Active client session
            params (Dict): Request parameters
            max_retries (int): Maximum number of retry attempts
            
        Returns:
            Optional[Dict]: JSON response data or None if all retries failed
        """
        retry_count = 0
        base_wait_time = 2  # Start with 2 seconds wait
        
        while retry_count <= max_retries:
            if not self._check_rate_limit():
                # If we've hit rate limits, inform and return None
                print("Rate limit reached. Cannot make more API requests today.")
                return None
                
            try:
                async with session.get(self.base_url, params=params) as response:
                    self._use_request()  # Count this request against our quota
                    
                    if response.status == 200:
                        return await response.json()
                    elif response.status == 429:  # Rate limit exceeded
                        print(f"Rate limit exceeded (429). Retrying after backoff...")
                        response_text = await response.text()
                        print(f"Error details: {response_text[:200]}")
                        
                        # Mark that we've hit the quota limit
                        self.requests_remaining = 0
                        return None
                    else:
                        print(f"API request failed with status: {response.status}")
                        response_text = await response.text()
                        print(f"Response content: {response_text[:200]}")
                        
                        # For other errors, we'll retry with backoff
                        if retry_count < max_retries:
                            wait_time = base_wait_time * (2 ** retry_count)
                            print(f"Retrying in {wait_time} seconds...")
                            await asyncio.sleep(wait_time)
                            retry_count += 1
                        else:
                            print("Max retries exceeded")
                            return None
            except Exception as e:
                print(f"Error during API request: {str(e)}")
                if retry_count < max_retries:
                    wait_time = base_wait_time * (2 ** retry_count)
                    print(f"Retrying in {wait_time} seconds...")
                    await asyncio.sleep(wait_time)
                    retry_count += 1
                else:
                    print("Max retries exceeded")
                    return None
        
        return None  # If we get here, all retries failed
            
    async def search_images(self, query: str, num_images: int = 3, content: str = None) -> List[Dict[str, str]]:
        """
        Search for images using Google Custom Search API, validate and rank them by relevance
        
        Args:
            query (str): The search query
            num_images (int): Number of images to return (max 10)
            content (str): Article content for ranking relevance (if None, uses query)
            
        Returns:
            List[Dict[str, str]]: List of image information including URL and title, ranked by relevance
        """
        # Use query as content if no content provided for ranking
        content_for_ranking = content or query
        
        # Choose optimization method based on configuration
        if self.use_llm:
            optimized_query = await self._optimize_query_with_llm(query)
        else:
            optimized_query = self._optimize_query_heuristic(query)
            
        print(f"Original query: {query[:100]}...")
        print(f"Optimized query: {optimized_query}")
        
        # Prepare search parameters
        params = {
            'key': self.api_key,
            'cx': self.custom_search_id,
            'q': optimized_query,
            'searchType': 'image',
            'num': min(num_images * 2, 10),  # Request more images than needed to account for filtering
            'safe': 'active',
        }
        
        ssl_context = ssl.create_default_context(cafile=certifi.where())
        connector = aiohttp.TCPConnector(ssl=ssl_context)
        
        async with aiohttp.ClientSession(connector=connector) as session:
            try:
                # Make API request with backoff and rate limiting
                data = await self._api_request_with_backoff(session, params)
                
                # If API request failed and we couldn't get fresh data
                if not data:
                    print("API request failed or rate-limited. No fallback available.")
                    return []
                
                # Process API response
                if 'error' in data:
                    print(f"API error: {data['error'].get('message', 'Unknown error')}")
                    return []
                    
                if 'items' in data:
                    # Create image data list
                    images = [{
                        'url': item['link'],
                        'title': item['title'],
                        'thumbnailUrl': item.get('image', {}).get('thumbnailLink', ''),
                        'width': item.get('image', {}).get('width', 0),
                        'height': item.get('image', {}).get('height', 0)
                    } for item in data['items']]
                    print(f"Found {len(images)} images from API")
                    
                    # Validate each image URL
                    print("Validating image URLs...")
                    validation_tasks = [self.validate_image_url(session, img) for img in images]
                    validation_results = await asyncio.gather(*validation_tasks)
                    
                    # Filter valid images
                    valid_images = [img for img, is_valid in zip(images, validation_results) if is_valid]
                    print(f"After validation: {len(valid_images)} valid images out of {len(images)} total")
                    
                    if valid_images:
                        # Rank the valid images by relevance to content
                        ranked_images = await self.rank_images(valid_images, content_for_ranking, optimized_query)
                        print(f"Ranked {len(ranked_images)} images by relevance to content")
                        
                        # Return only the requested number of ranked images
                        return ranked_images[:num_images]
                    else:
                        print("Warning: No valid images found after validation")
                        return []
                else:
                    error_msg = data.get('error', {}).get('message', 'No error message provided')
                    queries_info = data.get('searchInformation', {})
                    print(f"No items found in response. Error: {error_msg}")
                    print(f"Search information: {queries_info}")
                    return []
            except Exception as e:
                print(f"Error searching for images: {str(e)}")
                import traceback
                print(f"Full traceback: {traceback.format_exc()}")
                return []