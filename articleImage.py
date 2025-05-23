import os
import aiohttp
import asyncio
import ssl
import certifi
import io
import hashlib
from supabase import create_client, Client
from typing import List, Dict, Optional
from dotenv import load_dotenv
import re
import yaml
import json
import time
from pathlib import Path
from LLMSetup import initialize_model
from duckduckgo_search import DDGS
from datetime import datetime, timedelta
from google.genai import types  # Added import for types

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
        
        print("ImageSearcher initialized successfully")
        print(f"API Key length: {len(self.api_key)} characters")
        print(f"Custom Search ID length: {len(self.custom_search_id)} characters")
        print(f"Using LLM for query optimization: {self.use_llm}")
        
        # Rate limiting settings
        self.requests_per_day = 100
        self.requests_remaining = self.requests_per_day
        self.last_reset_time = time.time()
        self.reset_period = 24 * 60 * 60
        self.min_wait_time = 1
        self.last_request_time = 0
        
        self.base_url = "https://www.googleapis.com/customsearch/v1"
        self.ddgs_enabled = True  # Enable DuckDuckGo fallback

        # Initialize Supabase client for uploading images
        SUPABASE_URL = os.environ.get("SUPABASE_URL")
        SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
        if not SUPABASE_URL or not SUPABASE_KEY:
            raise ValueError("Missing SUPABASE_URL or SUPABASE_KEY environment variables")
        self.supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    
    def _check_rate_limit(self):
        current_time = time.time()
        if current_time - self.last_reset_time > self.reset_period:
            self.requests_remaining = self.requests_per_day
            self.last_reset_time = current_time
            print(f"Rate limit reset. {self.requests_remaining} requests available for today.")
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_wait_time:
            sleep_time = self.min_wait_time - time_since_last
            print(f"Waiting {sleep_time:.2f}s between requests...")
            time.sleep(sleep_time)
        self.last_request_time = time.time()
        if self.requests_remaining <= 0:
            seconds_until_reset = self.reset_period - (current_time - self.last_reset_time)
            print(f"API quota exceeded. Reset in {seconds_until_reset/60:.1f} minutes.")
            return False
        return True

    def _use_request(self):
        self.requests_remaining -= 1
        print(f"API request used. {self.requests_remaining} requests remaining for today.")
    
    async def _optimize_query_with_llm(self, query: str) -> str:
        """
        Use Gemini to generate an optimized image search query from article text.
        """
        try:
            prompt_template = self.prompts.get('image_search_prompt', '')
            if not prompt_template:
                raise ValueError("image_search_prompt not found in prompts.yml - cannot proceed with LLM query optimization")
            
            truncated_query = query[:1500]
            prompt = prompt_template.replace("{article_text}", truncated_query)
            
            response = await asyncio.to_thread(
                self.llm.generate_content,
                model=self.llm_config["model_name"],
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.2,
                    max_output_tokens=1024,
                    tools=self.llm_config["tools"]
                )
            )
            response_text = response.text
            if "[Search Query]:" in response_text:
                search_query = response_text.split("[Search Query]:")[1].strip()
            else:
                search_query = response_text.strip()
            search_query = search_query.replace('`', '').replace('"', '').strip()
            words = search_query.split()
            if len(words) > 5:
                search_query = " ".join(words[:5])
            print(f"LLM generated search query: {search_query}")
            return search_query
        except Exception as e:
            print(f"Error generating query with LLM: {e}")
            if "image_search_prompt not found" in str(e):
                raise
            return self._optimize_query_heuristic(query)
    
    def _optimize_query_heuristic(self, query: str) -> str:
        query = re.sub(r'<[^>]+>', '', query).strip()
        if len(query.split()) > 10:
            nouns_and_entities = self._extract_key_entities(query)
            if len(nouns_and_entities) >= 3:
                base_query = " ".join(nouns_and_entities[:6])
            else:
                base_query = " ".join(query.split()[:6])
        else:
            base_query = query
        return base_query
    
    def _extract_key_entities(self, text: str) -> List[str]:
        words = text.split()
        entities = set()
        for i, word in enumerate(words):
            clean_word = word.strip(',.!?:;()[]{}"\'-')
            if not clean_word:
                continue
            is_sentence_start = i == 0 or words[i-1][-1] in '.!?'
            if clean_word[0].isupper() and not is_sentence_start:
                entities.add(clean_word)
        numbers = [w.strip(',.!?:;()[]{}"\'-') for w in words if w.strip(',.!?:;()[]{}"\'-').isdigit()]
        phrases = []
        for i in range(len(words) - 1):
            w1 = words[i].strip(',.!?:;()[]{}"\'-')
            w2 = words[i+1].strip(',.!?:;()[]{}"\'-')
            if len(w1) > 2 and len(w2) > 2:
                phrases.append(f"{w1} {w2}")
        key_terms = list(entities) + numbers
        if len(key_terms) < 3 and phrases:
            key_terms.extend(phrases[:3])
        if len(key_terms) < 4:
            long_words = [w.strip(',.!?:;()[]{}"\'-') for w in words if len(w.strip(',.!?:;()[]{}"\'-')) > 6]
            key_terms.extend(long_words[:5])
        return list(dict.fromkeys(key_terms))
    
    async def validate_image_url(self, session: aiohttp.ClientSession, image_data: Dict[str, str]) -> bool:
        url = image_data['url']
        # Add blacklist check
        blacklisted_domains = ['lookaside.instagram.com', 'gettyimages.com', 'shutterstock.com', 'istockphoto.com', 'tiktok.com/', 'fanatics.com',  'static.nike.com', 'c8.alamy.com', 'alamy.com', 'fanatics.frgimages.com']
        if any(domain in url.lower() for domain in blacklisted_domains):
            print(f"Blacklisted image URL: {url}")
            return False
            
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36"
            }
            async with session.get(url, allow_redirects=True, timeout=5, headers=headers) as response:
                if response.status != 200:
                    print(f"Invalid image URL (status {response.status}): {url}")
                    return False
                content_type = response.headers.get('Content-Type', '').lower()
                valid_types = ['image/', 'application/octet-stream']
                if not any(t in content_type for t in valid_types):
                    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.webp']
                    if not any(url.lower().endswith(ext) for ext in image_extensions):
                        print(f"Invalid content type ({content_type}): {url}")
                        return False
                print(f"Successfully validated URL: {url}")
                return True
        except Exception as e:
            print(f"Error validating image URL {url}: {str(e)}")
            # Fallback: check if URL ends with an image extension
            image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.webp']
            if any(url.lower().endswith(ext) for ext in image_extensions):
                return True
            return False

    def rank_image_by_relevance(self, image: Dict[str, str], content: str, query: str) -> float:
        score = 0.0
        content_words = set(re.findall(r'\b\w+\b', content.lower()))
        query_words = set(query.lower().split())
        image_title = image['title'].lower()
        title_words = set(re.findall(r'\b\w+\b', image_title))
        
        # Content and query matching
        content_match_count = len(title_words.intersection(content_words))
        score += content_match_count * 1.0
        query_match_count = len(title_words.intersection(query_words))
        score += query_match_count * 2.0
        
        for term in query_words:
            if term.lower() in image_title:
                score += 0.5
        
        # Resolution scoring
        width = image.get('width', 0)
        height = image.get('height', 0)
        if width > 0 and height > 0:
            # Check if image meets minimum resolution requirements
            meets_min_width = width >= 1200
            meets_min_height = height >= 400
            
            if meets_min_width and meets_min_height:
                score += 3.0  # High bonus for meeting minimum requirements
            else:
                # Still give some points for images close to the requirements
                width_ratio = min(width / 1200.0, 1.0)
                height_ratio = min(height / 400.0, 1.0)
                score += (width_ratio + height_ratio) * 0.5
            
            # Check aspect ratio
            aspect_ratio = width / max(height, 1)
            if 1.5 <= aspect_ratio <= 4.0:  # Preferred range for news article images
                score += 0.5
        
        return score

    async def rank_images(self, images: List[Dict[str, str]], content: str, query: str) -> List[Dict[str, str]]:
        if not images:
            return []
        scored_images = [(image, self.rank_image_by_relevance(image, content, query)) for image in images]
        scored_images.sort(key=lambda x: x[1], reverse=True)
        print("\nImage ranking results:")
        for i, (image, score) in enumerate(scored_images, 1):
            print(f"Rank {i}: '{image['title']}' - Score: {score:.2f}")
        return [image for image, _ in scored_images]
    
    async def _api_request_with_backoff(self, session: aiohttp.ClientSession, params: Dict, max_retries: int = 3) -> Optional[Dict]:
        retry_count = 0
        base_wait_time = 2
        while retry_count <= max_retries:
            if not self._check_rate_limit():
                print("Rate limit reached. Cannot make more API requests today.")
                return None
            try:
                async with session.get(self.base_url, params=params) as response:
                    self._use_request()
                    if response.status == 200:
                        return await response.json()
                    elif response.status == 429:
                        print(f"Rate limit exceeded (429). Retrying after backoff...")
                        response_text = await response.text()
                        print(f"Error details: {response_text[:200]}")
                        self.requests_remaining = 0
                        return None
                    else:
                        print(f"API request failed with status: {response.status}")
                        response_text = await response.text()
                        print(f"Response content: {response_text[:200]}")
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
        return None

    async def _search_images_ddgs(self, query: str, num_images: int = 3) -> List[Dict[str, str]]:
        try:
            print("Using DuckDuckGo Search as fallback...")
            filtered_results = []
            all_results = []
            cutoff_date = datetime.now() - timedelta(days=14)
            
            with DDGS() as ddgs:
                page = 1
                max_pages = 5
                while page <= max_pages:
                    total_requested = page * 10
                    results = list(ddgs.images(query, max_results=total_requested))
                    for result in results:
                        if result not in all_results:
                            all_results.append(result)
                    current_page_results = results[(page-1)*10 : page*10]
                    if not current_page_results:
                        break
                    
                    for img in current_page_results:
                        try:
                            if 'width' in img and 'height' in img and img['height'] > 0:
                                # Check for minimum resolution requirements
                                if img['width'] >= 1200 and img['height'] >= 400:
                                    filtered_results.append(img)
                                elif len(filtered_results) < num_images:
                                    # Only add lower resolution images if we haven't found enough high-res ones
                                    aspect = img['width'] / img['height']
                                    if 1.5 <= aspect <= 4.0:  # Reasonable aspect ratio for news images
                                        filtered_results.append(img)
                        except Exception as e:
                            continue
                    
                    if len(filtered_results) >= num_images:
                        break
                    page += 1
                
                # If we still don't have enough images, use the best ones we found
                if len(filtered_results) < num_images:
                    remaining_needed = num_images - len(filtered_results)
                    # Sort remaining images by resolution
                    remaining_images = [r for r in all_results if r not in filtered_results]
                    remaining_images.sort(key=lambda x: x.get('width', 0) * x.get('height', 0), reverse=True)
                    filtered_results.extend(remaining_images[:remaining_needed])
                
                return [{
                    'url': result.get('image', ''),
                    'title': result.get('title', ''),
                    'thumbnailUrl': result.get('thumbnail', ''),
                    'width': result.get('width', 0),
                    'height': result.get('height', 0)
                } for result in filtered_results[:num_images]]
        except Exception as e:
            print(f"DuckDuckGo search error: {e}")
            return []
    
    async def download_image(self, url: str, max_retries: int = 3) -> bytes:
        """Asynchronously download image bytes from a URL with improved error handling and retry logic."""
        retry_count = 0
        base_wait_time = 1
        
        while retry_count <= max_retries:
            try:
                timeout = aiohttp.ClientTimeout(total=10)  # 10 second timeout
                
                # Create a context that doesn't verify SSL certificates if needed
                # This isn't ideal for security but helps overcome certificate issues
                ssl_context = ssl.create_default_context(cafile=certifi.where())
                ssl_context.check_hostname = False
                ssl_context.verify_mode = ssl.CERT_NONE
                
                connector = aiohttp.TCPConnector(ssl=ssl_context)
                
                async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
                    headers = {
                        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36"
                    }
                    async with session.get(url, allow_redirects=True, headers=headers) as response:
                        if response.status != 200:
                            print(f"Failed to download image, status: {response.status}, URL: {url}")
                            if retry_count < max_retries:
                                wait_time = base_wait_time * (2 ** retry_count)
                                print(f"Retrying download in {wait_time} seconds...")
                                await asyncio.sleep(wait_time)
                                retry_count += 1
                                continue
                            else:
                                raise Exception(f"Failed to download image after {max_retries} retries, status: {response.status}")
                        
                        # Verify content type
                        content_type = response.headers.get('Content-Type', '').lower()
                        if not any(img_type in content_type for img_type in ['image/', 'application/octet-stream']):
                            print(f"Warning: Unexpected content type: {content_type} for URL: {url}")
                        
                        image_data = await response.read()
                        # Verify we got actual data
                        if len(image_data) < 100:  # Extremely small for an image, likely an error
                            print(f"Warning: Downloaded data too small ({len(image_data)} bytes) from URL: {url}")
                            if retry_count < max_retries:
                                wait_time = base_wait_time * (2 ** retry_count)
                                print(f"Retrying download in {wait_time} seconds...")
                                await asyncio.sleep(wait_time)
                                retry_count += 1
                                continue
                            
                        print(f"Successfully downloaded {len(image_data)} bytes from {url}")
                        return image_data
            except Exception as e:
                print(f"Error downloading image from {url}: {str(e)}")
                if retry_count < max_retries:
                    wait_time = base_wait_time * (2 ** retry_count)
                    print(f"Retrying download in {wait_time} seconds...")
                    await asyncio.sleep(wait_time)
                    retry_count += 1
                else:
                    raise Exception(f"Failed to download image after {max_retries} retries: {str(e)}")
        
        raise Exception(f"Failed to download image after {max_retries} retries")
    
    def upload_image_to_supabase(self, image_bytes: bytes, destination_path: str) -> str:
        """
        Synchronously upload the image bytes to Supabase Storage with improved error handling.
        This function is wrapped using asyncio.to_thread to avoid blocking.
        """
        bucket_name = 'images'  # Define bucket_name at the start of the function
        content_type = "image/jpeg"
        print(f"Using content type: {content_type} for upload to {destination_path}")
        
        try:
            # Upload the file.
            response = self.supabase.storage.from_(bucket_name).upload(
                path=destination_path,
                file=image_bytes,
                file_options={"contentType": content_type}
            )
            
            # Handle cases where the client might return an error in the response object
            # instead of raising an exception (less common for 409 with supabase-py v2+).
            if isinstance(response, dict) and response.get("error"):
                error_message_from_response = str(response.get("error"))
                # This is a fallback, primary handling is in the except block
                if "Duplicate" in error_message_from_response and "409" in error_message_from_response:
                    print(f"Interpreting response object error as duplicate for {destination_path}. Using existing public URL.")
                    supabase_url = os.environ.get("SUPABASE_URL", "https://yqtiuzhedkfacwgormhn.supabase.co").rstrip('/')
                    public_url = f"{supabase_url}/storage/v1/object/public/{bucket_name}/{destination_path}"
                    return public_url
                print(f"Supabase upload error in response object: {error_message_from_response}")
                raise Exception(f"Upload failed with error in response object: {error_message_from_response}")

            # If no exception and no error in response, construct public URL
            supabase_url = os.environ.get("SUPABASE_URL", "https://yqtiuzhedkfacwgormhn.supabase.co").rstrip('/')
            public_url = f"{supabase_url}/storage/v1/object/public/{bucket_name}/{destination_path}"
            print(f"Successful upload to Supabase storage: {public_url}")
            return public_url
            
        except Exception as e: # Catches SupabaseAPIError, StorageAPIError, etc.
            handled_as_duplicate = False
            
            # Primary check: e.message is the dictionary with specific Supabase error details.
            # Based on logs: e.message is {'message': 'The resource already exists', 'error': 'Duplicate', 'statusCode': '409'}
            if hasattr(e, 'message') and isinstance(e.message, dict):
                details = e.message
                error_type = details.get('error')
                status_code = details.get('statusCode')

                if isinstance(error_type, str) and error_type.lower() == 'duplicate' and \
                   isinstance(status_code, str) and status_code == '409':
                    
                    print(f"CONFIRMED DUPLICATE: Image already exists at {destination_path} (Supabase error via e.message: Duplicate, Status: 409). Constructing and returning existing public URL.")
                    supabase_url = os.environ.get("SUPABASE_URL", "https://yqtiuzhedkfacwgormhn.supabase.co").rstrip('/')
                    public_url = f"{supabase_url}/storage/v1/object/public/{bucket_name}/{destination_path}"
                    return public_url # Return the URL of the existing resource
                else:
                    print(f"Debug: e.message was a dict, but not the expected 409 Duplicate. Details: error='{error_type}', statusCode='{status_code}'")

            if not handled_as_duplicate:
                # If not handled as a duplicate via e.message, log extensively and re-raise.
                # This block will execute if the error is not the specific 409 Duplicate we're targeting,
                # or if e.message was not the expected dictionary.
                print(f"UNHANDLED EXCEPTION during Supabase upload (type: {type(e)}): {str(e)}")
                if hasattr(e, 'status_code'): # For some exception types that have a direct status_code
                    print(f"  Exception's direct status_code: {e.status_code}")
                
                # Log details from e.response if available (common for HTTP client errors)
                if hasattr(e, 'response'):
                    if hasattr(e.response, 'status_code'):
                         print(f"  e.response.status_code: {e.response.status_code}")
                    if hasattr(e.response, 'text'): 
                        response_text = e.response.text
                        print(f"  e.response.text: {response_text[:500] if response_text else '[empty response text]'}")
                
                # Log e.message content again, as it's key
                if hasattr(e, 'message'):
                    message_content = e.message
                    try:
                        message_str = str(message_content) # This might be the dict or a string
                        if len(message_str) > 1000: message_str = message_str[:1000] + "..." # Increased length for dicts
                    except Exception as str_conv_err:
                        message_str = f"[Could not convert e.message to string: {str_conv_err}]"
                    print(f"  Log: e.message content: {message_str}")
                
                # Log e.args, as it might contain the error dictionary or status
                if hasattr(e, 'args') and e.args:
                    try:
                        args_str = str(e.args)
                        if len(args_str) > 1000: args_str = args_str[:1000] + "..."
                    except Exception as str_conv_err:
                        args_str = f"[Could not convert e.args to string: {str_conv_err}]"
                    print(f"  Log: e.args content: {args_str}")
                
                raise # Re-raise the original error if not handled as the specific duplicate case
    
    async def download_and_upload_image(self, image_url: str, destination_path: str) -> str:
        """Combine download and upload to return the Supabase public URL with improved error handling."""
        try:
            print(f"Downloading image from: {image_url}")
            image_bytes = await self.download_image(image_url)
            print(f"Successfully downloaded {len(image_bytes)} bytes")
            
            uploaded_url = await asyncio.to_thread(self.upload_image_to_supabase, image_bytes, destination_path)
            print(f"Successfully uploaded to: {uploaded_url}")
            
            return uploaded_url
        except Exception as e:
            print(f"Error in download_and_upload_image: {str(e)}")
            # Re-raise to be handled by the process_images method
            raise
    
    async def _process_images(self, images: List[Dict[str, str]], num_images: int) -> List[Dict[str, str]]:
        """
        For the top ranked images, download and upload each image to Supabase Storage,
        replacing the original URL with the new public URL.
        
        Improved with better error handling and fallback mechanisms.
        """
        processed_images = []
        successful_uploads = 0
        
        # Try processing each image, with fallbacks if needed
        for idx, image in enumerate(images[:num_images], 1):
            original_url = image['url']
            # Create a unique destination path for each image
            # Use jpg extension and public folder to comply with RLS
            hash_digest = hashlib.md5(original_url.encode()).hexdigest()
            destination_path = f"public/{hash_digest}_img{idx}.jpg"
            
            try:
                print(f"Processing image {idx}/{num_images}: {original_url[:60]}...")
                new_url = await self.download_and_upload_image(original_url, destination_path)
                
                # Replace with the new URL on success
                image['url'] = new_url
                image['original_url'] = original_url  # Store the original URL as reference
                image['processed'] = True
                successful_uploads += 1
                print(f"✅ Image {idx}/{num_images} successfully processed and uploaded to Supabase")
                
            except Exception as e:
                print(f"❌ Failed to process image {idx}/{num_images} ({original_url[:60]}...): {str(e)}")
                
                # Construct the expected Supabase URL even on failure
                supabase_url_env = os.environ.get("SUPABASE_URL", "https://yqtiuzhedkfacwgormhn.supabase.co").rstrip('/')
                bucket_name = 'images'  # As used in upload_image_to_supabase
                expected_supabase_url = f"{supabase_url_env}/storage/v1/object/public/{bucket_name}/{destination_path}"
                
                image['url'] = expected_supabase_url
                image['original_url'] = original_url  # For reference
                image['processed'] = False # Mark as not successfully processed
                
                print(f"    Image processing failed. Storing expected Supabase URL: {expected_supabase_url}")
                print(f"    Original URL was: {original_url[:60]}...")
                print("    Note: This image was NOT successfully uploaded to Supabase.")
                
                # We strictly avoid using thumbnails as requested
                print("    Not attempting with thumbnail URL as per requirements")
            
            processed_images.append(image)
        
        print(f"Image processing summary: {successful_uploads}/{len(processed_images)} images successfully processed")
        if successful_uploads == 0 and processed_images:
            print("⚠️ WARNING: All image processing attempts failed. Check Supabase credentials and bucket permissions.")
            
        return processed_images

    async def search_images(self, query: str, num_images: int = 3, content: str = None) -> List[Dict[str, str]]:
        content_for_ranking = content or query
        if self.use_llm:
            optimized_query = await self._optimize_query_with_llm(query)
        else:
            optimized_query = self._optimize_query_heuristic(query)
            
        print(f"Original query: {query[:100]}...")
        print(f"Optimized query: {optimized_query}")
        
        params = {
            'key': self.api_key,
            'cx': self.custom_search_id,
            'q': optimized_query,
            'searchType': 'image',
            'num': min(num_images * 2, 10),
            'safe': 'active',
            'imgSize': 'huge',  # Prefer large images
            'rights': 'cc_publicdomain,cc_attribute,cc_sharealike',  # Ensure proper licensing
        }
        
        ssl_context = ssl.create_default_context(cafile=certifi.where())
        connector = aiohttp.TCPConnector(ssl=ssl_context)
        
        async with aiohttp.ClientSession(connector=connector) as session:
            try:
                data = await self._api_request_with_backoff(session, params)
                if not data:
                    print("API request failed or rate-limited. Trying without size restriction...")
                    # Retry without size restriction
                    params.pop('imgSize')
                    data = await self._api_request_with_backoff(session, params)
                    
                if not data and self.ddgs_enabled:
                    print("Google API request failed. Trying DuckDuckGo fallback...")
                    ddgs_results = await self._search_images_ddgs(optimized_query, num_images)
                    if ddgs_results:
                        validation_tasks = [self.validate_image_url(session, img) for img in ddgs_results]
                        validation_results = await asyncio.gather(*validation_tasks)
                        valid_images = [img for img, is_valid in zip(ddgs_results, validation_results) if is_valid]
                        if valid_images:
                            ranked_images = await self.rank_images(valid_images, content_for_ranking, optimized_query)
                            return await self._process_images(ranked_images, num_images)
                    return []

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
                    
                    print(f"Found {len(images)} images from API")
                    print("Validating image URLs...")
                    validation_tasks = [self.validate_image_url(session, img) for img in images]
                    validation_results = await asyncio.gather(*validation_tasks)
                    valid_images = [img for img, is_valid in zip(images, validation_results) if is_valid]
                    
                    print(f"After validation: {len(valid_images)} valid images out of {len(images)} total")
                    if valid_images:
                        ranked_images = await self.rank_images(valid_images, content_for_ranking, optimized_query)
                        return await self._process_images(ranked_images, num_images)
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
