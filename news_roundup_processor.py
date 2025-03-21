import asyncio
import json
import os
from typing import Dict, List, Optional, Tuple
import re
import google.generativeai as genai
from dotenv import load_dotenv
import requests
from LLMSetup import initialize_model
import sys
from team_classifier import classify_team
from articleImage import ImageSearcher

# Load environment variables
load_dotenv()

# Initialize Supabase client
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_KEY")

# Initialize Gemini models - flash for content, lite for translation
content_model_info = initialize_model("gemini", "flash")
translation_model_info = initialize_model("gemini", "lite")
content_model = content_model_info["model"]
translation_model = translation_model_info["model"]

def count_words(text: str) -> int:
    """Count words in a text string"""
    return len(text.split())

def truncate_to_word_limit(text: str, limit: int = 70) -> str:
    """
    Truncate text to specified word limit while maintaining sentence integrity
    
    Args:
        text: Text to truncate
        limit: Maximum number of words allowed
        
    Returns:
        Truncated text
    """
    words = text.split()
    if len(words) <= limit:
        return text
        
    # Get the first 70 words
    truncated = ' '.join(words[:limit])
    
    # Find the last complete sentence
    sentences = re.split(r'(?<=[.!?])\s+', truncated)
    if len(sentences) > 1:
        # Remove incomplete last sentence
        return ' '.join(sentences[:-1])
    
    return truncated + "..."

async def fetch_news_roundups() -> List[Dict]:
    """
    Fetch articles from the SourceArticles table with ContentType 'news-round-up'
    that haven't been processed yet
    """
    try:
        print("Fetching news round-up articles...")
        headers = {
            "apikey": supabase_key,
            "Authorization": f"Bearer {supabase_key}"
        }
        url = f"{supabase_url}/rest/v1/SourceArticles"
        params = {
            "select": "*",
            "contentType": "eq.news-round-up",
            "isArticleCreated": "eq.false"
        }
        
        response = requests.get(url, headers=headers, params=params)
        
        if response.status_code == 200:
            articles = response.json()
            print(f"Successfully fetched {len(articles)} news round-up articles")
            return articles
        else:
            print(f"API request failed with status code: {response.status_code}")
            print(f"Response: {response.text}")
            return []
            
    except Exception as e:
        print(f"Error fetching news round-up articles: {e}")
        return []

async def extract_topics(content: str) -> List[Dict]:
    """
    Extract topics from a news round-up article using the flash model
    Returns a list of topics with their content
    """
    prompt = f"""
    You are a sports news analyst. The following text contains a round-up of multiple sports news topics.
    
    ### Instructions:  
        Analyze the text and extract each distinct topic. For each topic:  

        1. **Identify** where a new topic starts and ends.  
        2. **Provide** a **concise yet comprehensive** summary in **no more than 70 words**.  

        ### Requirements:  
        - Be **specific and factual**—name the **exact** teams, players, events, and outcomes if available.  
        - Avoid vague phrasing like *"a team won a game"*—instead, specify *"The Kansas City Chiefs defeated the San Francisco 49ers 31-20 in Super Bowl LIV."*  
        - Ensure the summary is **self-contained**, covering the most critical facts and implications.  
        - **Do not** include any personal opinions or subjective language.    
        
    Return your analysis ONLY as a JSON array where each object represents a topic:
        
    [
      {{
        "information": "One paragraph summary of topic 1 (max 70 words)"
      }},
      {{
        "information": "One paragraph summary of topic 2 (max 70 words)"
      }}
    ]
    
    Here's the text to analyze:
    {content}
    """
    
    try:
        response = await asyncio.to_thread(
            lambda: content_model.generate_content(
                prompt, 
                generation_config=genai.types.GenerationConfig(
                    temperature=0.2,
                    max_output_tokens=8192,
                )
            )
        )
        
        raw_response = response.text
        print("\nRaw topic extraction response:")
        print(raw_response[:500] + "..." if len(raw_response) > 500 else raw_response)
        
        # Clean the response to extract JSON
        # Remove code blocks if present
        if raw_response.strip().startswith("```"):
            lines = raw_response.strip().splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            raw_response = "\n".join(lines)
        
        # Find JSON array
        json_start = raw_response.find("[")
        json_end = raw_response.rfind("]") + 1
        
        if json_start != -1 and json_end != -1:
            json_text = raw_response[json_start:json_end]
            topics = json.loads(json_text)
            
            # Enforce word limit on each topic
            for topic in topics:
                if "information" in topic:
                    word_count = count_words(topic["information"])
                    if word_count > 70:
                        topic["information"] = truncate_to_word_limit(topic["information"])
                        print(f"Truncated topic from {word_count} to {count_words(topic['information'])} words")
            
            print(f"Successfully extracted {len(topics)} topics")
            return topics
        else:
            print("Failed to extract JSON from response")
            return []
            
    except Exception as e:
        print(f"Error extracting topics: {e}")
        return []

async def determine_team(topic_info: str) -> Tuple[Optional[int], str]:
    """
    Determine which team a topic is about
    Returns the team ID and the team name
    """
    try:
        # Create a dummy headline for the classifier
        headline = "News Roundup Item"
        
        # Get team classification
        team_name, confidence = await classify_team(headline, topic_info)
        
        # Format confidence properly for printing
        confidence_str = f"{confidence:.2f}" if confidence is not None else "0.00"
        
        if not team_name or confidence < 0.7 or team_name.lower() == "unknown":
            print(f"No confident team match: {team_name} (confidence: {confidence_str})")
            return None, "Unknown"
        
        # Fetch all teams to get the ID
        headers = {
            "apikey": supabase_key,
            "Authorization": f"Bearer {supabase_key}"
        }
        response = requests.get(
            f"{supabase_url}/rest/v1/Teams",
            headers=headers,
            params={"select": "id,fullName"}
        )
        
        if response.status_code != 200:
            print(f"Error fetching teams: {response.status_code}")
            return None, team_name
        
        teams_data = response.json()
        
        # Find matching team in database
        for team in teams_data:
            db_team_name = team.get('fullName', '').lower()
            if db_team_name and (
                team_name.lower() in db_team_name or 
                db_team_name in team_name.lower()
            ):
                team_id = team['id']
                print(f"Matched team: {team.get('fullName')} (ID: {team_id})")
                return team_id, team_name
        
        print(f"Could not match '{team_name}' to any team in database")
        return None, team_name
        
    except Exception as e:
        print(f"Error determining team: {e}")
        return None, "Unknown"

async def mark_article_as_processed(article_id: int) -> bool:
    """Mark an article as processed in the database"""
    try:
        headers = {
            "apikey": supabase_key,
            "Authorization": f"Bearer {supabase_key}",
            "Content-Type": "application/json",
            "Prefer": "return=minimal"
        }
        response = requests.patch(
            f"{supabase_url}/rest/v1/SourceArticles",
            headers=headers,
            params={"id": f"eq.{article_id}"},
            json={"isArticleCreated": True}
        )
        return response.status_code in [200, 204]
    except Exception as e:
        print(f"Error marking article {article_id} as processed: {e}")
        return False

async def translate_to_german(info: str, max_retries: int = 3) -> str:
    """
    Translate the English information paragraph to German using the lite model
    
    Args:
        info: The original English information paragraph
        max_retries: Maximum number of retry attempts
        
    Returns:
        German translation of the information paragraph
    """
    prompt = f"""
    Translate the following English sports news paragraph to German. 
    Maintain the same level of detail and professional sports journalism style.
    Make sure the translation is natural and fluent German, not a literal word-for-word translation.
    IMPORTANT: 
    1. Respond ONLY with the German translation, no explanations or other text
    2. The translation must not exceed 70 words
    3. Keep the same key information as the English text
    
    English paragraph:
    {info}
    """
    
    base_wait_time = 2  # Initial wait time in seconds
    
    for attempt in range(max_retries):
        try:
            # Add delay between retries with exponential backoff
            if attempt > 0:
                wait_time = base_wait_time * (2 ** (attempt - 1))
                print(f"Retry attempt {attempt + 1}, waiting {wait_time} seconds...")
                await asyncio.sleep(wait_time)
            
            response = await asyncio.to_thread(
                lambda: translation_model.generate_content(
                    prompt, 
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.2,
                        max_output_tokens=1024,
                    )
                )
            )
            
            if not response or not response.text:
                print(f"Empty response on attempt {attempt + 1}")
                continue
                
            german_info = response.text.strip()
            
            # Clean up any potential markdown or thinking process
            if german_info.startswith("```") or german_info.startswith("<"):
                # Try to find the actual translation content
                lines = german_info.split("\n")
                clean_lines = []
                
                for line in lines:
                    # Skip markdown code blocks, thinking process markers, and explanatory text
                    if any(marker in line for marker in ["```", "<", "**", "#", "Here's", "Let's", "Translation:", "Note:", "1.", "2.", "->", "German:"]):
                        continue
                    
                    # Keep actual content
                    if line.strip() and not line.strip().startswith("-"):
                        clean_lines.append(line)
                
                german_info = "\n".join(clean_lines).strip()
            
            # Enforce word limit
            word_count = count_words(german_info)
            if word_count > 70:
                german_info = truncate_to_word_limit(german_info)
                print(f"Truncated German translation from {word_count} to {count_words(german_info)} words")
            
            # Verify the translation looks valid (contains German-specific characters)
            if any(char in german_info for char in 'äöüßÄÖÜ') or 'die ' in german_info.lower() or 'der ' in german_info.lower() or 'das ' in german_info.lower():
                print(f"Generated German translation ({len(german_info)} chars, {count_words(german_info)} words)")
                return german_info
            else:
                print("Translation appears invalid, retrying...")
                continue
                
        except Exception as e:
            error_msg = str(e)
            print(f"Error on translation attempt {attempt + 1}: {error_msg}")
            
            # Check for specific error types
            if "429" in error_msg or "quota" in error_msg.lower() or "rate" in error_msg.lower():
                # If it's our last retry and we hit rate limit, wait longer
                if attempt == max_retries - 1:
                    print("Rate limit hit on final retry, waiting 30 seconds...")
                    await asyncio.sleep(30)
                continue
                
            # For other errors, if it's the last retry, return empty string
            if attempt == max_retries - 1:
                return ""
    
    print("All translation attempts failed")
    return ""

async def find_topic_image(topic_info: str, image_searcher: ImageSearcher) -> Dict:
    """
    Find a relevant image for a topic
    
    Args:
        topic_info: The topic information
        image_searcher: The ImageSearcher instance
        
    Returns:
        Dictionary with image URL and other metadata
    """
    try:
        print("Searching for topic image...")
        # Use the image_searcher to find a relevant image
        images = await image_searcher.search_images(topic_info, num_images=1, content=topic_info)
        
        if not images or len(images) == 0:
            print("No images found for topic")
            return {}
        
        # Return the first (best) image
        image = images[0]
        print(f"Found image: {image['url']}")
        return {
            "url": image.get("url", ""),
            "title": image.get("title", ""),
            "width": image.get("width", 0),
            "height": image.get("height", 0)
        }
        
    except Exception as e:
        print(f"Error finding image: {e}")
        return {}

async def insert_into_newsticker(topic_result: Dict) -> bool:
    """
    Insert a topic into the NewsTicker table
    
    Args:
        topic_result: Dictionary containing the topic information
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        headers = {
            "apikey": supabase_key,
            "Authorization": f"Bearer {supabase_key}",
            "Content-Type": "application/json",
            "Prefer": "return=minimal"
        }
        
        # Prepare the data according to the NewsTicker schema
        data = {
            "SourceArticle": topic_result["sourceArticle"],
            "EnglishInformation": topic_result["informationEnglish"],
            "Team": topic_result["team"],
            "GermanInformation": topic_result["informationGerman"],
            "Image": topic_result["imageUrl"]
        }
        
        response = requests.post(
            f"{supabase_url}/rest/v1/NewsTicker",
            headers=headers,
            json=data
        )
        
        if response.status_code in [200, 201]:
            print(f"Successfully inserted topic into NewsTicker table")
            return True
        else:
            print(f"Failed to insert topic into NewsTicker table: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"Error inserting into NewsTicker: {e}")
        return False

async def process_news_roundup(article: Dict, image_searcher: ImageSearcher) -> List[Dict]:
    """Process a single news round-up article and extract topics"""
    article_id = article['id']
    content = article.get('Content', '')
    
    if not content:
        print(f"Skipping article ID {article_id} - no content found")
        return []
        
    print(f"\n{'='*80}")
    print(f"Processing news round-up article ID: {article_id}")
    print(f"Original Headline: {article.get('headline', '')}")
    print(f"{'='*80}")
    
    # Extract topics from the article
    topics = await extract_topics(content)
    
    if not topics:
        print(f"No topics found in article {article_id}")
        return []
    
    results = []
    
    # Process each topic
    for i, topic in enumerate(topics):
        print(f"\n--- Processing Topic {i+1} ---")
        topic_info = topic.get('information', '')
        
        # Get team for this topic
        team_id, team_name = await determine_team(topic_info)
        
        # Translate information to German
        german_info = await translate_to_german(topic_info)
        
        # Find a relevant image
        image_data = await find_topic_image(topic_info, image_searcher)
        
        # Create result object
        result = {
            "sourceArticle": article_id,
            "informationEnglish": topic_info,
            "informationGerman": german_info,
            "team": team_id,  # This can be null if no team identified
            "matchedTeam": team_name if team_name != "Unknown" else "",
            "imageUrl": image_data.get("url", ""),
            "imageTitle": image_data.get("title", "")
        }
        
        # Insert into NewsTicker table
        success = await insert_into_newsticker(result)
        if success:
            results.append(result)
            print(f"Topic {i+1} successfully inserted into NewsTicker")
        else:
            print(f"Failed to insert topic {i+1} into NewsTicker")
    
    # Mark the source article as processed only if at least one topic was processed
    if results:
        success = await mark_article_as_processed(article_id)
        if not success:
            print(f"Warning: Failed to mark article {article_id} as processed")
    
    return results

async def process_batch(batch: List[Dict], image_searcher: ImageSearcher) -> Tuple[int, int]:
    """
    Process a batch of articles
    
    Args:
        batch: List of articles to process
        image_searcher: ImageSearcher instance
        
    Returns:
        Tuple of (processed_count, topics_count)
    """
    batch_processed = 0
    batch_topics = 0
    
    for article in batch:
        try:
            results = await process_news_roundup(article, image_searcher)
            if results:
                batch_processed += 1
                batch_topics += len(results)
        except asyncio.CancelledError:
            print("\nProcess interrupted by user. Finishing current article...")
            raise
        except Exception as e:
            print(f"Error processing article {article.get('id')}: {e}")
            continue
    
    return batch_processed, batch_topics

async def main():
    # Fetch news round-up articles
    articles = await fetch_news_roundups()
    
    if not articles:
        print("No news round-up articles found to process")
        return
    
    # Initialize image searcher with LLM support    
    image_searcher = ImageSearcher(use_llm=True)
    
    processed_count = 0
    topics_count = 0
    batch_size = 3
    cooldown_time = 30  # seconds
    
    print(f"\nProcessing {len(articles)} articles in batches of {batch_size}")
    print(f"Will pause for {cooldown_time} seconds between batches")
    
    try:
        # Process articles in batches
        for i in range(0, len(articles), batch_size):
            batch = articles[i:i + batch_size]
            print(f"\n{'='*40}")
            print(f"Processing batch {i//batch_size + 1} ({len(batch)} articles)")
            print(f"{'='*40}")
            
            try:
                # Process the current batch
                batch_processed, batch_topics = await process_batch(batch, image_searcher)
                processed_count += batch_processed
                topics_count += batch_topics
                
                # Print batch results
                print(f"\nBatch {i//batch_size + 1} complete:")
                print(f"- Processed {batch_processed} articles")
                print(f"- Created {batch_topics} news ticker entries")
                print(f"Running totals: {processed_count} articles, {topics_count} entries")
                
                # If there are more articles to process, wait before next batch
                if i + batch_size < len(articles):
                    print(f"\nWaiting {cooldown_time} seconds before next batch...")
                    await asyncio.sleep(cooldown_time)
                    
            except asyncio.CancelledError:
                print("\nProcessing interrupted. Saving progress...")
                break
            except Exception as e:
                print(f"\nError processing batch: {e}")
                print("Moving to next batch...")
                continue
    
    except Exception as e:
        print(f"\nUnexpected error: {e}")
    finally:
        # Print final summary
        print("\n" + "="*40)
        print("PROCESSING COMPLETE")
        print("="*40)
        print(f"Processed {processed_count} articles")
        print(f"Created {topics_count} news ticker entries")
        
        if processed_count < len(articles):
            remaining = len(articles) - processed_count
            print(f"Note: {remaining} articles remaining to be processed")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nProcess terminated by user")
    except Exception as e:
        print(f"\nFatal error: {e}")