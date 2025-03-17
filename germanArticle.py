import asyncio
import json
import google.generativeai as genai
import sys
import os
import yaml
import requests
from dotenv import load_dotenv
from supabase import create_client, Client

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from LLMSetup import initialize_model

# Load environment variables
load_dotenv()

# Initialize Supabase client
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(supabase_url, supabase_key)

# Initialize Gemini model using LLMSetup
model_info = initialize_model("gemini")
gemini_model = model_info["model"]

# Load prompts from YAML file
with open(os.path.join(os.path.dirname(__file__), "prompts.yml"), "r", encoding="utf-8") as f:
    prompts = yaml.safe_load(f)

async def generate_german_article(main_content: str, verbose: bool = False) -> dict:
    """
    Generates a German article with headline and structured content.
    Returns a dict with 'headline' and 'content'.
    If verbose is True, includes the raw Gemini response in the result under 'raw_response'.
    """
    prompt = f"""
{prompts['german_prompt']}

**Source Information:**
Main content (main_content) – the central story:
{main_content}

Please provide your answer strictly in the following JSON format without any additional text:
{{
  "headline": "<h1>Your generated headline</h1>",
  "content": "<div>Your structured article content as HTML, including <p>, <h2>, etc.</div>"
}}
"""
    try:
        # Generate content with increased max_output_tokens.
        response_obj = await asyncio.to_thread(
            gemini_model.generate_content,
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.7,
                max_output_tokens=8192,
            )
        )
        raw_response = response_obj.text
        if verbose:
            print("Raw Gemini response:")
            print(raw_response)
    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        raw_response = ""

    # Remove markdown code block markers if present.
    if (raw_response.strip().startswith("```")):
        lines = raw_response.strip().splitlines()
        # Remove the first line (```json) and the last line if it's a markdown fence.
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        raw_response = "\n".join(lines)
        
    json_start = raw_response.find("{")
    json_end = raw_response.rfind("}") + 1
    if (json_start != -1 and json_end != -1):
        raw_response_clean = raw_response[json_start:json_end]
    else:
        raw_response_clean = raw_response  # Fallback if markers are not found

    try:
        # Parse the JSON response
        response_data = json.loads(raw_response_clean)
        
        # Build the result from the parsed data directly
        result = {
            "headline": response_data.get("headline", ""),
            "content": response_data.get("content", "")
        }
        if verbose:
            result["raw_response"] = raw_response_clean
        return result
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON response: {e}")
        # Fallback: Try to extract content using string manipulation
        try:
            headline_start = raw_response_clean.find('"headline": "') + 12
            headline_end = raw_response_clean.find('",', headline_start)
            content_start = raw_response_clean.find('"content": "') + 11
            content_end = raw_response_clean.rfind('"')
            
            headline = raw_response_clean[headline_start:headline_end]
            content = raw_response_clean[content_start:content_end]
            
            return {
                "headline": headline,
                "content": content,
                "raw_response": raw_response_clean if verbose else ""
            }
        except Exception as e:
            print(f"Fallback parsing failed: {e}")
            return {"headline": "", "content": "", "raw_response": raw_response_clean if verbose else ""}
    except Exception as e:
        print(f"Unknown error: {e}")
        return {"headline": "", "content": "", "raw_response": raw_response_clean if verbose else ""}

async def fetch_articles_to_process():
    """
    Fetches articles from the SourceArticles table that meet the criteria:
    - From source 1, 2, or 4
    - Have contentType 'news_article'
    - isArticleCreated is false or null
    """
    try:
        print("Fetching articles using direct REST API call...")
        headers = {
            "apikey": supabase_key,
            "Authorization": f"Bearer {supabase_key}"
        }

        # The API expects query parameters in a specific format for PostgREST
        url = f"{supabase_url}/rest/v1/SourceArticles"
        params = {
            "select": "*",
            # Filter for source in (1,2,4)
            "source": "in.(1,2,4)",
            # Filter for contentType = 'news_article'
            "contentType": "eq.news_article",
            # Filter for isArticleCreated = false
            "isArticleCreated": "is.null,eq.false"
        }
        
        print(f"Calling URL: {url} with params: {params}")
        response = requests.get(url, headers=headers, params=params)
        
        # Check if request was successful
        if response.status_code == 200:
            data = response.json()
            print(f"Successfully fetched {len(data)} articles")
            return data
        else:
            print(f"API request failed with status code: {response.status_code}")
            print(f"Response: {response.text}")
            return []
            
    except Exception as e:
        print(f"Error fetching articles: {e}")
        return []

async def main():
    # Fetch articles that meet our criteria
    articles = await fetch_articles_to_process()
    print(f"Found {len(articles)} articles matching criteria (source 1/2/4, contentType 'news_article', not processed)")
    
    german_articles = {}
    
    # Loop through each article and generate the German version
    for article in articles:
        article_id = article['id']
        main_content = article.get('Content', '')
        
        if not main_content:
            print(f"Skipping article ID {article_id} - no content found")
            continue
            
        print(f"Generating German article for article ID: {article_id}")
        
        try:
            article_data = await generate_german_article(main_content, verbose=True)
            headline = article_data.get("headline", "")
            content = article_data.get("content", "")
            
            # Store the results in memory
            german_articles[str(article_id)] = {
                "headline": headline,
                "content": content,
                "originalSource": article.get('source'),
                "originalHeadline": article.get('headline', ''),
                "originalUrl": article.get('url', '')
            }
            
        except Exception as e:
            print(f"[ERROR] Error generating German article for {article_id}: {e}")
            german_articles[str(article_id)] = {
                "headline": "", 
                "content": "",
                "error": str(e)
            }
    
    # Save the generated German articles to a JSON file
    with open("German_articles.json", "w", encoding="utf-8") as f:
        json.dump(german_articles, f, ensure_ascii=False, indent=2)
    
    print(f"German article generation complete. Processed {len(german_articles)} articles.")
    print(f"Results saved to German_articles.json")

if __name__ == "__main__":
    asyncio.run(main())