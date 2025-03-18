import os
import requests
from typing import List, Dict
from dotenv import load_dotenv
from supabase import create_client, Client

# Load environment variables
load_dotenv()

# Initialize Supabase client
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(supabase_url, supabase_key)

async def fetch_unprocessed_articles() -> List[Dict]:
    """
    Fetches articles from the SourceArticles table that meet the criteria:
    - From source 1, 2, or 4
    - Have contentType 'news_article'
    - isArticleCreated is false or null
    
    Returns:
        List[Dict]: List of articles meeting the criteria
    """
    try:
        print("Fetching unprocessed articles from database...")
        headers = {
            "apikey": supabase_key,
            "Authorization": f"Bearer {supabase_key}"
        }

        url = f"{supabase_url}/rest/v1/SourceArticles"
        params = {
            "select": "*",
            "source": "in.(1,2,4)",
            "contentType": "eq.news_article",
            "isArticleCreated": "eq.false"
        }
        
        response = requests.get(url, headers=headers, params=params)
        
        if response.status_code == 200:
            articles = response.json()
            # Double-check the filters in Python just to be safe
            filtered_articles = [
                article for article in articles 
                if article.get('source') in [1, 2, 4] 
                and article.get('contentType') == 'news_article'
                and not article.get('isArticleCreated', False)
            ]
            print(f"Successfully fetched {len(filtered_articles)} unprocessed articles")
            return filtered_articles
        else:
            print(f"API request failed with status code: {response.status_code}")
            print(f"Response: {response.text}")
            return []
            
    except Exception as e:
        print(f"Error fetching articles: {e}")
        return []

async def fetch_teams() -> List[Dict]:
    """
    Fetches all teams from the Teams table.
    
    Returns:
        List[Dict]: List of teams with their IDs and fullNames
    """
    try:
        print("Fetching teams from database...")
        headers = {
            "apikey": supabase_key,
            "Authorization": f"Bearer {supabase_key}"
        }
        url = f"{supabase_url}/rest/v1/Teams"
        params = {
            "select": "id,fullName"  # Updated to use fullName instead of name
        }
        
        response = requests.get(url, headers=headers, params=params)
        
        if response.status_code == 200:
            teams = response.json()
            print(f"Successfully fetched {len(teams)} teams")
            return teams
        else:
            print(f"API request failed with status code: {response.status_code}")
            print(f"Response: {response.text}")
            return []
            
    except Exception as e:
        print(f"Error fetching teams: {e}")
        return []

async def mark_article_as_processed(article_id: int) -> bool:
    """
    Marks an article as processed in the database by setting isArticleCreated to true.
    
    Args:
        article_id (int): The ID of the article to mark as processed
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        data = {"isArticleCreated": True}
        response = supabase.table("SourceArticles").update(data).eq("id", article_id).execute()
        return True
    except Exception as e:
        print(f"Error marking article {article_id} as processed: {e}")
        return False

async def insert_processed_article(article_data: Dict) -> bool:
    """
    Inserts a processed article into the NewsArticles table.
    
    Args:
        article_data (Dict): Dictionary containing the article data
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        response = supabase.table("NewsArticles").insert({
            "created_at": article_data["created_at"],
            "headlineEnglish": article_data["headlineEnglish"],
            "headlineGerman": article_data["headlineGerman"],
            "ContentEnglish": article_data["ContentEnglish"],
            "ConetentGerman": article_data["ContentGerman"],  # Fixed typo in column name
            "Image1": article_data["Image1"],
            "Image2": article_data["Image2"],
            "Image3": article_data["Image3"],
            "SourceArticle": article_data["SourceArticle"],
            "team": article_data.get("team", None)  # Add the team field
        }).execute()
        return True
    except Exception as e:
        print(f"Error inserting processed article: {e}")
        return False