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

async def check_for_updates(source_article_id: int) -> bool:
    """
    Checks if the article should be marked as an update by looking up the ArticleVector table.
    
    Args:
        source_article_id (int): The SourceArticle ID to check
        
    Returns:
        bool: True if this article should be marked as an update
    """
    try:
        response = supabase.table("ArticleVector").select("update").eq("SourceArticle", source_article_id).execute()
        if response.data:
            article_vector = response.data[0]
            return bool(article_vector.get("update"))  # Return True if updates array is not empty
        return False
    except Exception as e:
        print(f"Error checking for updates: {e}")
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
        # Check if this article is an update
        is_update = await check_for_updates(article_data["SourceArticle"])
        
        response = supabase.table("NewsArticles").insert({
            "created_at": article_data["created_at"],
            "headlineEnglish": article_data["headlineEnglish"],
            "headlineGerman": article_data["headlineGerman"],
            "ContentEnglish": article_data["ContentEnglish"],
            "ConetentGerman": article_data["ContentGerman"],
            "Image1": article_data["Image1"],
            "Image2": article_data["Image2"],
            "Image3": article_data["Image3"],
            "SourceArticle": article_data["SourceArticle"],
            "team": article_data.get("team", None),
            "isUpdate": is_update
        }).execute()
        return True
    except Exception as e:
        print(f"Error inserting processed article: {e}")
        return False

async def batch_update_article_status() -> Dict:
    """
    Batch processes all existing articles in NewsArticles table to update their isUpdate status
    based on ArticleVector table data.
    
    Returns:
        Dict: Statistics about the operation (total processed, updated count, errors)
    """
    try:
        stats = {"total": 0, "updated": 0, "errors": 0}
        
        # Fetch all articles from NewsArticles table
        response = supabase.table("NewsArticles").select("id,SourceArticle,isUpdate").execute()
        
        if not response.data:
            print("No articles found in NewsArticles table")
            return stats
            
        articles = response.data
        stats["total"] = len(articles)
        print(f"Processing {len(articles)} articles...")
        
        # Process each article
        for article in articles:
            try:
                # Check ArticleVector table
                vector_response = supabase.table("ArticleVector").select("update").eq("SourceArticle", article["SourceArticle"]).execute()
                
                if vector_response.data:
                    article_vector = vector_response.data[0]
                    should_be_update = bool(article_vector.get("update"))
                    
                    # Only update if the status needs to change
                    if article.get("isUpdate") != should_be_update:
                        update_response = supabase.table("NewsArticles").update(
                            {"isUpdate": should_be_update}
                        ).eq("id", article["id"]).execute()
                        
                        if update_response.data:
                            stats["updated"] += 1
                            print(f"Updated article {article['id']} isUpdate status to {should_be_update}")
                
            except Exception as e:
                print(f"Error processing article {article['id']}: {e}")
                stats["errors"] += 1
                
        print(f"Batch processing complete. Stats: {stats}")
        return stats
        
    except Exception as e:
        print(f"Error in batch processing: {e}")
        return {"total": 0, "updated": 0, "errors": 1}