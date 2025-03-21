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

async def fetch_primary_sources() -> List[int]:
    """
    Fetches all primary news sources from the NewsSource table.
    
    Returns:
        List[int]: List of source IDs that have isPrimarySource set to True
    """
    try:
        print("Fetching primary news sources...")
        headers = {
            "apikey": supabase_key,
            "Authorization": f"Bearer {supabase_key}"
        }
        url = f"{supabase_url}/rest/v1/NewsSource"
        params = {
            "select": "id",
            "isPrimarySource": "eq.true"
        }
        
        response = requests.get(url, headers=headers, params=params)
        
        if response.status_code == 200:
            sources = response.json()
            source_ids = [source["id"] for source in sources]
            print(f"Successfully fetched {len(source_ids)} primary sources: {source_ids}")
            return source_ids
        else:
            print(f"API request failed with status code: {response.status_code}")
            print(f"Response: {response.text}")
            return []
            
    except Exception as e:
        print(f"Error fetching primary sources: {e}")
        return []

async def fetch_unprocessed_articles() -> List[Dict]:
    """
    Fetches articles from the SourceArticles table that meet the criteria:
    - From primary sources (isPrimarySource = true in NewsSource table)
    - Have contentType 'news_article'
    - isArticleCreated is false or null
    
    Returns:
        List[Dict]: List of articles meeting the criteria
    """
    try:
        # Get primary sources
        primary_sources = await fetch_primary_sources()
        if not primary_sources:
            print("No primary sources found, using fallback sources [1, 2, 4]")
            primary_sources = [1, 2, 4]  # Fallback to original sources if fetch fails
            
        print("Fetching unprocessed articles from database...")
        headers = {
            "apikey": supabase_key,
            "Authorization": f"Bearer {supabase_key}"
        }

        url = f"{supabase_url}/rest/v1/SourceArticles"
        
        # Convert list of IDs to comma-separated string for the in.() operator
        source_list = f"({','.join(map(str, primary_sources))})"
        
        params = {
            "select": "*",
            "source": f"in.{source_list}",
            "contentType": "eq.news_article",
            "isArticleCreated": "eq.false"
        }
        
        response = requests.get(url, headers=headers, params=params)
        
        if response.status_code == 200:
            articles = response.json()
            # Double-check the filters in Python just to be safe
            filtered_articles = [
                article for article in articles 
                if article.get('source') in primary_sources 
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

async def update_articles_updated_by(article_id: int, source_article_id: int) -> bool:
    """
    Updates the UpdatedBy field for articles that are updated by the current article.
    
    Args:
        article_id (int): The ID of the current article (the one doing the updating)
        source_article_id (int): The SourceArticle ID to check for updates
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Get the updated articles from ArticleVector table
        response = supabase.table("ArticleVector").select("update").eq("SourceArticle", source_article_id).execute()
        
        if response.data and response.data[0].get("update"):
            updated_articles = response.data[0]["update"]
            if updated_articles:
                # Update all articles that are being updated
                for updated_source_id in updated_articles:
                    # Find the NewsArticle with this SourceArticle ID
                    update_response = supabase.table("NewsArticles").update(
                        {"UpdatedBy": article_id}
                    ).eq("SourceArticle", updated_source_id).execute()
                    
                    if not update_response.data:
                        print(f"Warning: Could not update UpdatedBy for article with SourceArticle {updated_source_id}")
        
        return True
    except Exception as e:
        print(f"Error updating UpdatedBy field: {e}")
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
        
        # Insert the article
        response = supabase.table("NewsArticles").insert({
            "created_at": article_data["created_at"],
            "headlineEnglish": article_data["headlineEnglish"],
            "headlineGerman": article_data["headlineGerman"],
            "SummaryEnglish": article_data["SummaryEnglish"],
            "SummaryGerman": article_data["SummaryGerman"],
            "ContentEnglish": article_data["ContentEnglish"],
            "ContentGerman": article_data["ContentGerman"],
            "Image1": article_data["Image1"],
            "Image2": article_data["Image2"],
            "Image3": article_data["Image3"],
            "SourceArticle": article_data["SourceArticle"],
            "team": article_data.get("team", None),
            "isUpdate": is_update
        }).execute()

        if not response.data:
            return False

        # If article was successfully inserted and it's an update, update the UpdatedBy field
        # for articles that this one updates
        article_id = response.data[0]["id"]
        if is_update:
            await update_articles_updated_by(article_id, article_data["SourceArticle"])

        return True
    except Exception as e:
        print(f"Error inserting article: {e}")
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