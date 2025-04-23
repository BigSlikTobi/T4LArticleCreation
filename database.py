import os
import requests
from typing import List, Dict, Optional, Set # Added Set
from dotenv import load_dotenv
from supabase import create_client, Client
# Add imports for datetime, timedelta, UUID if not already present
from datetime import datetime, timedelta, timezone
from uuid import UUID
import logging # Add logging


# Load environment variables
load_dotenv()

# --- Add Logging Setup ---
# Configure logging level and format
# Use filename and line number for better debugging context
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)

# Initialize Supabase client
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_KEY")

# Add type hinting for the client
supabase: Optional[Client] = None
if supabase_url and supabase_key:
    try:
        supabase = create_client(supabase_url, supabase_key)
        logging.info("Supabase client initialized successfully.")
    except Exception as e:
        logging.critical(f"Failed to initialize Supabase client: {e}", exc_info=True)
        # Depending on your application's needs, you might exit or raise an error here
else:
    logging.critical("Supabase URL or Key not found in environment variables. Database functions will fail.")


# Helper function to check if Supabase client is available
def _check_supabase_client():
    if supabase is None:
        logging.error("Supabase client is not initialized.")
        return False
    return True

async def fetch_primary_sources() -> List[int]:
    """
    Fetches all primary news sources from the NewsSource table.

    Returns:
        List[int]: List of source IDs that have isPrimarySource set to True
    """
    if not _check_supabase_client() or not supabase_key or not supabase_url:
        return []
    try:
        logging.info("Fetching primary news sources...")
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
            logging.info(f"Successfully fetched {len(source_ids)} primary sources: {source_ids}")
            return source_ids
        else:
            logging.error(f"API request failed fetching primary sources. Status: {response.status_code}, Response: {response.text}")
            return []

    except Exception as e:
        logging.error(f"Error fetching primary sources: {e}", exc_info=True)
        return []

async def fetch_unprocessed_articles() -> List[Dict]:
    """
    Fetches articles from the SourceArticles table that meet the criteria:
    - From primary sources (isPrimarySource = true in NewsSource table)
    - Have contentType 'news_article'
    - isArticleCreated is false or null
    - duplication_of is null or empty

    Returns:
        List[Dict]: List of articles meeting the criteria
    """
    if not _check_supabase_client() or not supabase_key or not supabase_url:
        return []
    try:
        # Get primary sources
        primary_sources = await fetch_primary_sources()
        if not primary_sources:
            logging.warning("No primary sources found, using fallback sources [1, 2, 4]")
            primary_sources = [1, 2, 4]  # Fallback to original sources if fetch fails

        logging.info("Fetching unprocessed articles from database...")
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
            "isArticleCreated": "eq.false",
            "duplication_of": "is.null"  # Add filter for duplication_of being null
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
                and (article.get('duplication_of') is None or article.get('duplication_of') == '')  # Additional Python-side check
            ]
            logging.info(f"Successfully fetched {len(filtered_articles)} unprocessed articles")
            return filtered_articles
        else:
            logging.error(f"API request failed fetching unprocessed articles. Status: {response.status_code}, Response: {response.text}")
            return []

    except Exception as e:
        logging.error(f"Error fetching articles: {e}", exc_info=True)
        return []

async def fetch_unprocessed_team_articles() -> List[Dict]:
    """
    Fetches articles from the TeamSourceArticles table that have isArticleCreated set to false
    and contentType == 'news_article'.

    Returns:
        List[Dict]: List of unprocessed team articles
    """
    if not _check_supabase_client() or not supabase_key or not supabase_url:
        return []
    try:
        logging.info("Fetching unprocessed team articles...")
        headers = {
            "apikey": supabase_key,
            "Authorization": f"Bearer {supabase_key}"
        }
        url = f"{supabase_url}/rest/v1/TeamSourceArticles"
        params = {
            "select": "*",
            "isArticleCreated": "eq.false",
            "contentType": "eq.news_article"
        }
        response = requests.get(url, headers=headers, params=params)
        if response.status_code == 200:
            articles = response.json()
            # Filter out articles that are not news articles
            filtered_articles = [
                article for article in articles
                if article.get('contentType') == 'news_article' and not article.get('isArticleCreated', False)
            ]
            logging.info(f"Successfully fetched {len(filtered_articles)} unprocessed team articles")
            return filtered_articles
        else:
            logging.error(f"API request failed fetching unprocessed team articles. Status: {response.status_code}, Response: {response.text}")
            return []
    except Exception as e:
        logging.error(f"Error fetching unprocessed team articles: {e}", exc_info=True)
        return []

async def fetch_teams() -> List[Dict]:
    """
    Fetches all teams from the Teams table.

    Returns:
        List[Dict]: List of teams with their IDs and fullNames
    """
    if not _check_supabase_client() or not supabase_key or not supabase_url:
        return []
    try:
        logging.info("Fetching teams from database...")
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
            logging.info(f"Successfully fetched {len(teams)} teams")
            return teams
        else:
            logging.error(f"API request failed fetching teams. Status: {response.status_code}, Response: {response.text}")
            return []

    except Exception as e:
        logging.error(f"Error fetching teams: {e}", exc_info=True)
        return []

async def mark_article_as_processed(article_id: int) -> bool:
    """
    Marks an article as processed in the SourceArticles table by setting isArticleCreated to true.

    Args:
        article_id (int): The ID of the article to mark as processed

    Returns:
        bool: True if successful, False otherwise
    """
    if not _check_supabase_client():
        return False
    try:
        logging.info(f"Marking SourceArticle {article_id} as processed.")
        data = {"isArticleCreated": True}
        response = supabase.table("SourceArticles").update(data).eq("id", article_id).execute()
        # Check response, update can return data even if 0 rows affected if select() is used.
        # A more direct check might be needed if the client allows checking affected rows.
        # For now, assume lack of error means success.
        if hasattr(response, 'error') and response.error:
            logging.error(f"Error marking article {article_id} as processed: {response.error}")
            return False
        logging.info(f"Successfully marked SourceArticle {article_id} as processed.")
        return True # Assuming success if no error attribute
    except Exception as e:
        logging.error(f"Error marking article {article_id} as processed: {e}", exc_info=True)
        return False

async def mark_team_article_as_processed(article_id: int) -> bool:
    """
    Marks a team article as processed in the TeamSourceArticles table by setting isArticleCreated to true.

    Args:
        article_id (int): The ID of the article to mark as processed
    Returns:
        bool: True if successful, False otherwise
    """
    if not _check_supabase_client():
        return False
    try:
        logging.info(f"Marking TeamSourceArticle {article_id} as processed.")
        data = {"isArticleCreated": True}
        response = supabase.table("TeamSourceArticles").update(data).eq("id", article_id).execute()
        if hasattr(response, 'error') and response.error:
            logging.error(f"Error marking team article {article_id} as processed: {response.error}")
            return False
        logging.info(f"Successfully marked TeamSourceArticle {article_id} as processed.")
        return True
    except Exception as e:
        logging.error(f"Error marking team article {article_id} as processed: {e}", exc_info=True)
        return False

async def check_for_updates(source_article_id: int) -> bool:
    """
    Checks if the article should be marked as an update by looking up the ArticleVector table.

    Args:
        source_article_id (int): The SourceArticle ID to check

    Returns:
        bool: True if this article should be marked as an update
    """
    if not _check_supabase_client():
        return False
    try:
        logging.debug(f"Checking for updates status for SourceArticle {source_article_id}.")
        response = supabase.table("ArticleVector").select("update").eq("SourceArticle", source_article_id).limit(1).execute()
        if response.data:
            article_vector = response.data[0]
            is_update = bool(article_vector.get("update")) # Check if 'update' list/field is non-empty/true
            logging.debug(f"Update status for {source_article_id}: {is_update}")
            return is_update
        else:
            logging.debug(f"No ArticleVector entry found for {source_article_id}, assuming not an update.")
            return False
    except Exception as e:
        logging.error(f"Error checking for updates for {source_article_id}: {e}", exc_info=True)
        return False

async def update_articles_updated_by(article_id: int, source_article_id: int) -> bool:
    """
    Updates the UpdatedBy field for articles that are updated by the current article.

    Args:
        article_id (int): The ID of the current article (the one doing the updating) in NewsArticles table.
        source_article_id (int): The SourceArticle ID corresponding to the current article.

    Returns:
        bool: True if successful (or no updates needed), False if an error occurred during update attempts.
    """
    if not _check_supabase_client():
        return False
    overall_success = True
    try:
        logging.info(f"Checking ArticleVector for articles updated by SourceArticle {source_article_id}.")
        # Get the list of SourceArticle IDs that the current article updates
        vector_response = supabase.table("ArticleVector").select("update").eq("SourceArticle", source_article_id).limit(1).execute()

        if vector_response.data and vector_response.data[0].get("update"):
            updated_source_ids = vector_response.data[0]["update"] # This is expected to be a list of IDs
            if updated_source_ids and isinstance(updated_source_ids, list):
                logging.info(f"SourceArticle {source_article_id} updates {len(updated_source_ids)} other articles. Setting their UpdatedBy to {article_id}.")
                # Update all NewsArticles whose SourceArticle is in updated_source_ids
                update_response = supabase.table("NewsArticles").update(
                    {"UpdatedBy": article_id}
                ).in_("SourceArticle", updated_source_ids).execute()

                # Check for errors in the update response
                if hasattr(update_response, 'error') and update_response.error:
                     logging.error(f"Error updating UpdatedBy field for articles {updated_source_ids}: {update_response.error}")
                     overall_success = False
                # Optionally, check how many rows were affected if the client provides this info
                else:
                     logging.info(f"Successfully set UpdatedBy field for articles updated by {source_article_id}.")

            else:
                logging.info(f"SourceArticle {source_article_id} has an 'update' field in ArticleVector, but it's empty or not a list.")
        else:
            logging.info(f"SourceArticle {source_article_id} does not update any other articles according to ArticleVector.")

        return overall_success
    except Exception as e:
        logging.error(f"Error processing update_articles_updated_by for source {source_article_id}: {e}", exc_info=True)
        return False


async def insert_processed_article(article_data: Dict) -> Optional[int]:
    """
    Inserts a processed article into the NewsArticles table.
    Determines `isUpdate` status and handles `UpdatedBy` linkage.

    Args:
        article_data (Dict): Dictionary containing the article data. Must include 'SourceArticle'.

    Returns:
        The ID of the newly inserted NewsArticle if successful, None otherwise.
    """
    if not _check_supabase_client():
        return None

    required_keys = ["headlineEnglish", "headlineGerman", "SummaryEnglish", "SummaryGerman",
                     "ContentEnglish", "ContentGerman", "Image1", "SourceArticle"]
    missing_keys = [key for key in required_keys if key not in article_data]
    if missing_keys:
        logging.error(f"Missing required keys for inserting processed article: {missing_keys}")
        return None

    source_article_id = article_data["SourceArticle"]
    try:
        logging.info(f"Preparing to insert NewsArticle for SourceArticle {source_article_id}.")
        # Check if this article is an update
        is_update = await check_for_updates(source_article_id)
        article_data["isUpdate"] = is_update
        logging.info(f"isUpdate status for SourceArticle {source_article_id}: {is_update}")

        # Ensure timestamps are set correctly
        if "created_at" not in article_data:
            article_data["created_at"] = datetime.now(timezone.utc).isoformat()
        else:
            # Ensure it's in the correct format if provided
            try:
                 article_data['created_at'] = datetime.fromisoformat(article_data['created_at'].replace('Z', '+00:00')).astimezone(timezone.utc).isoformat()
            except ValueError:
                 logging.warning(f"Invalid created_at format provided: {article_data['created_at']}. Using current time.")
                 article_data["created_at"] = datetime.now(timezone.utc).isoformat()

        # Remove fields that might not exist in NewsArticles table or are handled separately
        # e.g., if 'status' isn't directly copied or used in the same way.
        # db_data = {k: v for k, v in article_data.items() if k in NEWS_ARTICLES_COLUMNS} # Replace with actual columns if known

        # Insert the article
        response = supabase.table("NewsArticles").insert(article_data).execute()

        if response.data:
            article_id = response.data[0]["id"]
            logging.info(f"Successfully inserted NewsArticle with ID {article_id} for SourceArticle {source_article_id}.")

            # If article was successfully inserted and it's an update, update the UpdatedBy field
            # for articles that this one updates.
            if is_update:
                await update_articles_updated_by(article_id, source_article_id)

            return article_id
        else:
            logging.error(f"Failed to insert NewsArticle for SourceArticle {source_article_id}. Response: {response}")
            if hasattr(response, 'error') and response.error:
                logging.error(f"Supabase error details: {response.error}")
            return None

    except Exception as e:
        logging.error(f"Error inserting NewsArticle for SourceArticle {source_article_id}: {e}", exc_info=True)
        return None


async def get_team_from_source(source_id: int) -> Optional[int]:
    """
    Gets the team ID associated with a team news source (from TeamNewsSource table).

    Args:
        source_id (int): The ID of the news source from the TeamNewsSource table.

    Returns:
        Optional[int]: The team ID if found, None otherwise.
    """
    if not _check_supabase_client() or not supabase_key or not supabase_url:
        return None
    try:
        logging.info(f"Fetching team for TeamNewsSource ID {source_id}...")
        headers = {
            "apikey": supabase_key,
            "Authorization": f"Bearer {supabase_key}"
        }
        url = f"{supabase_url}/rest/v1/TeamNewsSource"
        params = {
            "select": "Team", # Field containing the foreign key to the Teams table
            "id": f"eq.{source_id}"
        }

        response = requests.get(url, headers=headers, params=params)

        if response.status_code == 200:
            sources = response.json()
            if sources:
                team_id = sources[0].get("Team")
                if team_id is not None:
                    logging.info(f"Found team ID {team_id} for TeamNewsSource {source_id}")
                    return team_id
                else:
                    logging.warning(f"TeamNewsSource {source_id} found, but 'Team' field is null or missing.")
                    return None
            else:
                logging.warning(f"No TeamNewsSource found with ID {source_id}")
                return None
        else:
            logging.error(f"API request failed fetching team for source {source_id}. Status: {response.status_code}, Response: {response.text}")
            return None

    except Exception as e:
        logging.error(f"Error fetching team for TeamNewsSource {source_id}: {e}", exc_info=True)
        return None

async def insert_processed_team_article(article_data: Dict) -> bool:
    """
    Inserts a processed team article into the TeamNewsArticles table.
    Retrieves team ID from source if not provided.

    Args:
        article_data (Dict): Dictionary containing the article data. Expected keys match table schema.
                             Should include 'source' (TeamNewsSource ID) if 'team' is not directly provided.
                             Should include 'id' of the original TeamSourceArticle for marking processed.

    Returns:
        bool: True if successful, False otherwise
    """
    if not _check_supabase_client():
        return False

    required_keys = ["headlineEnglish", "headlineGerman", "summaryEnglish", "summaryGerman",
                     "contentEnglish", "contentGerman", "image1", "sourceArticle", "id"] # 'id' is original TeamSourceArticle ID
    missing_keys = [key for key in required_keys if key not in article_data]
    if missing_keys:
        logging.error(f"Missing required keys for inserting processed team article: {missing_keys}")
        return False

    original_team_source_article_id = article_data["id"] # ID from TeamSourceArticles table

    try:
        logging.info(f"Preparing to insert TeamNewsArticle for TeamSourceArticle {original_team_source_article_id}.")
        # Get the team ID from the source if it's not already provided
        team_id = article_data.get("team")
        if team_id is None and "source" in article_data:
            source_id = article_data.get("source")
            if source_id:
                team_id = await get_team_from_source(source_id)
                logging.info(f"Retrieved team ID {team_id} from TeamNewsSource {source_id}")
            else:
                 logging.warning(f"Article data for {original_team_source_article_id} has no 'source' field to look up team ID.")
        elif team_id is not None:
             logging.info(f"Using provided team ID: {team_id}")
        else:
             logging.warning(f"Cannot determine team ID for TeamSourceArticle {original_team_source_article_id} - 'team' and 'source' fields missing/null.")
             # Decide if insertion should fail or proceed with team=null
             # Let's assume for now it proceeds with null if not found/provided

        # Prepare data for insertion into TeamNewsArticles
        # Ensure we don't insert the original 'id' field
        db_data = {
            "headlineEnglish": article_data["headlineEnglish"],
            "headlineGerman": article_data["headlineGerman"],
            "summaryEnglish": article_data["summaryEnglish"],
            "summaryGerman": article_data["summaryGerman"],
            "contentEnglish": article_data["contentEnglish"],
            "contentGerman": article_data["contentGerman"],
            "image1": article_data["image1"],
            "image2": article_data.get("image2"), # Optional
            "image3": article_data.get("image3"), # Optional
            "teamSourceArticle": article_data["sourceArticle"], # Link to original TeamSourceArticle
            "team": team_id, # Use the team ID we found or the one provided (can be null)
            "status": article_data.get("status", "NEW") # Default to NEW if not provided
        }

        # Set created_at if not provided
        if "created_at" not in article_data:
             db_data["created_at"] = datetime.now(timezone.utc).isoformat()
        else:
             try:
                  db_data['created_at'] = datetime.fromisoformat(article_data['created_at'].replace('Z', '+00:00')).astimezone(timezone.utc).isoformat()
             except ValueError:
                  logging.warning(f"Invalid created_at format provided: {article_data['created_at']}. Using current time.")
                  db_data["created_at"] = datetime.now(timezone.utc).isoformat()


        response = supabase.table("TeamNewsArticles").insert(db_data).execute()

        if response.data:
            inserted_id = response.data[0].get("id")
            logging.info(f"Successfully inserted TeamNewsArticle {inserted_id} for TeamSourceArticle {original_team_source_article_id}")
            # Mark the original TeamSourceArticle as processed ONLY AFTER successful insertion
            marked_success = await mark_team_article_as_processed(original_team_source_article_id)
            if not marked_success:
                 # Log a warning but don't necessarily roll back the insert
                 logging.warning(f"Failed to mark original TeamSourceArticle {original_team_source_article_id} as processed, but insertion succeeded.")
            return True
        else:
            logging.error(f"Failed to insert TeamNewsArticle for TeamSourceArticle {original_team_source_article_id}. Response: {response}")
            if hasattr(response, 'error') and response.error:
                logging.error(f"Supabase error details: {response.error}")
            return False
    except Exception as e:
        logging.error(f"Error inserting team article for {original_team_source_article_id}: {e}", exc_info=True)
        return False


async def batch_update_article_status() -> Dict:
    """
    Helper function to update the isUpdate status of articles in the NewsArticles table.
    Batch processes all existing articles in NewsArticles table to update their isUpdate status
    based on ArticleVector table data.

    Returns:
        Dict: Statistics about the operation (total processed, updated count, errors)
    """
    if not _check_supabase_client():
        return {"total": 0, "updated": 0, "errors": 1, "message": "Supabase client not initialized"}

    stats = {"total": 0, "updated": 0, "errors": 0}
    page = 0
    page_size = 1000 # Process in batches to avoid memory issues

    try:
        logging.info("Starting batch update of NewsArticles.isUpdate status...")

        while True:
            logging.info(f"Fetching NewsArticles page {page + 1}...")
            response = supabase.table("NewsArticles") \
                .select("id, SourceArticle, isUpdate", count='exact') \
                .range(page * page_size, (page + 1) * page_size - 1) \
                .execute()

            if not response.data and page == 0:
                logging.info("No articles found in NewsArticles table.")
                return stats

            if not response.data:
                 logging.info("No more articles found.")
                 break # Exit loop if no more data

            articles = response.data
            total_count = response.count # Get total count from the first response
            if page == 0:
                logging.info(f"Total articles to process: {total_count}")
                stats["total"] = total_count

            logging.info(f"Processing {len(articles)} articles from page {page + 1}...")

            # Prepare source IDs for batch query to ArticleVector
            source_ids = [article["SourceArticle"] for article in articles if article.get("SourceArticle") is not None]
            article_map = {article["id"]: article for article in articles} # Map by NewsArticle ID

            if not source_ids:
                logging.info("No SourceArticle IDs found in this batch.")
                page += 1
                continue

            # Fetch update status from ArticleVector for the batch
            logging.debug(f"Fetching ArticleVector entries for {len(source_ids)} source IDs...")
            vector_response = supabase.table("ArticleVector") \
                .select("SourceArticle, update") \
                .in_("SourceArticle", source_ids) \
                .execute()

            # Create a map of source_id -> should_be_update
            vector_update_status = {}
            if vector_response.data:
                for vector_entry in vector_response.data:
                    vector_update_status[vector_entry["SourceArticle"]] = bool(vector_entry.get("update"))
            logging.debug(f"Found ArticleVector update status for {len(vector_update_status)} source IDs.")

            # Process each article in the current page
            updates_to_make = []
            for article in articles:
                article_id = article["id"]
                source_article_id = article.get("SourceArticle")
                current_status = article.get("isUpdate")

                if source_article_id is None:
                    continue # Skip if no source article link

                should_be_update = vector_update_status.get(source_article_id, False) # Default to False if no vector entry

                # Only update if the status needs to change
                if current_status != should_be_update:
                    updates_to_make.append({
                        "id": article_id,
                        "isUpdate": should_be_update
                    })
                    logging.info(f"Article {article_id} status change needed: {current_status} -> {should_be_update}")


            # Perform batch update if changes are needed
            if updates_to_make:
                logging.info(f"Attempting to update {len(updates_to_make)} articles...")
                # Supabase Python client v2 doesn't have a direct batch update.
                # We need to iterate and update individually or use PostgREST directly if needed for performance.
                # Let's update individually for simplicity here.
                for update_data in updates_to_make:
                    try:
                        update_response = supabase.table("NewsArticles") \
                            .update({"isUpdate": update_data["isUpdate"]}) \
                            .eq("id", update_data["id"]) \
                            .execute()
                        if hasattr(update_response, 'error') and update_response.error:
                             logging.error(f"Error updating article {update_data['id']}: {update_response.error}")
                             stats["errors"] += 1
                        else:
                             # Check if update actually returned data (might depend on Prefer header)
                             if update_response.data:
                                 stats["updated"] += 1
                                 logging.debug(f"Successfully updated article {update_data['id']} isUpdate status to {update_data['isUpdate']}")
                             else:
                                 # Log warning if no data returned, might mean row not found or other issue
                                 logging.warning(f"Update for article {update_data['id']} did not return data. Status might not be updated.")
                                 # Optionally re-check the status here or treat as error
                                 stats["errors"] += 1 # Treat as error for now if update returns no data
                    except Exception as e:
                         logging.error(f"Exception updating article {update_data['id']}: {e}", exc_info=True)
                         stats["errors"] += 1
            else:
                 logging.info(f"No status changes needed for articles on page {page + 1}.")

            page += 1 # Move to the next page

        logging.info(f"Batch update processing complete. Final Stats: {stats}")
        return stats

    except Exception as e:
        logging.error(f"Error during batch update of article status: {e}", exc_info=True)
        stats["errors"] += 1
        return stats


# --- New Functions for Cluster Pipeline ---

async def fetch_recent_clusters(time_window_hours: int = 48, content_type: str = 'news_article') -> Dict[str, set[int]]:
    """
    Fetches distinct cluster_ids and their associated SourceArticle IDs for recent articles.

    Args:
        time_window_hours: How many hours back to look for articles based on created_at.
        content_type: The contentType to filter articles by.

    Returns:
        A dictionary mapping cluster_id (str UUID) to a set of SourceArticle IDs.
    """
    if not _check_supabase_client() or not supabase_key or not supabase_url:
        return {}

    clusters = {}
    try:
        # Calculate the cutoff time
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=time_window_hours)
        cutoff_iso = cutoff_time.isoformat()
        logging.info(f"Fetching clusters for articles created after: {cutoff_iso}")

        headers = {
            "apikey": supabase_key,
            "Authorization": f"Bearer {supabase_key}"
        }
        url = f"{supabase_url}/rest/v1/SourceArticles"
        params = {
            "select": "id,cluster_id",
            "created_at": f"gte.{cutoff_iso}",  # Articles created recently
            "cluster_id": "not.is.null",        # Must have a cluster_id
            "contentType": f"eq.{content_type}", # Match desired content type
            "duplication_of": "is.null"         # Exclude duplicates
        }
        # Add ordering for pagination if needed, though not strictly necessary here
        # params["order"] = "created_at.desc"

        # Fetch all matching articles (consider pagination if expecting huge numbers)
        all_articles = []
        current_offset = 0
        limit = 1000 # Supabase default/max limit per request

        while True:
            paginated_params = params.copy()
            paginated_params["offset"] = current_offset
            paginated_params["limit"] = limit
            logging.debug(f"Fetching articles with cluster_id, offset {current_offset}, limit {limit}")
            response = requests.get(url, headers=headers, params=paginated_params)

            if response.status_code == 200:
                batch = response.json()
                if not batch:
                    logging.debug("No more articles found in pagination.")
                    break # Exit loop if no more articles
                all_articles.extend(batch)
                current_offset += len(batch)
                if len(batch) < limit:
                    break # Exit if last page fetched
            else:
                logging.error(f"Failed to fetch recent clusters batch. Status: {response.status_code}, Response: {response.text}")
                # Decide whether to return partial data or empty dict on error
                return {} # Return empty on error

        logging.info(f"Fetched {len(all_articles)} recent articles with cluster IDs.")
        for article in all_articles:
            # Ensure cluster_id is treated as string, handle potential nulls again just in case
            cluster_id_val = article.get('cluster_id')
            if cluster_id_val is None:
                continue
            cluster_id_str = str(cluster_id_val)
            article_id = article.get('id')
            if article_id is None:
                 continue

            if cluster_id_str not in clusters:
                clusters[cluster_id_str] = set()
            clusters[cluster_id_str].add(article_id)

        logging.info(f"Identified {len(clusters)} distinct clusters.")
        # Log first 5 cluster IDs and counts for verification
        logged_count = 0
        for cid, ids in clusters.items():
            if logged_count < 5:
                logging.info(f"  Cluster {cid}: {len(ids)} articles")
                logged_count += 1
            else:
                break
        return clusters

    except Exception as e:
        logging.error(f"Error fetching recent clusters: {e}", exc_info=True)
        return {}

async def fetch_recent_cluster_stories(time_window_days: int = 7) -> List[Dict]:
    """
    Fetches recent ClusterStories to compare against new clusters.

    Args:
        time_window_days: How many days back to look for stories based on updated_at.

    Returns:
        A list of dictionaries, each representing a ClusterStory
        with 'id', 'cluster_id', and 'source_article_ids'.
    """
    if not _check_supabase_client():
        return []
    try:
        # Calculate the cutoff time
        cutoff_time = datetime.now(timezone.utc) - timedelta(days=time_window_days)
        cutoff_iso = cutoff_time.isoformat()
        logging.info(f"Fetching cluster stories updated after: {cutoff_iso}")

        response = supabase.table("ClusterStories") \
            .select("id, cluster_id, source_article_ids") \
            .gte("updated_at", cutoff_iso) \
            .execute()

        if response.data:
            logging.info(f"Successfully fetched {len(response.data)} recent cluster stories.")
            # Convert cluster_id to string for consistent matching
            for story in response.data:
                # Ensure 'id' and 'cluster_id' are strings
                story['id'] = str(story['id'])
                if story.get('cluster_id'):
                    story['cluster_id'] = str(story['cluster_id'])
                # Ensure 'source_article_ids' is a list of ints, handle null
                if story.get('source_article_ids') is None:
                    story['source_article_ids'] = []
                elif not isinstance(story['source_article_ids'], list):
                    logging.warning(f"Found non-list source_article_ids for story {story['id']}: {story['source_article_ids']}. Treating as empty.")
                    story['source_article_ids'] = []
                else:
                    # Ensure elements are integers
                    story['source_article_ids'] = [int(id_val) for id_val in story['source_article_ids'] if isinstance(id_val, (int, str)) and str(id_val).isdigit()]

            return response.data
        else:
            # It's not necessarily an error if no recent stories are found
            logging.info("No recent cluster stories found matching the criteria.")
            return []

    except Exception as e:
        logging.error(f"Error fetching recent cluster stories: {e}", exc_info=True)
        return []

async def get_source_articles_content(article_ids: List[int]) -> Dict[int, str]:
    """
    Fetches the English content ('Content' field) for a list of SourceArticle IDs.

    Args:
        article_ids: A list of SourceArticle IDs.

    Returns:
        A dictionary mapping article_id to its English content. Returns empty strings
        for articles not found or without content.
    """
    if not _check_supabase_client():
        return {article_id: '' for article_id in article_ids} # Return default on client error
    content_map = {}
    if not article_ids:
        return content_map

    try:
        # Fetch in batches to avoid overly long URL parameters if list is huge
        batch_size = 100
        fetched_content_count = 0
        for i in range(0, len(article_ids), batch_size):
            batch_ids = article_ids[i:i+batch_size]
            ids_string = ','.join(map(str, batch_ids))
            logging.debug(f"Fetching content for {len(batch_ids)} SourceArticle IDs (Batch {i//batch_size + 1})...")

            response = supabase.table("SourceArticles") \
                .select("id, Content") \
                .in_("id", batch_ids) \
                .execute()

            if response.data:
                batch_found_ids = set()
                for article in response.data:
                    article_id = article['id']
                    content = article.get('Content', '') # Use 'Content' field name
                    content_map[article_id] = content if content else '' # Ensure empty string, not None
                    batch_found_ids.add(article_id)
                    fetched_content_count += 1

                # Log content length for the first article in the first batch as a sample check
                if i == 0 and batch_found_ids:
                     first_id = list(batch_found_ids)[0]
                     logging.info(f"  Sample content length for article {first_id}: {len(content_map[first_id])} chars")
            elif hasattr(response, 'error') and response.error:
                 logging.error(f"Error fetching source article content batch: {response.error}")
                 # Decide if we should continue or abort the whole operation
                 # For now, continue and report missing later
            else:
                 logging.debug(f"No content data returned for batch starting with ID {batch_ids[0]}.")

        # After processing all batches, check for missing articles
        all_fetched_ids = set(content_map.keys())
        missing_ids = set(article_ids) - all_fetched_ids
        if missing_ids:
            logging.warning(f"Could not find content for {len(missing_ids)} article IDs: {list(missing_ids)}")
            for missing_id in missing_ids:
                content_map[missing_id] = '' # Assign empty string if not found

        logging.info(f"Finished fetching content. Found content for {fetched_content_count}/{len(article_ids)} requested articles.")
        return content_map

    except Exception as e:
        logging.error(f"Error fetching source article content: {e}", exc_info=True)
        # Return empty strings for all requested IDs on error
        return {article_id: '' for article_id in article_ids}


async def get_article_translation(source_article_id: int, language_code: str = 'de') -> Optional[str]:
    """
    Checks the ArticleTranslations table for an existing translation.

    Args:
        source_article_id: The ID of the source article.
        language_code: The language code (e.g., 'de').

    Returns:
        The translated content string if found, otherwise None.
    """
    if not _check_supabase_client():
        return None
    try:
        logging.debug(f"Checking translation for source article {source_article_id}, language {language_code}")
        response = supabase.table("ArticleTranslations") \
            .select("translated_content") \
            .eq("source_article_id", source_article_id) \
            .eq("language_code", language_code) \
            .limit(1) \
            .execute()

        if response.data:
            translation = response.data[0].get("translated_content")
            if translation is not None: # Check explicitly for None vs empty string
                logging.debug(f"Found existing translation for article {source_article_id} ({language_code}). Length: {len(translation)}")
                return translation
            else:
                logging.debug(f"Found existing translation entry for article {source_article_id} ({language_code}), but content is null.")
                return None # Treat null content as not found for practical purposes
        else:
            logging.debug(f"No existing translation found for article {source_article_id} ({language_code}).")
            return None
    except Exception as e:
        logging.error(f"Error checking article translation for {source_article_id} ({language_code}): {e}", exc_info=True)
        return None

async def insert_article_translation(source_article_id: int, language_code: str, translated_content: str) -> bool:
    """
    Inserts a new translation into the ArticleTranslations table.

    Args:
        source_article_id: The ID of the source article.
        language_code: The language code (e.g., 'de').
        translated_content: The translated text.

    Returns:
        True if insertion was successful, False otherwise.
    """
    if not _check_supabase_client():
        return False
    try:
        logging.info(f"Inserting translation for source article {source_article_id}, language {language_code}. Length: {len(translated_content)}")
        data_to_insert = {
            "source_article_id": source_article_id,
            "language_code": language_code,
            "translated_content": translated_content
            # created_at is handled by the database default trigger/policy
        }
        response = supabase.table("ArticleTranslations").insert(data_to_insert).execute()

        # Check if the insert operation returned data (indicating success)
        if response.data:
             logging.info(f"Successfully inserted translation for {source_article_id} ({language_code}). New ID: {response.data[0].get('id')}")
             return True
        else:
            # Log the actual response if data is empty or doesn't indicate success
            logging.error(f"Failed to insert translation for {source_article_id} ({language_code}). Response: {response}")
            # Attempt to extract specific error if possible
            error_info = getattr(response, 'error', None)
            if error_info:
                 logging.error(f"Supabase error details: {error_info}")
            return False

    except Exception as e:
        # Catch potential database constraint errors, etc.
        logging.error(f"Error inserting article translation for {source_article_id} ({language_code}): {e}", exc_info=True)
        return False


async def insert_cluster_story(story_data: Dict) -> Optional[str]:
    """
    Inserts a new record into the ClusterStories table.

    Args:
        story_data: A dictionary containing the data for the new story.
                    Expected keys match the ClusterStories table columns.
                    'created_at' and 'updated_at' should be set to ISO strings.

    Returns:
        The UUID string of the newly created story if successful, otherwise None.
    """
    if not _check_supabase_client():
        return None

    required_keys = [
        "cluster_id", "source_article_ids", "headline_english", "headline_german",
        "summary_english", "summary_german", "body_english", "body_german",
        "image1_url", "status", "created_at", "updated_at"
    ]
    # Optional keys: image2_url, image3_url
    missing_keys = [key for key in required_keys if key not in story_data]
    if missing_keys:
        logging.error(f"Missing required keys for inserting cluster story: {missing_keys}. Data: {story_data}")
        return None

    # --- Data Validation ---
    try:
        # Validate cluster_id is a valid UUID string
        cluster_uuid = UUID(str(story_data['cluster_id']))
        story_data['cluster_id'] = str(cluster_uuid) # Ensure it's passed as string

        # Validate source_article_ids is a list of integers
        if not isinstance(story_data['source_article_ids'], list) or not all(isinstance(i, int) for i in story_data['source_article_ids']):
            raise ValueError(f"Invalid source_article_ids format: {story_data['source_article_ids']}. Must be a list of integers.")

        # Validate timestamps are valid ISO format strings
        story_data['created_at'] = datetime.fromisoformat(story_data['created_at'].replace('Z', '+00:00')).astimezone(timezone.utc).isoformat()
        story_data['updated_at'] = datetime.fromisoformat(story_data['updated_at'].replace('Z', '+00:00')).astimezone(timezone.utc).isoformat()

    except (ValueError, TypeError) as e:
         logging.error(f"Data validation failed for cluster story insertion: {e}. Data: {story_data}")
         return None
    # --- End Validation ---

    try:
        logging.info(f"Inserting new ClusterStory for cluster_id: {story_data['cluster_id']}")

        response = supabase.table("ClusterStories").insert(story_data).execute()

        if response.data and response.data[0].get("id"):
            new_story_id = response.data[0].get("id")
            logging.info(f"Successfully inserted ClusterStory with ID: {new_story_id}")
            return str(new_story_id) # Return the string representation of the UUID
        else:
            logging.error(f"Failed to insert ClusterStory. Response: {response}")
            error_info = getattr(response, 'error', None)
            if error_info:
                 logging.error(f"Supabase error details: {error_info}")
            return None

    except Exception as e:
        logging.error(f"Error inserting cluster story for cluster {story_data.get('cluster_id')}: {e}", exc_info=True)
        return None


async def update_cluster_story(story_id: str, story_data: Dict) -> bool:
    """
    Updates an existing record in the ClusterStories table.

    Args:
        story_id: The UUID string of the story to update.
        story_data: Dictionary containing the fields to update. Must include 'updated_at'.
                    Should include the updated 'source_article_ids'.

    Returns:
        True if the update was successful (affected at least one row), False otherwise.
    """
    if not _check_supabase_client():
        return False
    if not story_id:
        logging.error("Missing story_id for update.")
        return False

    # --- Data Validation ---
    try:
        # Validate story_id is a UUID
        UUID(story_id)

        # Validate updated_at
        if 'updated_at' not in story_data:
             raise ValueError("Missing 'updated_at' timestamp in update data.")
        story_data['updated_at'] = datetime.fromisoformat(story_data['updated_at'].replace('Z', '+00:00')).astimezone(timezone.utc).isoformat()

        # Validate source_article_ids if present
        if 'source_article_ids' in story_data:
            if not isinstance(story_data['source_article_ids'], list) or not all(isinstance(i, int) for i in story_data['source_article_ids']):
                raise ValueError(f"Invalid source_article_ids format for update: {story_data['source_article_ids']}. Must be a list of integers.")

    except (ValueError, TypeError) as e:
         logging.error(f"Data validation failed for cluster story update (ID: {story_id}): {e}. Data: {story_data}")
         return False
    # --- End Validation ---

    try:
        logging.info(f"Updating ClusterStory with ID: {story_id}")
        # Explicitly remove 'id' and 'created_at' if they accidentally got into the update dict
        story_data.pop('id', None)
        story_data.pop('created_at', None)
        story_data.pop('cluster_id', None) # Should not change cluster_id on update

        if not story_data:
            logging.warning(f"No fields to update for ClusterStory {story_id} after validation.")
            return False # Or True if no update needed is considered success

        response = supabase.table("ClusterStories") \
            .update(story_data) \
            .eq("id", story_id) \
            .execute()

        # Check if the update operation returned data (indicating success and row found)
        if response.data:
             logging.info(f"Successfully updated ClusterStory {story_id}.")
             return True
        # Handle cases where no rows were updated (e.g., story_id didn't exist) or an error occurred
        else:
            logging.warning(f"ClusterStory update for {story_id} did not return data.")
            error_info = getattr(response, 'error', None)
            if error_info:
                logging.error(f"Supabase error details during update: {error_info}")
                return False

            # If no data and no explicit error, check if the record still exists
            logging.debug(f"Checking if story {story_id} exists after update attempt...")
            check_response = supabase.table("ClusterStories").select("id", count='exact').eq("id", story_id).execute()
            if check_response.count == 0:
                logging.error(f"Failed to update ClusterStory {story_id}: Story ID not found.")
            else:
                # If it exists but didn't update/return data, could be another issue or config.
                logging.error(f"Failed to update ClusterStory {story_id} (record exists but update likely failed or returned minimal).")
            return False

    except Exception as e:
        logging.error(f"Error updating cluster story {story_id}: {e}", exc_info=True)
        return False

# --- End of New Cluster Functions ---
