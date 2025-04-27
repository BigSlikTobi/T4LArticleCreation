import os
import requests
from typing import List, Dict, Optional, Set # Added Set
from dotenv import load_dotenv
from supabase import create_client, Client
# Add imports for datetime, timedelta, UUID if not already present
from datetime import datetime, timedelta, timezone
from uuid import UUID
import logging # Add logging

# --- Logging Setup ---
# Configure logging level and format only once
# Use filename and line number for better debugging context
logging.basicConfig(
    level=logging.INFO, # Set the desired global level
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    force=True # Force re-configuration if already configured elsewhere (e.g., by root logger)
)
# Create a specific logger for this module if needed for finer control
logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG) # Uncomment to enable DEBUG level just for this file

# Load environment variables
load_dotenv()

# Initialize Supabase client
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_KEY")

# Add type hinting for the client
supabase: Optional[Client] = None
if supabase_url and supabase_key:
    try:
        supabase = create_client(supabase_url, supabase_key)
        logger.info("Supabase client initialized successfully.")
    except Exception as e:
        logger.critical(f"Failed to initialize Supabase client: {e}", exc_info=True)
        # Depending on your application's needs, you might exit or raise an error here
else:
    logger.critical("Supabase URL or Key not found in environment variables. Database functions will fail.")


# Helper function to check if Supabase client is available
def _check_supabase_client():
    if supabase is None:
        logger.error("Supabase client is not initialized.")
        return False
    return True

async def fetch_primary_sources() -> List[int]:
    """
    Fetches all primary news sources from the NewsSource table.

    Returns:
        List[int]: List of source IDs that have isPrimarySource set to True
    """
    if not _check_supabase_client() or not supabase_key or not supabase_url:
        logger.warning("Cannot fetch primary sources: Supabase client/config missing.")
        return []
    try:
        logger.info("Fetching primary news sources...")
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
            logger.info(f"Successfully fetched {len(source_ids)} primary sources: {source_ids}")
            return source_ids
        else:
            logger.error(f"API request failed fetching primary sources. Status: {response.status_code}, Response: {response.text}")
            return []

    except Exception as e:
        logger.error(f"Error fetching primary sources: {e}", exc_info=True)
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
        logger.warning("Cannot fetch unprocessed articles: Supabase client/config missing.")
        return []
    try:
        # Get primary sources
        primary_sources = await fetch_primary_sources()
        if not primary_sources:
            logger.warning("No primary sources found, using fallback sources [1, 2, 4]")
            primary_sources = [1, 2, 4]  # Fallback to original sources if fetch fails

        logger.info("Fetching unprocessed articles from database...")
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

        # Pagination might be needed here if the result set is very large
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
            logger.info(f"Successfully fetched {len(filtered_articles)} unprocessed articles")
            return filtered_articles
        else:
            logger.error(f"API request failed fetching unprocessed articles. Status: {response.status_code}, Response: {response.text}")
            return []

    except Exception as e:
        logger.error(f"Error fetching articles: {e}", exc_info=True)
        return []

async def fetch_unprocessed_team_articles() -> List[Dict]:
    """
    Fetches articles from the TeamSourceArticles table that have isArticleCreated set to false
    and contentType == 'news_article'.

    Returns:
        List[Dict]: List of unprocessed team articles
    """
    if not _check_supabase_client() or not supabase_key or not supabase_url:
        logger.warning("Cannot fetch unprocessed team articles: Supabase client/config missing.")
        return []
    try:
        logger.info("Fetching unprocessed team articles...")
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
            logger.info(f"Successfully fetched {len(filtered_articles)} unprocessed team articles")
            return filtered_articles
        else:
            logger.error(f"API request failed fetching unprocessed team articles. Status: {response.status_code}, Response: {response.text}")
            return []
    except Exception as e:
        logger.error(f"Error fetching unprocessed team articles: {e}", exc_info=True)
        return []

async def fetch_teams() -> List[Dict]:
    """
    Fetches all teams from the Teams table.

    Returns:
        List[Dict]: List of teams with their IDs and fullNames
    """
    if not _check_supabase_client() or not supabase_key or not supabase_url:
        logger.warning("Cannot fetch teams: Supabase client/config missing.")
        return []
    try:
        logger.info("Fetching teams from database...")
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
            logger.info(f"Successfully fetched {len(teams)} teams")
            return teams
        else:
            logger.error(f"API request failed fetching teams. Status: {response.status_code}, Response: {response.text}")
            return []

    except Exception as e:
        logger.error(f"Error fetching teams: {e}", exc_info=True)
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
        logger.info(f"Marking SourceArticle {article_id} as processed.")
        data = {"isArticleCreated": True}
        response = supabase.table("SourceArticles").update(data).eq("id", article_id).execute()
        # Basic check for success (no explicit error)
        error_info = getattr(response, 'error', None)
        if error_info:
            logger.error(f"Error marking article {article_id} as processed: {error_info}")
            return False
        # Note: Update might return empty data even on success depending on Prefer header.
        # Consider success if no error is raised/returned.
        logger.info(f"Successfully marked SourceArticle {article_id} as processed.")
        return True
    except Exception as e:
        logger.error(f"Exception marking article {article_id} as processed: {e}", exc_info=True)
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
        logger.info(f"Marking TeamSourceArticle {article_id} as processed.")
        data = {"isArticleCreated": True}
        response = supabase.table("TeamSourceArticles").update(data).eq("id", article_id).execute()
        error_info = getattr(response, 'error', None)
        if error_info:
            logger.error(f"Error marking team article {article_id} as processed: {error_info}")
            return False
        logger.info(f"Successfully marked TeamSourceArticle {article_id} as processed.")
        return True
    except Exception as e:
        logger.error(f"Exception marking team article {article_id} as processed: {e}", exc_info=True)
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
        logger.debug(f"Checking for updates status for SourceArticle {source_article_id}.")
        response = supabase.table("ArticleVector").select("update").eq("SourceArticle", source_article_id).limit(1).execute()
        if response.data:
            article_vector = response.data[0]
            is_update = bool(article_vector.get("update")) # Check if 'update' list/field is non-empty/true
            logger.debug(f"Update status for {source_article_id}: {is_update}")
            return is_update
        else:
            logger.debug(f"No ArticleVector entry found for {source_article_id}, assuming not an update.")
            return False
    except Exception as e:
        logger.error(f"Error checking for updates for {source_article_id}: {e}", exc_info=True)
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
        logger.info(f"Checking ArticleVector for articles updated by SourceArticle {source_article_id}.")
        # Get the list of SourceArticle IDs that the current article updates
        vector_response = supabase.table("ArticleVector").select("update").eq("SourceArticle", source_article_id).limit(1).execute()

        if vector_response.data and vector_response.data[0].get("update"):
            updated_source_ids = vector_response.data[0]["update"] # This is expected to be a list of IDs
            # Ensure it's a non-empty list before proceeding
            if updated_source_ids and isinstance(updated_source_ids, list):
                logger.info(f"SourceArticle {source_article_id} updates {len(updated_source_ids)} other articles. Setting their UpdatedBy to {article_id}.")
                # Update all NewsArticles whose SourceArticle is in updated_source_ids
                update_response = supabase.table("NewsArticles").update(
                    {"UpdatedBy": article_id}
                ).in_("SourceArticle", updated_source_ids).execute()

                # Check for errors in the update response
                error_info = getattr(update_response, 'error', None)
                if error_info:
                     logger.error(f"Error updating UpdatedBy field for articles {updated_source_ids}: {error_info}")
                     overall_success = False
                else:
                     # Optionally log success based on data presence (though might be empty)
                     logger.info(f"UpdatedBy update executed for articles updated by {source_article_id}. Response data length: {len(update_response.data) if hasattr(update_response, 'data') else 'N/A'}")
            else:
                logger.info(f"SourceArticle {source_article_id} has an 'update' field in ArticleVector, but it's empty or not a list.")
        else:
            logger.info(f"SourceArticle {source_article_id} does not update any other articles according to ArticleVector.")

        return overall_success
    except Exception as e:
        logger.error(f"Error processing update_articles_updated_by for source {source_article_id}: {e}", exc_info=True)
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
        logger.error(f"Missing required keys for inserting processed article: {missing_keys}")
        return None

    source_article_id = article_data["SourceArticle"]
    try:
        logger.info(f"Preparing to insert NewsArticle for SourceArticle {source_article_id}.")
        # Check if this article is an update
        is_update = await check_for_updates(source_article_id)
        article_data["isUpdate"] = is_update
        logger.info(f"isUpdate status for SourceArticle {source_article_id}: {is_update}")

        # Ensure timestamps are set correctly
        if "created_at" not in article_data or not article_data["created_at"]:
            article_data["created_at"] = datetime.now(timezone.utc).isoformat()
        else:
            try:
                 article_data['created_at'] = datetime.fromisoformat(str(article_data['created_at']).replace('Z', '+00:00')).astimezone(timezone.utc).isoformat()
            except (ValueError, TypeError):
                 logger.warning(f"Invalid created_at format provided: {article_data['created_at']}. Using current time.")
                 article_data["created_at"] = datetime.now(timezone.utc).isoformat()

        # Define expected columns to avoid inserting extra fields
        # Adjust this list based on your actual NewsArticles table schema
        allowed_columns = {
            "created_at", "headlineEnglish", "headlineGerman", "SummaryEnglish",
            "SummaryGerman", "ContentEnglish", "ContentGerman", "Image1", "Image2",
            "Image3", "SourceArticle", "team", "isUpdate", "UpdatedBy", "status" # Add/remove as needed
        }
        db_data = {k: v for k, v in article_data.items() if k in allowed_columns}

        # Insert the article
        response = supabase.table("NewsArticles").insert(db_data).execute()

        if response.data and response.data[0].get("id"):
            article_id = response.data[0]["id"]
            logger.info(f"Successfully inserted NewsArticle with ID {article_id} for SourceArticle {source_article_id}.")

            # If article was successfully inserted and it's an update, update the UpdatedBy field
            if is_update:
                await update_articles_updated_by(article_id, source_article_id)

            return article_id
        else:
            error_info = getattr(response, 'error', None)
            logger.error(f"Failed to insert NewsArticle for SourceArticle {source_article_id}. Error: {error_info}. Response: {response}")
            return None

    except Exception as e:
        logger.error(f"Error inserting NewsArticle for SourceArticle {source_article_id}: {e}", exc_info=True)
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
        logger.warning("Cannot get team from source: Supabase client/config missing.")
        return None
    try:
        logger.info(f"Fetching team for TeamNewsSource ID {source_id}...")
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
                    logger.info(f"Found team ID {team_id} for TeamNewsSource {source_id}")
                    return team_id
                else:
                    logger.warning(f"TeamNewsSource {source_id} found, but 'Team' field is null or missing.")
                    return None
            else:
                logger.warning(f"No TeamNewsSource found with ID {source_id}")
                return None
        else:
            logger.error(f"API request failed fetching team for source {source_id}. Status: {response.status_code}, Response: {response.text}")
            return None

    except Exception as e:
        logger.error(f"Error fetching team for TeamNewsSource {source_id}: {e}", exc_info=True)
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
        logger.error(f"Missing required keys for inserting processed team article: {missing_keys}")
        return False

    original_team_source_article_id = article_data["id"] # ID from TeamSourceArticles table

    try:
        logger.info(f"Preparing to insert TeamNewsArticle for TeamSourceArticle {original_team_source_article_id}.")
        # Get the team ID from the source if it's not already provided
        team_id = article_data.get("team")
        if team_id is None and "source" in article_data:
            source_id = article_data.get("source")
            if source_id:
                team_id = await get_team_from_source(source_id)
                logger.info(f"Retrieved team ID {team_id} from TeamNewsSource {source_id}")
            else:
                 logger.warning(f"Article data for {original_team_source_article_id} has no 'source' field to look up team ID.")
        elif team_id is not None:
             logger.info(f"Using provided team ID: {team_id}")
        else:
             logger.warning(f"Cannot determine team ID for TeamSourceArticle {original_team_source_article_id} - 'team' and 'source' fields missing/null.")
             # team_id remains None

        # Define expected columns for TeamNewsArticles table
        allowed_columns = {
             "headlineEnglish", "headlineGerman", "summaryEnglish", "summaryGerman",
             "contentEnglish", "contentGerman", "image1", "image2", "image3",
             "teamSourceArticle", "team", "status", "created_at" # Adjust as per actual schema
        }
        db_data = {k: v for k, v in article_data.items() if k in allowed_columns}

        # Ensure required fields not from article_data are set
        db_data["team"] = team_id # Use looked-up or provided ID
        db_data["teamSourceArticle"] = article_data["sourceArticle"] # Link to original
        db_data["status"] = article_data.get("status", "NEW")

        # Set created_at if not provided
        if "created_at" not in db_data or not db_data["created_at"]:
             db_data["created_at"] = datetime.now(timezone.utc).isoformat()
        else:
             try:
                  db_data['created_at'] = datetime.fromisoformat(str(db_data['created_at']).replace('Z', '+00:00')).astimezone(timezone.utc).isoformat()
             except (ValueError, TypeError):
                  logger.warning(f"Invalid created_at format provided: {db_data['created_at']}. Using current time.")
                  db_data["created_at"] = datetime.now(timezone.utc).isoformat()


        response = supabase.table("TeamNewsArticles").insert(db_data).execute()

        if response.data and response.data[0].get("id"):
            inserted_id = response.data[0].get("id")
            logger.info(f"Successfully inserted TeamNewsArticle {inserted_id} for TeamSourceArticle {original_team_source_article_id}")
            # Mark the original TeamSourceArticle as processed ONLY AFTER successful insertion
            marked_success = await mark_team_article_as_processed(original_team_source_article_id)
            if not marked_success:
                 logger.warning(f"Failed to mark original TeamSourceArticle {original_team_source_article_id} as processed, but insertion succeeded.")
            return True
        else:
            error_info = getattr(response, 'error', None)
            logger.error(f"Failed to insert TeamNewsArticle for TeamSourceArticle {original_team_source_article_id}. Error: {error_info}. Response: {response}")
            return False
    except Exception as e:
        logger.error(f"Error inserting team article for {original_team_source_article_id}: {e}", exc_info=True)
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
    total_count = None # Initialize total count

    try:
        logger.info("Starting batch update of NewsArticles.isUpdate status...")

        while True:
            logger.info(f"Fetching NewsArticles page {page + 1}...")
            # Fetch with count on the first request only for efficiency
            current_count_method = 'exact' if page == 0 else None
            response = supabase.table("NewsArticles") \
                .select("id, SourceArticle, isUpdate", count=current_count_method) \
                .range(page * page_size, (page + 1) * page_size - 1) \
                .execute()

            if hasattr(response, 'error') and response.error:
                 logger.error(f"Error fetching NewsArticles page {page + 1}: {response.error}")
                 stats["errors"] += 1
                 break # Stop processing if a page fetch fails

            # Get total count from the first response
            if page == 0:
                total_count = getattr(response, 'count', 0) if hasattr(response, 'count') else len(response.data)
                if total_count is None: total_count = 0 # Handle null count
                logger.info(f"Total articles to process: {total_count}")
                stats["total"] = total_count
                if not response.data and total_count == 0:
                     logger.info("No articles found in NewsArticles table.")
                     return stats

            if not response.data:
                 logger.info("No more articles found.")
                 break # Exit loop if no more data

            articles = response.data
            logger.info(f"Processing {len(articles)} articles from page {page + 1}...")

            # Prepare source IDs for batch query to ArticleVector
            source_ids = [article["SourceArticle"] for article in articles if article.get("SourceArticle") is not None]

            if not source_ids:
                logger.info("No SourceArticle IDs found in this batch.")
                page += 1
                continue

            # Fetch update status from ArticleVector for the batch
            logger.debug(f"Fetching ArticleVector entries for {len(source_ids)} source IDs...")
            vector_response = supabase.table("ArticleVector") \
                .select("SourceArticle, update") \
                .in_("SourceArticle", source_ids) \
                .execute()

            # Create a map of source_id -> should_be_update
            vector_update_status = {}
            if hasattr(vector_response, 'error') and vector_response.error:
                 logger.error(f"Error fetching ArticleVector batch: {vector_response.error}")
                 # Decide how to handle - skip batch? Assume false? For now, log and continue (articles won't be marked as update)
                 stats["errors"] += len(articles) # Count all articles in batch as error? Or just log?
            elif vector_response.data:
                for vector_entry in vector_response.data:
                    vector_update_status[vector_entry["SourceArticle"]] = bool(vector_entry.get("update"))
            logger.debug(f"Found ArticleVector update status for {len(vector_update_status)} source IDs.")

            # Process each article in the current page
            updates_to_make = []
            for article in articles:
                article_id = article["id"]
                source_article_id = article.get("SourceArticle")
                current_status = article.get("isUpdate")

                if source_article_id is None: continue

                should_be_update = vector_update_status.get(source_article_id, False)

                if current_status != should_be_update:
                    updates_to_make.append({ "id": article_id, "isUpdate": should_be_update })
                    logger.info(f"Article {article_id}: Status change needed: {current_status} -> {should_be_update}")

            # Perform batch update if changes are needed
            if updates_to_make:
                logger.info(f"Attempting to update {len(updates_to_make)} articles individually...")
                for update_data in updates_to_make:
                    try:
                        # Set return=minimal to avoid fetching updated row data
                        update_response = supabase.table("NewsArticles") \
                            .update({"isUpdate": update_data["isUpdate"]}) \
                            .eq("id", update_data["id"]) \
                            .execute()
                        error_info = getattr(update_response, 'error', None)
                        if error_info:
                             logger.error(f"Error updating article {update_data['id']}: {error_info}")
                             stats["errors"] += 1
                        else:
                             # Success if no error with return=minimal
                             stats["updated"] += 1
                             logger.debug(f"Successfully updated article {update_data['id']} isUpdate status to {update_data['isUpdate']}")
                    except Exception as e_update:
                         logger.error(f"Exception updating article {update_data['id']}: {e_update}", exc_info=True)
                         stats["errors"] += 1
            else:
                 logger.info(f"No status changes needed for articles on page {page + 1}.")

            page += 1 # Move to the next page

        logger.info(f"Batch update processing complete. Final Stats: {stats}")
        return stats

    except Exception as e:
        logger.error(f"Error during batch update of article status: {e}", exc_info=True)
        stats["errors"] += 1 # Increment general error count
        return stats


# --- New Functions for Cluster Pipeline ---

async def fetch_recent_clusters(time_window_hours: int = 48, content_type: str = 'news_article') -> Dict[str, set[int]]:
    """Fetches distinct cluster_ids and their associated SourceArticle IDs for recent articles."""
    if not _check_supabase_client() or not supabase_key or not supabase_url:
        return {}

    clusters: Dict[str, set[int]] = {} # Enforce type hint
    try:
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=time_window_hours)
        cutoff_iso = cutoff_time.isoformat()
        logger.info(f"Fetching clusters for articles created after: {cutoff_iso}")

        headers = {
            "apikey": supabase_key,
            "Authorization": f"Bearer {supabase_key}"
        }
        url = f"{supabase_url}/rest/v1/SourceArticles"
        params = { # Ensure params is defined correctly as a dict here
            "select": "id,cluster_id",
            "created_at": f"gte.{cutoff_iso}",
            "cluster_id": "not.is.null",
            "contentType": f"eq.{content_type}",
            "duplication_of": "is.null"
        }

        all_articles = []
        current_offset = 0
        limit = 1000 # Supabase default/max limit per request? Check docs.

        while True:
            # Make sure params is a dictionary before copying
            if not isinstance(params, dict):
                logger.error(f"Internal error: params is not a dict before pagination loop. Type: {type(params)}")
                return {}
            paginated_params = params.copy() # Create a copy for each request

            # Check type of paginated_params before assignment
            if not isinstance(paginated_params, dict):
                 logger.error(f"Internal error: paginated_params is not a dict before assignment. Type: {type(paginated_params)}")
                 return {}

            paginated_params["offset"] = current_offset # Assign offset
            paginated_params["limit"] = limit # Assign limit

            logger.debug(f"Fetching articles with cluster_id, offset {current_offset}, limit {limit}")
            response = requests.get(url, headers=headers, params=paginated_params)

            if response.status_code == 200:
                 batch = response.json()
                 if not batch:
                     logger.debug("No more articles found in pagination.")
                     break # Exit loop if no more articles
                 all_articles.extend(batch)
                 current_offset += len(batch)
                 # Exit if fewer items were returned than the limit (last page)
                 if len(batch) < limit:
                     break
            else:
                logger.error(f"Failed to fetch recent clusters batch. Status: {response.status_code}, Response: {response.text}")
                return {} # Return empty on error

        # Now process all_articles after fetching is complete
        logger.info(f"Fetched {len(all_articles)} recent articles with cluster IDs.")
        logger.info("Mapping articles to clusters:") # Log mapping start
        for article in all_articles:
            cluster_id_val = article.get('cluster_id')
            article_id = article.get('id')
            if cluster_id_val is None or article_id is None:
                continue
            cluster_id_str = str(cluster_id_val)

            logger.debug(f"  Article ID {article_id} belongs to Cluster ID {cluster_id_str}")

            if cluster_id_str not in clusters:
                clusters[cluster_id_str] = set()
            clusters[cluster_id_str].add(article_id)

        logger.info(f"Identified {len(clusters)} distinct clusters.")
        logged_count = 0
        # Log cluster IDs and counts
        for cid, ids in clusters.items():
            if logged_count < 10: # Log more examples if needed
                logger.info(f"  Final Cluster Map Example: {cid}: {len(ids)} articles - {ids}") # Show count and set
                logged_count += 1
            elif logged_count == 10:
                 logger.info("  (Logging first 10 clusters only)")
                 logged_count += 1 # Prevent logging this message repeatedly
            else:
                 pass # Don't log after the first 10

        return clusters

    except Exception as e:
        logger.error(f"Error fetching recent clusters: {e}", exc_info=True)
        return {}

async def fetch_recent_cluster_stories(time_window_days: int = 7) -> List[Dict]:
    """
    Fetches recent ClusterStories to compare against new clusters.

    Args:
        time_window_days: How many days back to look for stories based on updated_at.

    Returns:
        A list of dictionaries, each representing a ClusterStory
        with 'id', 'cluster_id', and 'source_article_ids'. Returns empty list on error.
    """
    if not _check_supabase_client():
        return []
    try:
        cutoff_time = datetime.now(timezone.utc) - timedelta(days=time_window_days)
        cutoff_iso = cutoff_time.isoformat()
        logger.info(f"Fetching cluster stories updated after: {cutoff_iso}")

        response = supabase.table("ClusterStories") \
            .select("id, cluster_id, source_article_ids") \
            .gte("updated_at", cutoff_iso) \
            .execute()

        error_info = getattr(response, 'error', None)
        if error_info:
            logger.error(f"Error fetching recent cluster stories: {error_info}")
            return []

        if response.data:
            logger.info(f"Successfully fetched {len(response.data)} recent cluster stories.")
            processed_stories = []
            for story in response.data:
                # Ensure 'id' and 'cluster_id' are strings
                story_id = str(story['id']) if story.get('id') else None
                cluster_id = str(story['cluster_id']) if story.get('cluster_id') else None

                if not story_id or not cluster_id:
                     logger.warning(f"Skipping fetched story due to missing id or cluster_id: {story}")
                     continue

                story['id'] = story_id
                story['cluster_id'] = cluster_id

                # Ensure 'source_article_ids' is a list of ints, handle null/incorrect types
                raw_ids = story.get('source_article_ids')
                clean_ids = []
                if raw_ids is None:
                    pass # Keep clean_ids empty
                elif isinstance(raw_ids, list):
                    for id_val in raw_ids:
                        try:
                            clean_ids.append(int(id_val))
                        except (ValueError, TypeError):
                             logger.warning(f"Invalid value '{id_val}' in source_article_ids for story {story_id}. Skipping value.")
                else:
                     logger.warning(f"Found non-list source_article_ids for story {story_id}: {raw_ids}. Treating as empty.")

                story['source_article_ids'] = clean_ids # Assign processed list
                processed_stories.append(story)

            return processed_stories
        else:
            logger.info("No recent cluster stories found matching the criteria.")
            return []

    except Exception as e:
        logger.error(f"Error fetching recent cluster stories: {e}", exc_info=True)
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
        return {article_id: '' for article_id in article_ids}
    content_map = {}
    if not article_ids:
        return content_map

    try:
        batch_size = 100 # Adjust batch size based on performance/URL length limits
        fetched_content_count = 0
        logger.info(f"Fetching content for {len(article_ids)} SourceArticle IDs in batches of {batch_size}...")
        for i in range(0, len(article_ids), batch_size):
            batch_ids = article_ids[i:i+batch_size]
            logger.debug(f"Fetching content for batch {i//batch_size + 1} (IDs: {batch_ids[:5]}...).")

            response = supabase.table("SourceArticles") \
                .select("id, Content") \
                .in_("id", batch_ids) \
                .execute()

            error_info = getattr(response, 'error', None)
            if error_info:
                 logger.error(f"Error fetching source article content batch {i//batch_size + 1}: {error_info}")
                 # Mark IDs in this batch as empty in the map? Or raise error?
                 for aid in batch_ids: content_map[aid] = '' # Mark as failed/empty
                 continue # Continue to next batch or stop? Let's continue.

            if response.data:
                batch_found_count = 0
                for article in response.data:
                    article_id = article.get('id')
                    if article_id is None: continue # Skip if ID is missing in response

                    content = article.get('Content') # Field name from schema
                    content_map[article_id] = content if content is not None else '' # Store empty string if content is null
                    fetched_content_count += 1
                    batch_found_count += 1

                # Log content length for the first article in the first batch as a sample check
                if i == 0 and batch_found_count > 0:
                     first_id = response.data[0].get('id')
                     if first_id:
                         logger.info(f"  Sample content length for article {first_id}: {len(content_map[first_id])} chars")
            else:
                 logger.debug(f"No content data returned for batch starting with ID {batch_ids[0]}.")

        # After processing all batches, check for missing articles
        all_fetched_ids = set(content_map.keys())
        missing_ids = set(article_ids) - all_fetched_ids
        if missing_ids:
            logger.warning(f"Could not find/fetch content for {len(missing_ids)} article IDs: {list(missing_ids)}")
            for missing_id in missing_ids:
                content_map[missing_id] = '' # Ensure entry exists, even if empty

        logger.info(f"Finished fetching content. Populated map for {len(content_map)} articles (Found non-null content for {fetched_content_count}).")
        return content_map

    except Exception as e:
        logger.error(f"Error fetching source article content: {e}", exc_info=True)
        return {article_id: '' for article_id in article_ids} # Return defaults on error


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
        logger.debug(f"Checking translation for source article {source_article_id}, language {language_code}")
        response = supabase.table("ArticleTranslations") \
            .select("translated_content") \
            .eq("source_article_id", source_article_id) \
            .eq("language_code", language_code) \
            .limit(1) \
            .execute()

        error_info = getattr(response, 'error', None)
        if error_info:
             logger.error(f"Error checking article translation for {source_article_id} ({language_code}): {error_info}")
             return None

        if response.data:
            translation = response.data[0].get("translated_content")
            if translation is not None:
                logger.debug(f"Found existing translation for article {source_article_id} ({language_code}). Length: {len(translation)}")
                return translation
            else:
                logger.debug(f"Found existing translation entry for article {source_article_id} ({language_code}), but content is null.")
                return None
        else:
            logger.debug(f"No existing translation found for article {source_article_id} ({language_code}).")
            return None
    except Exception as e:
        logger.error(f"Error checking article translation for {source_article_id} ({language_code}): {e}", exc_info=True)
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
        # Prevent inserting empty translations if desired
        if not translated_content:
             logger.warning(f"Attempted to insert empty translation for {source_article_id} ({language_code}). Skipping.")
             return False

        logger.info(f"Inserting translation for source article {source_article_id}, language {language_code}. Length: {len(translated_content)}")
        data_to_insert = {
            "source_article_id": source_article_id,
            "language_code": language_code,
            "translated_content": translated_content
        }
        response = supabase.table("ArticleTranslations").insert(data_to_insert).execute()

        error_info = getattr(response, 'error', None)
        if error_info:
             logger.error(f"Failed to insert translation for {source_article_id} ({language_code}). Error: {error_info}")
             return False

        if response.data and response.data[0].get("id"):
             logger.info(f"Successfully inserted translation for {source_article_id} ({language_code}). New ID: {response.data[0].get('id')}")
             return True
        else:
            # Might happen if RLS prevents insert but doesn't raise error, or other reasons
            logger.error(f"Failed to insert translation for {source_article_id} ({language_code}). No data/ID returned in response: {response}")
            return False

    except Exception as e:
        logger.error(f"Error inserting article translation for {source_article_id} ({language_code}): {e}", exc_info=True)
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
    missing_keys = [key for key in required_keys if key not in story_data or story_data[key] is None] # Also check for None
    # Allow image2/3 to be missing/None
    optional_images = {"image2_url", "image3_url"}
    missing_keys = [k for k in missing_keys if k not in optional_images]

    if missing_keys:
        logger.error(f"Missing or None values for required keys for inserting cluster story: {missing_keys}. Data: {story_data}")
        return None

    # --- Data Validation and Preparation ---
    try:
        # Prepare a clean dict for insertion
        db_data = {}

        # Validate and format cluster_id
        db_data['cluster_id'] = str(UUID(str(story_data['cluster_id'])))

        # Validate source_article_ids is a list of integers
        raw_ids = story_data['source_article_ids']
        if not isinstance(raw_ids, list) or not all(isinstance(i, int) for i in raw_ids):
            raise ValueError("source_article_ids must be a list of integers.")
        db_data['source_article_ids'] = raw_ids

        # Validate timestamps
        db_data['created_at'] = datetime.fromisoformat(str(story_data['created_at']).replace('Z', '+00:00')).astimezone(timezone.utc).isoformat()
        db_data['updated_at'] = datetime.fromisoformat(str(story_data['updated_at']).replace('Z', '+00:00')).astimezone(timezone.utc).isoformat()

        # Copy other validated fields (assuming they are strings or compatible)
        for key in ["headline_english", "headline_german", "summary_english", "summary_german",
                    "body_english", "body_german", "image1_url", "image2_url", "image3_url", "status"]:
             db_data[key] = story_data.get(key) # Use get for optional fields like image2/3

        # Ensure optional image URLs are None if empty string
        if db_data.get("image2_url") == "": db_data["image2_url"] = None
        if db_data.get("image3_url") == "": db_data["image3_url"] = None

    except (ValueError, TypeError, KeyError) as e:
         logger.error(f"Data validation/preparation failed for cluster story insertion: {e}. Data: {story_data}", exc_info=True)
         return None
    # --- End Validation ---

    try:
        logger.info(f"Inserting new ClusterStory for cluster_id: {db_data['cluster_id']}")
        response = supabase.table("ClusterStories").insert(db_data).execute()

        error_info = getattr(response, 'error', None)
        if error_info:
             logger.error(f"Failed to insert ClusterStory. Error: {error_info}")
             return None

        if response.data and response.data[0].get("id"):
            new_story_id = response.data[0].get("id")
            logger.info(f"Successfully inserted ClusterStory with ID: {new_story_id}")
            return str(new_story_id)
        else:
            logger.error(f"Failed to insert ClusterStory, no data/ID returned. Response: {response}")
            return None

    except Exception as e:
        logger.error(f"Exception during ClusterStory insertion for cluster {db_data.get('cluster_id')}: {e}", exc_info=True)
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
        logger.error("Missing story_id for update.")
        return False

    # --- Data Validation and Preparation ---
    try:
        UUID(story_id) # Validate story_id format

        # Prepare update dict, excluding immutable fields
        update_data = story_data.copy()
        update_data.pop('id', None)
        update_data.pop('created_at', None)
        update_data.pop('cluster_id', None) # Cluster ID should not change

        # Validate updated_at
        if 'updated_at' not in update_data:
             raise ValueError("Missing 'updated_at' timestamp in update data.")
        update_data['updated_at'] = datetime.fromisoformat(str(update_data['updated_at']).replace('Z', '+00:00')).astimezone(timezone.utc).isoformat()

        # Validate source_article_ids if present
        if 'source_article_ids' in update_data:
            raw_ids = update_data['source_article_ids']
            if not isinstance(raw_ids, list) or not all(isinstance(i, int) for i in raw_ids):
                raise ValueError("Invalid source_article_ids format for update.")
            # update_data['source_article_ids'] is already validated list

        # Ensure optional image URLs are None if empty string
        if update_data.get("image2_url") == "": update_data["image2_url"] = None
        if update_data.get("image3_url") == "": update_data["image3_url"] = None

        if not update_data: # Check if anything is left to update
            logger.warning(f"No valid fields to update for ClusterStory {story_id} after validation.")
            return False # Or True if no change needed is success?

    except (ValueError, TypeError, KeyError) as e:
         logger.error(f"Data validation/preparation failed for cluster story update (ID: {story_id}): {e}. Data: {story_data}", exc_info=True)
         return False
    # --- End Validation ---

    try:
        logger.info(f"Updating ClusterStory with ID: {story_id}")
        response = supabase.table("ClusterStories") \
            .update(update_data) \
            .eq("id", story_id) \
            .execute()

        error_info = getattr(response, 'error', None)
        if error_info:
            logger.error(f"Failed to update ClusterStory {story_id}. Error: {error_info}")
            return False

        # Check if the update returned data. If Prefer=minimal is used (not default),
        # data might be empty on success. A more reliable check might be needed,
        # e.g., checking affected rows if the client provides it.
        # For now, assume success if no error.
        if response.data:
             logger.info(f"Successfully updated ClusterStory {story_id} (data returned).")
             return True
        else:
             # Check if the row still exists to differentiate 'not found' from other issues
             logger.warning(f"ClusterStory update for {story_id} returned no data. Checking if row exists...")
             check_response = supabase.table("ClusterStories").select("id", count='exact').eq("id", story_id).execute()
             if hasattr(check_response, 'count') and check_response.count > 0:
                 logger.info(f"Successfully updated ClusterStory {story_id} (row exists, likely Prefer=minimal or no actual change).")
                 return True # Consider success if row exists and no error occurred
             else:
                 logger.error(f"Failed to update ClusterStory {story_id}: Story ID not found or check failed (Count: {getattr(check_response, 'count', 'N/A')}, Error: {getattr(check_response, 'error', 'N/A')}).")
                 return False

    except Exception as e:
        logger.error(f"Exception during ClusterStory update for {story_id}: {e}", exc_info=True)
        return False
    
async def fetch_cluster_story_details(story_id: str) -> Optional[Dict]:
    """
    Fetches specific details of an existing ClusterStory for updates.

    Args:
        story_id: The UUID string of the ClusterStory to fetch.

    Returns:
        A dictionary containing story details (headlines, images) if found,
        otherwise None.
    """
    if not _check_supabase_client():
        return None
    if not story_id:
        logger.error("fetch_cluster_story_details called with empty story_id.")
        return None

    try:
        logger.info(f"Fetching existing details for ClusterStory ID: {story_id}")
        # Select only the fields needed for reuse during update
        select_fields = "id, headline_english, headline_german, image1_url, image2_url, image3_url"

        response = supabase.table("ClusterStories") \
            .select(select_fields) \
            .eq("id", story_id) \
            .limit(1) \
            .execute()

        if response.data:
            logger.info(f"Successfully fetched details for ClusterStory {story_id}.")
            return response.data[0]
        else:
            logger.error(f"Could not find existing ClusterStory with ID: {story_id}. Cannot perform update.")
            error_info = getattr(response, 'error', None)
            if error_info:
                logger.error(f"Supabase error details: {error_info}")
            return None

    except Exception as e:
        logger.error(f"Error fetching details for cluster story {story_id}: {e}", exc_info=True)
        return None

# --- End of New Cluster Functions ---
