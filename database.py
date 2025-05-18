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

# Team article functions removed (deprecated)

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

# Team article functions removed (deprecated)

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


# Team-related function removed (deprecated)

# Team article function removed (deprecated)


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


# --- All cluster story creation, fetching, updating, and matching logic has been removed as it is now handled in a different repository. ---

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
