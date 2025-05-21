import os
import requests
from typing import List, Dict, Optional, Set, Union # Added Set, Union
from dotenv import load_dotenv
from supabase import create_client, Client
# Add imports for datetime, timedelta, UUID if not already present
from datetime import datetime, timedelta, timezone
from uuid import UUID, uuid4 # Added uuid4 for generating new UUIDs if needed for cluster_articles.id
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
        error_info = getattr(response, 'error', None)
        if error_info:
            logger.error(f"Error marking article {article_id} as processed: {error_info}")
            return False
        logger.info(f"Successfully marked SourceArticle {article_id} as processed.")
        return True
    except Exception as e:
        logger.error(f"Exception marking article {article_id} as processed: {e}", exc_info=True)
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
        vector_response = supabase.table("ArticleVector").select("update").eq("SourceArticle", source_article_id).limit(1).execute()

        if vector_response.data and vector_response.data[0].get("update"):
            updated_source_ids = vector_response.data[0]["update"] # This is expected to be a list of IDs
            if updated_source_ids and isinstance(updated_source_ids, list):
                logger.info(f"SourceArticle {source_article_id} updates {len(updated_source_ids)} other articles. Setting their UpdatedBy to {article_id}.")
                update_response = supabase.table("NewsArticles").update(
                    {"UpdatedBy": article_id}
                ).in_("SourceArticle", updated_source_ids).execute()

                error_info = getattr(update_response, 'error', None)
                if error_info:
                     logger.error(f"Error updating UpdatedBy field for articles {updated_source_ids}: {error_info}")
                     overall_success = False
                else:
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
        is_update = await check_for_updates(source_article_id)
        article_data["isUpdate"] = is_update
        logger.info(f"isUpdate status for SourceArticle {source_article_id}: {is_update}")

        if "created_at" not in article_data or not article_data["created_at"]:
            article_data["created_at"] = datetime.now(timezone.utc).isoformat()
        else:
            try:
                 article_data['created_at'] = datetime.fromisoformat(str(article_data['created_at']).replace('Z', '+00:00')).astimezone(timezone.utc).isoformat()
            except (ValueError, TypeError):
                 logger.warning(f"Invalid created_at format provided: {article_data['created_at']}. Using current time.")
                 article_data["created_at"] = datetime.now(timezone.utc).isoformat()

        allowed_columns = {
            "created_at", "headlineEnglish", "headlineGerman", "SummaryEnglish",
            "SummaryGerman", "ContentEnglish", "ContentGerman", "Image1", "Image2",
            "Image3", "SourceArticle", "team", "isUpdate", "UpdatedBy", "status"
        }
        db_data = {k: v for k, v in article_data.items() if k in allowed_columns}

        response = supabase.table("NewsArticles").insert(db_data).execute()

        if response.data and response.data[0].get("id"):
            article_id = response.data[0]["id"]
            logger.info(f"Successfully inserted NewsArticle with ID {article_id} for SourceArticle {source_article_id}.")

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
    page_size = 1000 
    total_count = None 

    try:
        logger.info("Starting batch update of NewsArticles.isUpdate status...")

        while True:
            logger.info(f"Fetching NewsArticles page {page + 1}...")
            current_count_method = 'exact' if page == 0 else None
            response = supabase.table("NewsArticles") \
                .select("id, SourceArticle, isUpdate", count=current_count_method) \
                .range(page * page_size, (page + 1) * page_size - 1) \
                .execute()

            if hasattr(response, 'error') and response.error:
                 logger.error(f"Error fetching NewsArticles page {page + 1}: {response.error}")
                 stats["errors"] += 1
                 break 

            if page == 0:
                total_count = getattr(response, 'count', 0) if hasattr(response, 'count') else len(response.data)
                if total_count is None: total_count = 0 
                logger.info(f"Total articles to process: {total_count}")
                stats["total"] = total_count
                if not response.data and total_count == 0:
                     logger.info("No articles found in NewsArticles table.")
                     return stats

            if not response.data:
                 logger.info("No more articles found.")
                 break 

            articles = response.data
            logger.info(f"Processing {len(articles)} articles from page {page + 1}...")

            source_ids = [article["SourceArticle"] for article in articles if article.get("SourceArticle") is not None]

            if not source_ids:
                logger.info("No SourceArticle IDs found in this batch.")
                page += 1
                continue

            logger.debug(f"Fetching ArticleVector entries for {len(source_ids)} source IDs...")
            vector_response = supabase.table("ArticleVector") \
                .select("SourceArticle, update") \
                .in_("SourceArticle", source_ids) \
                .execute()

            vector_update_status = {}
            if hasattr(vector_response, 'error') and vector_response.error:
                 logger.error(f"Error fetching ArticleVector batch: {vector_response.error}")
                 stats["errors"] += len(articles) 
            elif vector_response.data:
                for vector_entry in vector_response.data:
                    vector_update_status[vector_entry["SourceArticle"]] = bool(vector_entry.get("update"))
            logger.debug(f"Found ArticleVector update status for {len(vector_update_status)} source IDs.")

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

            if updates_to_make:
                logger.info(f"Attempting to update {len(updates_to_make)} articles individually...")
                for update_data in updates_to_make:
                    try:
                        update_response = supabase.table("NewsArticles") \
                            .update({"isUpdate": update_data["isUpdate"]}) \
                            .eq("id", update_data["id"]) \
                            .execute()
                        error_info = getattr(update_response, 'error', None)
                        if error_info:
                             logger.error(f"Error updating article {update_data['id']}: {error_info}")
                             stats["errors"] += 1
                        else:
                             stats["updated"] += 1
                             logger.debug(f"Successfully updated article {update_data['id']} isUpdate status to {update_data['isUpdate']}")
                    except Exception as e_update:
                         logger.error(f"Exception updating article {update_data['id']}: {e_update}", exc_info=True)
                         stats["errors"] += 1
            else:
                 logger.info(f"No status changes needed for articles on page {page + 1}.")

            page += 1 

        logger.info(f"Batch update processing complete. Final Stats: {stats}")
        return stats

    except Exception as e:
        logger.error(f"Error during batch update of article status: {e}", exc_info=True)
        stats["errors"] += 1 
        return stats


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
        batch_size = 100 
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
                 for aid in batch_ids: content_map[aid] = '' 
                 continue 

            if response.data:
                batch_found_count = 0
                for article in response.data:
                    article_id = article.get('id')
                    if article_id is None: continue 

                    content = article.get('Content') 
                    content_map[article_id] = content if content is not None else '' 
                    fetched_content_count += 1
                    batch_found_count += 1

                if i == 0 and batch_found_count > 0:
                     first_id = response.data[0].get('id')
                     if first_id:
                         logger.info(f"  Sample content length for article {first_id}: {len(content_map[first_id])} chars")
            else:
                 logger.debug(f"No content data returned for batch starting with ID {batch_ids[0]}.")

        all_fetched_ids = set(content_map.keys())
        missing_ids = set(article_ids) - all_fetched_ids
        if missing_ids:
            logger.warning(f"Could not find/fetch content for {len(missing_ids)} article IDs: {list(missing_ids)}")
            for missing_id in missing_ids:
                content_map[missing_id] = '' 

        logger.info(f"Finished fetching content. Populated map for {len(content_map)} articles (Found non-null content for {fetched_content_count}).")
        return content_map

    except Exception as e:
        logger.error(f"Error fetching source article content: {e}", exc_info=True)
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
            logger.error(f"Failed to insert translation for {source_article_id} ({language_code}). No data/ID returned in response: {response}")
            return False

    except Exception as e:
        logger.error(f"Error inserting article translation for {source_article_id} ({language_code}): {e}", exc_info=True)
        return False

# --- START: New functions for Cluster Story Processing ---

async def fetch_clusters_to_process(status: str) -> List[Dict]:
    """
    Fetches clusters that need processing based on status and isContent flag.

    Args:
        status: The cluster status to filter by (e.g., 'NEW', 'UPDATED').

    Returns:
        List of cluster dictionaries (containing at least 'cluster_id').
    """
    if not _check_supabase_client():
        return []
    try:
        logger.info(f"Fetching clusters with status '{status}' and isContent=false.")
        response = supabase.table("clusters") \
            .select("cluster_id") \
            .eq("status", status) \
            .eq("isContent", False) \
            .execute()

        error_info = getattr(response, 'error', None)
        if error_info:
            logger.error(f"Error fetching clusters to process (status {status}): {error_info}")
            return []
        
        if response.data:
            logger.info(f"Found {len(response.data)} clusters with status '{status}' and isContent=false.")
            return response.data
        else:
            logger.info(f"No clusters found with status '{status}' and isContent=false.")
            return []
    except Exception as e:
        logger.error(f"Exception fetching clusters to process (status {status}): {e}", exc_info=True)
        return []

async def fetch_source_articles_for_cluster(cluster_id: Union[str, UUID]) -> List[Dict]:
    """
    Fetches source articles associated with a given cluster_id.

    Args:
        cluster_id: The UUID of the cluster.

    Returns:
        List of source article dictionaries, ordered by created_at ascending.
        Each dictionary contains 'id', 'headline', 'Content', 'created_at'.
    """
    if not _check_supabase_client():
        return []
    try:
        logger.info(f"Fetching source articles for cluster_id: {cluster_id}")
        response = supabase.table("SourceArticles") \
            .select("id, headline, Content, created_at") \
            .eq("cluster_id", str(cluster_id)) \
            .order("created_at", desc=False) \
            .execute() # Default order is ascending, explicitly setting for clarity

        error_info = getattr(response, 'error', None)
        if error_info:
            logger.error(f"Error fetching source articles for cluster {cluster_id}: {error_info}")
            return []
        
        if response.data:
            logger.info(f"Found {len(response.data)} source articles for cluster {cluster_id}.")
            return response.data
        else:
            logger.info(f"No source articles found for cluster {cluster_id}.")
            return []
    except Exception as e:
        logger.error(f"Exception fetching source articles for cluster {cluster_id}: {e}", exc_info=True)
        return []

async def get_existing_cluster_article(cluster_id: Union[str, UUID]) -> Optional[Dict]:
    """
    Fetches the most recent existing synthesized article for a cluster.

    Args:
        cluster_id: The UUID of the cluster.

    Returns:
        A dictionary containing the cluster article data ('id', 'headline', 'summary', 'content', 'source_article_ids')
        or None if not found.
    """
    if not _check_supabase_client():
        return None
    try:
        logger.info(f"Fetching existing synthesized article for cluster_id: {cluster_id}")
        response = supabase.table("cluster_articles") \
            .select("id, headline, summary, content, source_article_ids, created_at, updated_at") \
            .eq("cluster_id", str(cluster_id)) \
            .order("updated_at", desc=True) \
            .limit(1) \
            .execute()

        error_info = getattr(response, 'error', None)
        if error_info:
            logger.error(f"Error fetching existing cluster article for {cluster_id}: {error_info}")
            return None
        
        if response.data:
            logger.info(f"Found existing synthesized article for cluster {cluster_id}.")
            return response.data[0]
        else:
            logger.info(f"No existing synthesized article found for cluster {cluster_id}.")
            return None
    except Exception as e:
        logger.error(f"Exception fetching existing cluster article for {cluster_id}: {e}", exc_info=True)
        return None

async def insert_cluster_article(
    cluster_id: Union[str, UUID], 
    source_article_ids: List[int], 
    article_data: Dict
) -> Optional[UUID]:
    """
    Inserts a new synthesized article into the cluster_articles table.

    Args:
        cluster_id: The UUID of the cluster.
        source_article_ids: List of SourceArticle IDs used for synthesis.
        article_data: Dictionary containing 'headline', 'summary', 'content'.

    Returns:
        The UUID (id) of the newly inserted cluster article, or None on failure.
    """
    if not _check_supabase_client():
        return None
    try:
        new_article_id = uuid4() # Generate a new UUID for the cluster_article
        logger.info(f"Inserting new synthesized article for cluster_id: {cluster_id} with new ID: {new_article_id}")
        
        data_to_insert = {
            "id": str(new_article_id),
            "cluster_id": str(cluster_id),
            "source_article_ids": source_article_ids,
            "headline": article_data.get("headline"),
            "summary": article_data.get("summary"),
            "content": article_data.get("content"),
            # image_url and image2_url will be null by default or can be added if available
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        
        # Validate required fields from article_data
        if not all([data_to_insert["headline"], data_to_insert["summary"], data_to_insert["content"]]):
            logger.error(f"Missing headline, summary, or content for inserting cluster article for cluster {cluster_id}.")
            return None

        response = supabase.table("cluster_articles").insert(data_to_insert).execute()

        error_info = getattr(response, 'error', None)
        if error_info:
            # Specific check for unique constraint violation if cluster_id was meant to be unique
            # This depends on your actual DB schema and whether cluster_id in cluster_articles is unique
            if "unique constraint" in str(error_info).lower() and "cluster_id" in str(error_info).lower():
                 logger.error(f"Unique constraint violation on cluster_id {cluster_id} when inserting. Consider using an upsert if this is intended.")
            logger.error(f"Error inserting cluster article for cluster {cluster_id}: {error_info}")
            return None
        
        if response.data and response.data[0].get("id"):
            inserted_id = UUID(response.data[0]["id"])
            logger.info(f"Successfully inserted cluster article with ID: {inserted_id} for cluster {cluster_id}.")
            return inserted_id
        elif response.data : # if data is present but no ID (should not happen with default return=representation)
            logger.warning(f"Inserted cluster article for cluster {cluster_id}, but ID not found in response data[0]: {response.data[0]}")
            return new_article_id # return the one we generated
        else:
            logger.error(f"Failed to insert cluster article for cluster {cluster_id}. No data returned in response: {response}")
            return None

    except Exception as e:
        logger.error(f"Exception inserting cluster article for cluster {cluster_id}: {e}", exc_info=True)
        return None

async def update_cluster_article(
    cluster_article_id: Union[str, UUID], 
    source_article_ids: List[int], 
    article_data: Dict
) -> bool:
    """
    Updates an existing synthesized article in the cluster_articles table.

    Args:
        cluster_article_id: The UUID (id) of the cluster_article to update.
        source_article_ids: List of SourceArticle IDs used for the updated synthesis.
        article_data: Dictionary containing 'headline', 'summary', 'content'.

    Returns:
        True if update was successful, False otherwise.
    """
    if not _check_supabase_client():
        return False
    try:
        logger.info(f"Updating synthesized article with ID: {cluster_article_id}")
        
        data_to_update = {
            "source_article_ids": source_article_ids,
            "headline": article_data.get("headline"),
            "summary": article_data.get("summary"),
            "content": article_data.get("content"),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

        if not all([data_to_update["headline"], data_to_update["summary"], data_to_update["content"]]):
            logger.error(f"Missing headline, summary, or content for updating cluster article {cluster_article_id}.")
            return False

        response = supabase.table("cluster_articles") \
            .update(data_to_update) \
            .eq("id", str(cluster_article_id)) \
            .execute()

        error_info = getattr(response, 'error', None)
        if error_info:
            logger.error(f"Error updating cluster article {cluster_article_id}: {error_info}")
            return False
        
        # According to Supabase Python client, a successful update with Prefer: return=representation (default)
        # will return data. If Prefer: return=minimal, data will be empty on success.
        # We can assume success if no error is raised/returned.
        logger.info(f"Successfully updated cluster article {cluster_article_id}.")
        return True
        
    except Exception as e:
        logger.error(f"Exception updating cluster article {cluster_article_id}: {e}", exc_info=True)
        return False

async def mark_cluster_content_processed(cluster_id: Union[str, UUID]) -> bool:
    """
    Marks a cluster's isContent flag to True.

    Args:
        cluster_id: The UUID of the cluster.

    Returns:
        True if successful, False otherwise.
    """
    if not _check_supabase_client():
        return False
    try:
        logger.info(f"Marking cluster {cluster_id} as content processed (isContent=true).")
        response = supabase.table("clusters") \
            .update({"isContent": True}) \
            .eq("cluster_id", str(cluster_id)) \
            .execute()

        error_info = getattr(response, 'error', None)
        if error_info:
            logger.error(f"Error marking cluster {cluster_id} as content processed: {error_info}")
            return False
        
        logger.info(f"Successfully marked cluster {cluster_id} as content processed.")
        return True
    except Exception as e:
        logger.error(f"Exception marking cluster {cluster_id} as content processed: {e}", exc_info=True)
        return False
    
async def update_cluster_article_images(
    cluster_article_id: Union[str, UUID], 
    image_url: Optional[str] = None, 
    image2_url: Optional[str] = None
) -> bool:
    """
    Updates an existing cluster_articles record with image URLs.

    Args:
        cluster_article_id: The UUID (id) of the cluster_article to update.
        image_url: URL for the primary image.
        image2_url: URL for the secondary image.

    Returns:
        True if update was successful, False otherwise.
    """
    if not _check_supabase_client():
        return False
    
    if not cluster_article_id:
        logger.error("Cannot update cluster article images: cluster_article_id is missing.")
        return False

    # Only include fields in the update if they are provided (not None)
    # to avoid overwriting existing image URLs with None if only one new image is found.
    data_to_update = {}
    if image_url is not None: # Allow empty string to clear an image if needed
        data_to_update["image_url"] = image_url
    if image2_url is not None: # Allow empty string
        data_to_update["image2_url"] = image2_url
    
    if not data_to_update:
        logger.info(f"No new image URLs provided for cluster article {cluster_article_id}. Skipping image update.")
        return True # No update needed, considered success for this operation

    # Add updated_at timestamp
    data_to_update["updated_at"] = datetime.now(timezone.utc).isoformat()
    
    try:
        logger.info(f"Updating images for cluster_article ID: {cluster_article_id} with data: {data_to_update}")
        response = supabase.table("cluster_articles") \
            .update(data_to_update) \
            .eq("id", str(cluster_article_id)) \
            .execute()

        error_info = getattr(response, 'error', None)
        if error_info:
            logger.error(f"Error updating images for cluster article {cluster_article_id}: {error_info}")
            return False
        
        logger.info(f"Successfully updated images for cluster article {cluster_article_id}.")
        return True
        
    except Exception as e:
        logger.error(f"Exception updating images for cluster article {cluster_article_id}: {e}", exc_info=True)
        return False