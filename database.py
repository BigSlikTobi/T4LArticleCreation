import os
import requests
from typing import List, Dict, Optional, Set, Union 
from dotenv import load_dotenv
from supabase import create_client, Client
from postgrest.exceptions import APIError # Import APIError
from datetime import datetime, timedelta, timezone
from uuid import UUID, uuid4 
import logging 

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    force=True 
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize Supabase client
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_KEY")

supabase: Optional[Client] = None
if supabase_url and supabase_key:
    try:
        supabase = create_client(supabase_url, supabase_key) # Assuming this is how client is created
    except Exception as e:
        logger.critical(f"Failed to initialize Supabase client: {e}") # Added logging for failure
else:
    logger.critical("Supabase URL or Key not found in environment variables. Database functions will fail.")


# Helper function to check if Supabase client is available
def _check_supabase_client():
    """
    Checks if the Supabase client is initialized and available.

    Returns:
        bool: True if the client is available, False otherwise.
    """
    if supabase is None:
        logger.error("Supabase client is not initialized.") # Added logging
    return True

async def fetch_primary_sources() -> List[int]:
    """
    Fetches all primary news sources from the NewsSource table.

    Returns:
        List[int]: List of primary news source IDs.
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
        response = requests.get(url, headers=headers, params=params) # Using sync requests here, consider aiohttp for full async
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
    Fetches articles from the SourceArticles table that meet the criteria for processing.

    Returns:
        List[Dict]: List of unprocessed article records as dictionaries.
    """
    if not _check_supabase_client() or not supabase_key or not supabase_url:
        logger.warning("Cannot fetch unprocessed articles: Supabase client/config missing.")
        return []
    try:
        primary_sources = await fetch_primary_sources()
        if not primary_sources:
            logger.warning("No primary sources found, using fallback sources [1, 2, 4]")
            primary_sources = [1, 2, 4]
        logger.info("Fetching unprocessed articles from database...")
        headers = {
            "apikey": supabase_key,
            "Authorization": f"Bearer {supabase_key}"
        }
        url = f"{supabase_url}/rest/v1/SourceArticles"
        source_list = f"({','.join(map(str, primary_sources))})"
        params = {
            "select": "*",
            "source": f"in.{source_list}",
            "contentType": "eq.news_article",
            "isArticleCreated": "eq.false",
            "duplication_of": "is.null"
        }
        response = requests.get(url, headers=headers, params=params) # Using sync requests
        if response.status_code == 200:
            articles = response.json()
            filtered_articles = [
                article for article in articles
                if article.get('source') in primary_sources
                and article.get('contentType') == 'news_article'
                and not article.get('isArticleCreated', False)
                and (article.get('duplication_of') is None or article.get('duplication_of') == '')
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
        List[Dict]: List of team records as dictionaries.
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
        params = {"select": "id,fullName"}
        response = requests.get(url, headers=headers, params=params) # Using sync requests
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

async def mark_article_as_processed(article_id: int, article_created: bool = True) -> bool:
    """
    Marks an article as processed in the SourceArticles table.

    Args:
        article_id (int): The ID of the article to mark as processed.
        article_created (bool): Whether the article was successfully created and saved.

    Returns:
        bool: True if the update was successful, False otherwise.
    """
    if not _check_supabase_client(): return False
    try:
        logger.info(f"Marking SourceArticle {article_id} as processed, article_created: {article_created}.")
        data = {"isArticleCreated": article_created}  # Use the passed in status
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
    Checks if the article should be marked as an update via ArticleVector table.

    Args:
        source_article_id (int): The ID of the source article to check for updates.

    Returns:
        bool: True if the article is an update, False otherwise.
    """
    if not _check_supabase_client(): return False
    try:
        logger.debug(f"Checking for updates status for SourceArticle {source_article_id}.")
        response = supabase.table("ArticleVector").select("update").eq("SourceArticle", source_article_id).limit(1).execute()
        if response.data:
            is_update = bool(response.data[0].get("update"))
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
    Updates the UpdatedBy field for articles updated by the current article.

    Args:
        article_id (int): The ID of the article to update.
        source_article_id (int): The ID of the source article that updated this article.

    Returns:
        bool: True if the update was successful, False otherwise.
    """
    if not _check_supabase_client(): return False
    overall_success = True
    try:
        logger.info(f"Checking ArticleVector for articles updated by SourceArticle {source_article_id}.")
        vector_response = supabase.table("ArticleVector").select("update").eq("SourceArticle", source_article_id).limit(1).execute()
        if vector_response.data and vector_response.data[0].get("update"):
            updated_source_ids = vector_response.data[0]["update"]
            if updated_source_ids and isinstance(updated_source_ids, list):
                logger.info(f"SourceArticle {source_article_id} updates {len(updated_source_ids)} other articles. Setting their UpdatedBy to {article_id}.")
                update_response = supabase.table("NewsArticles").update({"UpdatedBy": article_id}).in_("SourceArticle", updated_source_ids).execute()
                error_info = getattr(update_response, 'error', None)
                if error_info:
                     logger.error(f"Error updating UpdatedBy field for articles {updated_source_ids}: {error_info}")
                     overall_success = False
                else:
                     logger.info(f"UpdatedBy update executed for articles updated by {source_article_id}. Response data: {len(update_response.data) if hasattr(update_response, 'data') else 'N/A'}")
            else:
                logger.info(f"SourceArticle {source_article_id} has an 'update' field but it's empty/not a list.")
        else:
            logger.info(f"SourceArticle {source_article_id} does not update any other articles.")
        return overall_success
    except Exception as e:
        logger.error(f"Error processing update_articles_updated_by for source {source_article_id}: {e}", exc_info=True)
        return False

async def insert_processed_article(article_data: Dict) -> Optional[int]:
    """
    Inserts a processed article into the NewsArticles table.

    Args:
        article_data (Dict): Dictionary containing article data to insert.

    Returns:
        Optional[int]: The ID of the inserted article, or None if insertion failed.
    """
    if not _check_supabase_client(): return None
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
            try: article_data['created_at'] = datetime.fromisoformat(str(article_data['created_at']).replace('Z', '+00:00')).astimezone(timezone.utc).isoformat()
            except (ValueError, TypeError):
                 logger.warning(f"Invalid created_at format: {article_data['created_at']}. Using current time.")
                 article_data["created_at"] = datetime.now(timezone.utc).isoformat()
        allowed_columns = {"created_at", "headlineEnglish", "headlineGerman", "SummaryEnglish", "SummaryGerman", 
                           "ContentEnglish", "ContentGerman", "Image1", "Image2", "Image3", "SourceArticle", 
                           "team", "isUpdate", "UpdatedBy", "status"}
        db_data = {k: v for k, v in article_data.items() if k in allowed_columns}
        response = supabase.table("NewsArticles").insert(db_data).execute()
        if response.data and response.data[0].get("id"):
            article_id = response.data[0]["id"]
            logger.info(f"Successfully inserted NewsArticle ID {article_id} for SourceArticle {source_article_id}.")
            if is_update: await update_articles_updated_by(article_id, source_article_id)
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
    Batch updates isUpdate status for NewsArticles.

    Returns:
        Dict: Statistics about the batch update (total, updated, errors).
    """
    if not _check_supabase_client(): return {"total": 0, "updated": 0, "errors": 1, "message": "Supabase client not initialized"}
    stats = {"total": 0, "updated": 0, "errors": 0}
    page, page_size, total_count = 0, 1000, None
    try:
        logger.info("Starting batch update of NewsArticles.isUpdate status...")
        while True:
            logger.info(f"Fetching NewsArticles page {page + 1}...")
            current_count_method = 'exact' if page == 0 else None
            response = supabase.table("NewsArticles").select("id, SourceArticle, isUpdate", count=current_count_method).range(page * page_size, (page + 1) * page_size - 1).execute()
            if hasattr(response, 'error') and response.error:
                 logger.error(f"Error fetching NewsArticles page {page + 1}: {response.error}"); stats["errors"] += 1; break
            if page == 0:
                total_count = getattr(response, 'count', 0) if hasattr(response, 'count') else len(response.data)
                if total_count is None: total_count = 0
                logger.info(f"Total articles to process: {total_count}"); stats["total"] = total_count
                if not response.data and total_count == 0: logger.info("No articles found in NewsArticles."); return stats
            if not response.data: logger.info("No more articles found."); break
            articles = response.data
            source_ids = [article["SourceArticle"] for article in articles if article.get("SourceArticle") is not None]
            if not source_ids: logger.info("No SourceArticle IDs in batch."); page += 1; continue
            vector_response = supabase.table("ArticleVector").select("SourceArticle, update").in_("SourceArticle", source_ids).execute()
            vector_update_status = {}
            if hasattr(vector_response, 'error') and vector_response.error:
                 logger.error(f"Error fetching ArticleVector batch: {vector_response.error}"); # Decide on error impact
            elif vector_response.data:
                for entry in vector_response.data: vector_update_status[entry["SourceArticle"]] = bool(entry.get("update"))
            updates_to_make = []
            for article in articles:
                if article.get("SourceArticle") is None: continue
                should_be_update = vector_update_status.get(article["SourceArticle"], False)
                if article.get("isUpdate") != should_be_update:
                    updates_to_make.append({"id": article["id"], "isUpdate": should_be_update})
            if updates_to_make:
                logger.info(f"Updating {len(updates_to_make)} articles individually...")
                for update_data in updates_to_make:
                    try:
                        update_resp = supabase.table("NewsArticles").update({"isUpdate": update_data["isUpdate"]}).eq("id", update_data["id"]).execute()
                        if getattr(update_resp, 'error', None):
                             logger.error(f"Error updating article {update_data['id']}: {update_resp.error}"); stats["errors"] += 1
                        else: stats["updated"] += 1
                    except Exception as e_upd: logger.error(f"Exception updating article {update_data['id']}: {e_upd}"); stats["errors"] += 1
            page += 1
        logger.info(f"Batch update complete. Stats: {stats}"); return stats
    except Exception as e:
        logger.error(f"Error in batch_update_article_status: {e}", exc_info=True); stats["errors"] += 1; return stats

async def get_source_articles_content(article_ids: List[int]) -> Dict[int, str]:
    """
    Fetches English content ('Content') for a list of SourceArticle IDs.

    Args:
        article_ids (List[int]): List of source article IDs to fetch content for.

    Returns:
        Dict[int, str]: Mapping from article ID to content string.
    """
    if not _check_supabase_client(): return {aid: '' for aid in article_ids}
    content_map = {}; 
    if not article_ids: return content_map
    try:
        batch_size, fetched_count = 100, 0
        logger.info(f"Fetching content for {len(article_ids)} SourceArticle IDs...")
        for i in range(0, len(article_ids), batch_size):
            batch = article_ids[i:i+batch_size]
            response = supabase.table("SourceArticles").select("id, Content").in_("id", batch).execute()
            if getattr(response, 'error', None):
                 logger.error(f"Error fetching source content batch: {response.error}"); continue
            if response.data:
                for art in response.data:
                    content_map[art['id']] = art.get('Content', '')
                    fetched_count += 1
        missing = set(article_ids) - set(content_map.keys())
        if missing: logger.warning(f"Could not find content for {len(missing)} IDs: {list(missing)[:5]}...")
        for mid in missing: content_map[mid] = ''
        logger.info(f"Fetched content for {fetched_count} articles. Map size: {len(content_map)}.")
        return content_map
    except Exception as e:
        logger.error(f"Error in get_source_articles_content: {e}", exc_info=True)
        return {aid: '' for aid in article_ids}

async def get_article_translation(source_article_id: int, language_code: str = 'de') -> Optional[str]:
    """
    Checks ArticleTranslations table for existing translation.

    Args:
        source_article_id (int): The ID of the source article.
        language_code (str): The language code to check for (default: 'de').

    Returns:
        Optional[str]: The translated content if found, else None.
    """
    # This function seems to be for the old 'ArticleTranslations' table, 
    # distinct from the new 'cluster_article_int' logic. Keep if still used.
    if not _check_supabase_client(): return None
    try:
        logger.debug(f"Checking ArticleTranslations for source_article_id {source_article_id}, lang {language_code}")
        response = supabase.table("ArticleTranslations").select("translated_content").eq("source_article_id", source_article_id).eq("language_code", language_code).limit(1).execute()
        if getattr(response, 'error', None):
             logger.error(f"Error checking ArticleTranslations: {response.error}"); return None
        if response.data and response.data[0].get("translated_content") is not None:
            return response.data[0]["translated_content"]
        return None
    except Exception as e:
        logger.error(f"Error in get_article_translation: {e}", exc_info=True); return None

async def insert_article_translation(source_article_id: int, language_code: str, translated_content: str) -> bool:
    """
    Inserts a translation into the ArticleTranslations table.

    Args:
        source_article_id (int): The ID of the source article.
        language_code (str): The language code for the translation.
        translated_content (str): The translated content to insert.

    Returns:
        bool: True if the insertion was successful, False otherwise.
    """
    # This function also seems for the old 'ArticleTranslations' table.
    if not _check_supabase_client(): return False
    if not translated_content: logger.warning("Empty translation content, skipping insert."); return False
    try:
        data = {"source_article_id": source_article_id, "language_code": language_code, "translated_content": translated_content}
        response = supabase.table("ArticleTranslations").insert(data).execute()
        if getattr(response, 'error', None):
             logger.error(f"Error inserting to ArticleTranslations: {response.error}"); return False
        return bool(response.data and response.data[0].get("id"))
    except Exception as e:
        logger.error(f"Error in insert_article_translation: {e}", exc_info=True); return False

# --- START: Functions for Cluster Story Processing ---

async def fetch_clusters_to_process(status: str) -> List[Dict]:
    """
    Fetches clusters to process based on the given status.

    Args:
        status (str): The status to filter clusters by.

    Returns:
        List[Dict]: List of cluster records as dictionaries.
    """
    if not _check_supabase_client(): return []
    try:
        logger.info(f"Fetching clusters: status='{status}', isContent=false.")
        response = supabase.table("clusters").select("cluster_id").eq("status", status).eq("isContent", False).execute()
        if getattr(response, 'error', None):
            logger.error(f"Error fetching clusters to process: {response.error}"); return []
        logger.info(f"Found {len(response.data)} clusters for status '{status}'.")
        return response.data if response.data else []
    except Exception as e:
        logger.error(f"Exception fetching clusters: {e}", exc_info=True); return []

async def fetch_source_articles_for_cluster(cluster_id: Union[str, UUID]) -> List[Dict]:
    """
    Fetches source articles for a given cluster_id, ordered by creation date.
    Ensures 'id', 'headline', 'Content', 'created_at', and source name are selected.
    Joins with NewsSource table to get the source name.

    Args:
        cluster_id (Union[str, UUID]): The cluster ID to fetch articles for.

    Returns:
        List[Dict]: List of source article records as dictionaries.
    """
    if not _check_supabase_client() or supabase is None:
        logger.error(f"Supabase client not initialized. Cannot fetch articles for cluster {cluster_id}.")
        return []
    try:
        # Explicitly include id field in the select statement
        logger.info(f"Fetching source articles for cluster: {cluster_id}")
        response = supabase.table("SourceArticles") \
            .select("id, headline, Content, created_at, NewsSource(Name)") \
            .eq("cluster_id", str(cluster_id)) \
            .order("created_at", desc=False) \
            .execute()
        
        # Debug log to verify the structure
        if response.data and len(response.data) > 0:
            logger.info(f"Successfully fetched {len(response.data)} articles for cluster {cluster_id}")
            logger.debug(f"First article keys: {list(response.data[0].keys())}")
            logger.debug(f"First article ID: {response.data[0].get('id', 'ID_NOT_FOUND')}")
        else:
            logger.warning(f"No articles found for cluster {cluster_id}")
            
        return response.data if response.data else []
    except APIError as e:
        logger.error(f"APIError fetching source articles for cluster {cluster_id}: {e.message} - {e.details}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error fetching source articles for cluster {cluster_id}: {e}")
        return []

async def get_existing_cluster_article(cluster_id: Union[str, UUID]) -> Optional[Dict]:
    """
    Fetches the existing synthesized article for a given cluster ID, if any.

    Args:
        cluster_id (Union[str, UUID]): The cluster ID to fetch the article for.

    Returns:
        Optional[Dict]: The article record if found, else None.
    """
    if not _check_supabase_client(): return None
    try:
        logger.info(f"Fetching existing synthesized article for cluster_id: {cluster_id}")
        response = supabase.table("cluster_articles").select("id, headline, summary, content, source_article_ids, created_at, updated_at").eq("cluster_id", str(cluster_id)).order("updated_at", desc=True).limit(1).execute()
        if getattr(response, 'error', None):
            logger.error(f"Error fetching existing cluster article for {cluster_id}: {response.error}"); return None
        return response.data[0] if response.data else None
    except Exception as e:
        logger.error(f"Exception fetching existing cluster article: {e}", exc_info=True); return None

async def insert_cluster_article(cluster_id: Union[str, UUID], source_article_ids: List[int], article_data: Dict) -> Optional[UUID]:
    """
    Inserts a synthesized article for a cluster into the database.

    Args:
        cluster_id (Union[str, UUID]): The cluster ID for the article.
        source_article_ids (List[int]): List of source article IDs used for synthesis.
        article_data (Dict): The synthesized article data.

    Returns:
        Optional[UUID]: The UUID of the inserted cluster article, or None if insertion failed.
    """
    if not _check_supabase_client(): return None
    try:
        new_article_id = uuid4()
        logger.info(f"Inserting new synthesized cluster article ID: {new_article_id} for cluster_id: {cluster_id}")
        data_to_insert = {
            "id": str(new_article_id), "cluster_id": str(cluster_id), "source_article_ids": source_article_ids,
            "headline": article_data.get("headline"), "summary": article_data.get("summary"), "content": article_data.get("content"),
            "created_at": datetime.now(timezone.utc).isoformat(), "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        if not all(data_to_insert[k] for k in ["headline", "summary", "content"]):
            logger.error(f"Missing data for inserting cluster article for {cluster_id}."); return None
        response = supabase.table("cluster_articles").insert(data_to_insert).execute()
        if getattr(response, 'error', None):
            logger.error(f"Error inserting cluster article for {cluster_id}: {response.error}"); return None
        if response.data and response.data[0].get("id"): return UUID(response.data[0]["id"])
        logger.warning(f"Cluster article inserted for {cluster_id}, but no ID in response. Returning generated ID {new_article_id}.")
        return new_article_id # Fallback to generated ID if response format is unexpected but no error
    except Exception as e:
        logger.error(f"Exception inserting cluster article: {e}", exc_info=True); return None

async def update_cluster_article(cluster_article_id: Union[str, UUID], source_article_ids: List[int], article_data: Dict) -> bool:
    """
    Updates a synthesized article for a cluster in the database.

    Args:
        cluster_article_id (Union[str, UUID]): The UUID of the cluster article to update.
        source_article_ids (List[int]): List of source article IDs used for synthesis.
        article_data (Dict): The updated article data.

    Returns:
        bool: True if the update was successful, False otherwise.
    """
    if not _check_supabase_client(): return False
    try:
        logger.info(f"Updating synthesized cluster article ID: {cluster_article_id}")
        data_to_update = {
            "source_article_ids": source_article_ids, "headline": article_data.get("headline"),
            "summary": article_data.get("summary"), "content": article_data.get("content"),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        if not all(data_to_update[k] for k in ["headline", "summary", "content"]):
            logger.error(f"Missing data for updating cluster article {cluster_article_id}."); return False
        response = supabase.table("cluster_articles").update(data_to_update).eq("id", str(cluster_article_id)).execute()
        if getattr(response, 'error', None):
            logger.error(f"Error updating cluster article {cluster_article_id}: {response.error}"); return False
        logger.info(f"Successfully updated cluster article {cluster_article_id}.")
        return True
    except Exception as e:
        logger.error(f"Exception updating cluster article: {e}", exc_info=True); return False

async def mark_cluster_content_processed(cluster_id: Union[str, UUID]) -> bool:
    """
    Marks the content for a cluster as processed in the database.

    Args:
        cluster_id (Union[str, UUID]): The cluster ID to mark as processed.

    Returns:
        bool: True if the update was successful, False otherwise.
    """
    if not _check_supabase_client(): return False
    try:
        logger.info(f"Marking cluster {cluster_id} as content processed.")
        response = supabase.table("clusters").update({"isContent": True}).eq("cluster_id", str(cluster_id)).execute()
        if getattr(response, 'error', None):
            logger.error(f"Error marking cluster {cluster_id} processed: {response.error}"); return False
        logger.info(f"Successfully marked cluster {cluster_id} processed.")
        return True
    except Exception as e:
        logger.error(f"Exception marking cluster processed: {e}", exc_info=True); return False
    
async def update_cluster_article_images(cluster_article_id: Union[str, UUID], image_url: Optional[str] = None, image2_url: Optional[str] = None) -> bool:
    """
    Updates image URLs for a cluster article in the database.

    Args:
        cluster_article_id (Union[str, UUID]): The UUID of the cluster article to update.
        image_url (Optional[str]): The first image URL to update (if any).
        image2_url (Optional[str]): The second image URL to update (if any).

    Returns:
        bool: True if the update was successful, False otherwise.
    """
    if not _check_supabase_client(): return False
    if not cluster_article_id: logger.error("Missing cluster_article_id for image update."); return False
    data_to_update = {}
    if image_url is not None: data_to_update["image_url"] = image_url
    if image2_url is not None: data_to_update["image2_url"] = image2_url
    if not data_to_update: logger.info(f"No image URLs for cluster article {cluster_article_id}. Skipping update."); return True
    data_to_update["updated_at"] = datetime.now(timezone.utc).isoformat()
    try:
        logger.info(f"Updating images for cluster_article ID: {cluster_article_id}")
        response = supabase.table("cluster_articles").update(data_to_update).eq("id", str(cluster_article_id)).execute()
        if getattr(response, 'error', None):
            logger.error(f"Error updating images for cluster article {cluster_article_id}: {response.error}"); return False
        logger.info(f"Successfully updated images for cluster article {cluster_article_id}.")
        return True
    except Exception as e:
        logger.error(f"Exception updating images for cluster article: {e}", exc_info=True); return False
    
# --- START: New functions for Cluster Article Internationalization (cluster_article_int) ---

async def get_cluster_article_translation(cluster_article_id: Union[str, UUID], language_code: str) -> Optional[Dict]:
    """
    Fetches a translation for a cluster article in the specified language.

    Args:
        cluster_article_id (Union[str, UUID]): The UUID of the cluster article.
        language_code (str): The language code for the translation.

    Returns:
        Optional[Dict]: The translation record if found, else None.
    """
    if not _check_supabase_client(): return None
    try:
        logger.debug(f"Checking translation for cluster_article_id {cluster_article_id}, language {language_code}")
        response = supabase.table("cluster_article_int").select("headline, summary, content").eq("cluster_article_id", str(cluster_article_id)).eq("language_code", language_code).limit(1).execute()
        if getattr(response, 'error', None):
            logger.error(f"Error checking cluster article translation: {response.error}"); return None
        return response.data[0] if response.data else None
    except APIError as e:
        logger.error(f"APIError checking translation for {cluster_article_id} ({language_code}): {e.message}")
        return None
    except Exception as e:
        logger.error(f"General error checking translation: {e}", exc_info=True); return None

async def insert_cluster_article_translation(cluster_article_id: Union[str, UUID], language_code: str, translated_data: Dict) -> bool:
    """
    Inserts a translation for a cluster article into the database.

    Args:
        cluster_article_id (Union[str, UUID]): The UUID of the cluster article.
        language_code (str): The language code for the translation.
        translated_data (Dict): The translated article data.

    Returns:
        bool: True if the insertion was successful, False otherwise.
    """
    if not _check_supabase_client(): return False
    headline, summary, content = translated_data.get("translated_headline"), translated_data.get("translated_summary"), translated_data.get("translated_content")
    if not all([headline, summary, content]):
        logger.error(f"Missing translated components for {cluster_article_id} ({language_code})."); return False
    try:
        logger.info(f"Inserting translation for {cluster_article_id} ({language_code}).")
        data = {"cluster_article_id": str(cluster_article_id), "language_code": language_code, "headline": headline, "summary": summary, "content": content}
        response = supabase.table("cluster_article_int").insert(data).execute()
        if getattr(response, 'error', None):
            err_msg = str(response.error.message if hasattr(response.error, 'message') else response.error)
            if "duplicate key value violates unique constraint" in err_msg.lower():
                 logger.warning(f"Translation for {cluster_article_id} ({language_code}) already exists (constraint violation).")
                 # Consider returning True if "already exists" is not a failure for the caller's intent
                 return False 
            logger.error(f"Failed to insert translation for {cluster_article_id} ({language_code}): {err_msg}")
            return False
        return bool(response.data and response.data[0].get("id"))
    except APIError as e: # Catch PostgREST errors
        logger.error(f"APIError inserting translation for {cluster_article_id} ({language_code}): {e.message}")
        if "duplicate key value violates unique constraint" in e.message.lower():
            logger.warning(f"Translation for {cluster_article_id} ({language_code}) already exists (APIError).")
        return False
    except Exception as e: # Catch other errors
        logger.error(f"General error inserting translation: {e}", exc_info=True); return False

async def get_cluster_articles_needing_translation(language_code: str) -> List[Dict]:
    """
    Fetches cluster articles that are missing translations for the specified language.

    Args:
        language_code (str): The language code to check for missing translations.

    Returns:
        List[Dict]: List of cluster article records needing translation.
    """
    if not _check_supabase_client(): return []
    try:
        logger.info(f"Fetching cluster articles needing translation for language_code '{language_code}'.")
        # SQL Function needs to be created in Supabase:
        # CREATE OR REPLACE FUNCTION get_articles_without_translation(target_language_code TEXT)
        # RETURNS SETOF cluster_articles 
        # AS $$ BEGIN RETURN QUERY SELECT ca.* FROM cluster_articles ca LEFT JOIN cluster_article_int cai ON ca.id = cai.cluster_article_id AND cai.language_code = target_language_code WHERE cai.id IS NULL; END; $$ LANGUAGE plpgsql;
        
        response = supabase.rpc("get_articles_without_translation", {"target_language_code": language_code}).execute()
        # RPC execute() raises APIError on failure. If no error, data is in response.data
        return response.data if response.data else []
    except APIError as e:
        logger.error(f"APIError calling RPC get_articles_without_translation for {language_code}: {e.message} - Details: {getattr(e, 'details', '')}")
        return []
    except Exception as e:
        logger.error(f"Unexpected exception in get_cluster_articles_needing_translation for {language_code}: {e}", exc_info=True)
        return []

# --- END: New functions for Cluster Article Internationalization ---

async def fetch_all_cluster_ids() -> List[Union[str, UUID]]:
    """
    Fetches all cluster IDs from the database.

    Returns:
        List[Union[str, UUID]]: List of all cluster IDs.
    """
    if not _check_supabase_client() or supabase is None:
        logger.error("Supabase client not initialized. Cannot fetch cluster IDs.")
        return []
    try:
        # First, get all cherry-picked clusters
        response = supabase.table("clusters").select("cluster_id").eq("cherry_pick", True).execute()
        
        if not response.data:
            logger.info("No cherry-picked clusters found.")
            return []
            
        cherry_picked_clusters = [item['cluster_id'] for item in response.data]
        logger.info(f"Found {len(cherry_picked_clusters)} cherry-picked clusters.")
        
        # Then, get all cluster_ids that already have timelines
        # We need to extract the cluster_id from the JSONB timeline_data field
        existing_timelines_response = supabase.table("timelines").select("timeline_data").execute()
        
        existing_cluster_ids = set()
        if existing_timelines_response.data:
            for item in existing_timelines_response.data:
                timeline_data = item.get('timeline_data', {})
                if isinstance(timeline_data, dict) and 'cluster_id' in timeline_data:
                    existing_cluster_ids.add(timeline_data['cluster_id'])
        
        # Filter out clusters that already have timelines
        new_clusters = [cluster_id for cluster_id in cherry_picked_clusters 
                       if cluster_id not in existing_cluster_ids]
        
        logger.info(f"Found {len(new_clusters)} clusters without timelines.")
        return new_clusters
        
    except APIError as e:
        logger.error(f"APIError fetching cluster IDs: {e.message} - {e.details}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error fetching cluster IDs: {e}")
        return []

async def save_timeline_to_database(cluster_id: Union[str, UUID], timeline_entries: List[Dict]) -> bool:
    """
    Saves timeline entries for a cluster to the database.

    Args:
        cluster_id (Union[str, UUID]): The cluster ID for the timeline.
        timeline_entries (List[Dict]): List of timeline entry records to save.

    Returns:
        bool: True if the save was successful, False otherwise.
    """
    if not _check_supabase_client(): 
        return False
    
    try:
        # Generate a UUID for this timeline
        timeline_id = str(uuid4())
        
        # Group entries by date (only the date part, ignoring time)
        from collections import defaultdict
        from datetime import datetime
        
        grouped_by_date = defaultdict(list)
        
        for entry in timeline_entries:
            created_at_str = entry.get('created_at', '')
            if created_at_str:
                # Parse the datetime and extract just the date part
                try:
                    dt = datetime.fromisoformat(created_at_str.replace('Z', '+00:00'))
                    date_key = dt.date().isoformat()  # YYYY-MM-DD format
                    
                    # Create a copy of the entry without full_content for database storage
                    db_entry = {k: v for k, v in entry.items() if k != 'full_content'}
                    grouped_by_date[date_key].append(db_entry)
                except ValueError:
                    logger.warning(f"Could not parse date: {created_at_str}, skipping entry")
                    continue
        
        # Sort dates in descending order and create final timeline structure
        sorted_dates = sorted(grouped_by_date.keys(), reverse=True)
        
        timeline_data = []
        for date_key in sorted_dates:
            entries_for_date = grouped_by_date[date_key]
            
            if len(entries_for_date) == 1:
                # Single entry for this date
                timeline_data.append(entries_for_date[0])
            else:
                # Multiple entries for this date - group them
                grouped_entry = {
                    "date": date_key,
                    "articles": entries_for_date
                }
                timeline_data.append(grouped_entry)
        
        # Prepare data for insertion - include cluster_id within timeline_data JSONB
        timeline_with_metadata = {
            "cluster_id": str(cluster_id),
            "timeline": timeline_data,
            "generated_at": datetime.utcnow().isoformat() + "Z"
        }
        
        insert_data = {
            "id": timeline_id,
            "cluster_id": str(cluster_id),  # Add cluster_id as separate column
            "timeline_data": timeline_with_metadata
        }
        
        # Insert into database
        response = supabase.table("timelines").insert(insert_data).execute()
        
        error_info = getattr(response, 'error', None)
        if error_info:
            logger.error(f"Error saving timeline for cluster {cluster_id}: {error_info}")
            return False
            
        logger.info(f"Successfully saved timeline {timeline_id} for cluster {cluster_id} with {len(timeline_data)} date groups")
        return timeline_id  # Return the timeline_id so it can be used for translations
        
    except Exception as e:
        logger.error(f"Exception saving timeline for cluster {cluster_id}: {e}", exc_info=True)
        return False


async def save_translated_timeline(timeline_id: Union[str, UUID], language_code: str, translated_data: Dict, cluster_id: Union[str, UUID] = None) -> bool:
    """
    Saves a translated timeline to the database.

    Args:
        timeline_id (Union[str, UUID]): The timeline ID to save the translation for.
        language_code (str): The language code for the translation.
        translated_data (Dict): The translated timeline data.
        cluster_id (Union[str, UUID], optional): The cluster ID associated with the timeline.

    Returns:
        bool: True if the save was successful, False otherwise.
    """
    if not _check_supabase_client(): return False
    
    try:
        # If cluster_id is not provided, try to extract it from the original timeline
        if cluster_id is None:
            try:
                timeline_response = supabase.table("timelines") \
                    .select("cluster_id, timeline_data") \
                    .eq("id", str(timeline_id)) \
                    .execute()
                
                if timeline_response.data and len(timeline_response.data) > 0:
                    timeline_record = timeline_response.data[0]
                    # Try to get cluster_id from the separate column first
                    cluster_id = timeline_record.get('cluster_id')
                    
                    # If not found in column, try to extract from timeline_data JSONB
                    if not cluster_id and timeline_record.get('timeline_data'):
                        # Handle both formats - direct key or nested in the timeline_data
                        if isinstance(timeline_record['timeline_data'], dict):
                            cluster_id = timeline_record['timeline_data'].get('cluster_id')
                        
                if not cluster_id:
                    logger.warning(f"Could not determine cluster_id for timeline {timeline_id}")
            except Exception as e:
                logger.warning(f"Failed to extract cluster_id for timeline {timeline_id}: {e}")
        
        # Prepare insert data
        insert_data = {
            "timeline_id": str(timeline_id),
            "language_code": language_code,
            "timeline_data": translated_data
        }
        
        # Add cluster_id if available
        if cluster_id:
            insert_data["cluster_id"] = str(cluster_id)
        
        # Check if a translation already exists and update it if it does
        check_response = supabase.table("timelines_int") \
            .select("id") \
            .eq("timeline_id", str(timeline_id)) \
            .eq("language_code", language_code) \
            .execute()
            
        if check_response.data and len(check_response.data) > 0:
            # Update existing translation
            existing_id = check_response.data[0]["id"]
            logger.info(f"Updating existing {language_code} translation for timeline {timeline_id}")
            
            update_data = {"timeline_data": translated_data}
            if cluster_id:
                update_data["cluster_id"] = str(cluster_id)
            
            response = supabase.table("timelines_int") \
                .update(update_data) \
                .eq("id", existing_id) \
                .execute()
        else:
            # Insert new translation
            logger.info(f"Inserting new {language_code} translation for timeline {timeline_id}")
            response = supabase.table("timelines_int") \
                .insert(insert_data) \
                .execute()
        
        if getattr(response, 'error', None):
            logger.error(f"Error saving {language_code} translation for timeline {timeline_id}: {response.error}")
            return False
            
        logger.info(f"Successfully saved {language_code} translation for timeline {timeline_id}")
        return True
        
    except APIError as e:
        logger.error(f"APIError saving translated timeline: {e.message} - {e.details}")
        return False
    except Exception as e:
        logger.error(f"Exception saving translated timeline for {timeline_id}: {e}")
        return False

async def get_untranslated_timelines(language_code: str = 'de') -> List[Dict]:
    """
    Fetches timelines that are missing translations for the specified language.

    Args:
        language_code (str, optional): The language code to check for missing translations (default: 'de').

    Returns:
        List[Dict]: List of timeline records needing translation.
    """
    if not _check_supabase_client(): return []
    try:
        logger.info(f"Fetching timelines without {language_code} translations")
        
        # First get all timeline ids that already have translations for this language
        translated_response = supabase.table("timelines_int") \
            .select("timeline_id") \
            .eq("language_code", language_code) \
            .execute()
            
        if translated_response.data:
            # Get IDs of already translated timelines
            translated_ids = [item['timeline_id'] for item in translated_response.data]
            
            # Now get all timelines except those that already have translations
            if translated_ids:
                timelines_response = supabase.table("timelines") \
                    .select("id, timeline_data") \
                    .not_in("id", translated_ids) \
                    .execute()
            else:
                # If no translations exist yet, get all timelines
                timelines_response = supabase.table("timelines") \
                    .select("id, timeline_data") \
                    .execute()
        else:
            # If no translations exist yet, get all timelines
            timelines_response = supabase.table("timelines") \
                .select("id, timeline_data") \
                .execute()
                
        logger.info(f"Found {len(timelines_response.data)} timelines without {language_code} translations")
        return timelines_response.data if timelines_response.data else []
        
    except APIError as e:
        logger.error(f"APIError fetching untranslated timelines: {e.message} - {e.details}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error fetching untranslated timelines: {e}")
        return []

# --- START: Story Line View functions ---

async def save_story_line_view(
    cluster_id: Union[str, UUID],
    viewpoint_name: str,
    viewpoint_justification: str,
    deep_dive_article: Dict[str, str]
) -> bool:
    """
    Saves a story line view for a cluster to the database.

    Args:
        cluster_id (Union[str, UUID]): The cluster ID for the story line view.
        viewpoint_name (str): The name of the viewpoint.
        viewpoint_justification (str): The justification for the viewpoint.
        deep_dive_article (Dict[str, str]): The deep dive article data.

    Returns:
        bool: True if the save was successful, False otherwise.
    """
    if not _check_supabase_client(): 
        return False
    
    try:
        # Extract content from the deep dive article
        headline = deep_dive_article.get("headline", "")
        content = deep_dive_article.get("article", "")  # Main article content
        introduction = deep_dive_article.get("introduction", "")
        
        # Basic validation
        if not all([headline, content, viewpoint_name]):
            logger.error(f"Missing required data for story line view. headline: {bool(headline)}, content: {bool(content)}, viewpoint: {bool(viewpoint_name)}")
            return False
        
        # Prepare data for insertion
        insert_data = {
            "cluster_id": str(cluster_id),
            "view": viewpoint_name,
            "justification": viewpoint_justification,
            "headline": headline,
            "introduction": introduction,
            "content": content,
            "created_at": datetime.now(timezone.utc).isoformat()
        }
        
        logger.info(f"Saving story line view for cluster {cluster_id}, viewpoint: '{viewpoint_name}'")
        
        # Insert into database
        response = supabase.table("story_line_view").insert(insert_data).execute()
        
        if getattr(response, 'error', None):
            logger.error(f"Error saving story line view for cluster {cluster_id}: {response.error}")
            return False
            
        logger.info(f"Successfully saved story line view for cluster {cluster_id}, viewpoint: '{viewpoint_name}'")
        return True
        
    except APIError as e:
        logger.error(f"APIError saving story line view: {e.message} - {e.details}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error saving story line view: {e}", exc_info=True)
        return False


async def save_multiple_story_line_views(
    cluster_id: Union[str, UUID],
    viewpoints: List[Dict],
    deep_dive_articles: List[Dict[str, str]]
) -> Dict[str, int]:
    """
    Saves multiple story line views for a cluster to the database.

    Args:
        cluster_id (Union[str, UUID]): The cluster ID for the story line views.
        viewpoints (List[Dict]): List of viewpoint records.
        deep_dive_articles (List[Dict[str, str]]): List of deep dive article data.

    Returns:
        Dict[str, int]: Dictionary with counts of successful and failed saves.
    """
    if not _check_supabase_client():
        return {"total": 0, "saved": 0, "failed": 0}
    
    stats = {"total": len(viewpoints), "saved": 0, "failed": 0}
    
    if len(viewpoints) != len(deep_dive_articles):
        logger.error(f"Mismatch between viewpoints ({len(viewpoints)}) and articles ({len(deep_dive_articles)}) for cluster {cluster_id}")
        stats["failed"] = stats["total"]
        return stats
    
    logger.info(f"Saving {stats['total']} story line views for cluster {cluster_id}")
    
    for viewpoint, article in zip(viewpoints, deep_dive_articles):
        try:
            success = await save_story_line_view(
                cluster_id=cluster_id,
                viewpoint_name=viewpoint.get("name", ""),
                viewpoint_justification=viewpoint.get("justification", ""),
                deep_dive_article=article
            )
            
            if success:
                stats["saved"] += 1
            else:
                stats["failed"] += 1
                
        except Exception as e:
            logger.error(f"Exception saving individual story line view for viewpoint '{viewpoint.get('name')}': {e}")
            stats["failed"] += 1
    
    logger.info(f"Story line view batch save complete for cluster {cluster_id}. Stats: {stats}")
    return stats

# --- END: Story Line View functions ---

# --- START: Story Line View Internationalization functions ---

async def get_untranslated_story_line_views(language_code: str = 'de') -> List[Dict]:
    """
    Fetches story line views that are missing translations for the specified language.

    Args:
        language_code (str, optional): The language code to check for missing translations (default: 'de').

    Returns:
        List[Dict]: List of story line view records needing translation.
    """
    if not _check_supabase_client():
        return []
    
    try:
        # First get all translated story line view IDs for the specified language
        translated_response = supabase.table("story_line_view_int") \
            .select("story_line_view_id") \
            .eq("language_code", language_code) \
            .execute()
            
        translated_ids = [item['story_line_view_id'] for item in translated_response.data] if translated_response.data else []
        
        # Get story line views that are not in the translated list
        if translated_ids:
            response = supabase.table("story_line_view") \
                .select("id, cluster_id, view, justification, headline, introduction, content") \
                .not_.in_("id", translated_ids) \
                .execute()
        else:
            # If no translations exist yet, get all story line views
            response = supabase.table("story_line_view") \
                .select("id, cluster_id, view, justification, headline, introduction, content") \
                .execute()
                
        logger.info(f"Found {len(response.data)} story line views without {language_code} translations")
        return response.data if response.data else []
        
    except APIError as e:
        logger.error(f"APIError fetching untranslated story line views: {e.message} - {e.details}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error fetching untranslated story line views: {e}")
        return []

async def save_translated_story_line_view(
    story_line_view_id: Union[str, UUID],
    language_code: str,
    translated_data: Dict[str, str]
) -> bool:
    """
    Saves a translated story line view to the database.

    Args:
        story_line_view_id (Union[str, UUID]): The ID of the story line view to save the translation for.
        language_code (str): The language code for the translation.
        translated_data (Dict[str, str]): The translated story line view data.

    Returns:
        bool: True if the save was successful, False otherwise.
    """
    if not _check_supabase_client():
        return False
    
    try:
        # Prepare data for insertion
        insert_data = {
            "story_line_view_id": str(story_line_view_id),
            "language_code": language_code,
            "headline": translated_data.get("headline", ""),
            "content": translated_data.get("content", ""),
            "introduction": translated_data.get("introduction", ""),
            "view": translated_data.get("view", ""),
            "justification": translated_data.get("justification", ""),
            "created_at": datetime.now(timezone.utc).isoformat()
        }
        
        logger.info(f"Saving {language_code} translation for story line view {story_line_view_id}")
        
        # Insert into database
        response = supabase.table("story_line_view_int").insert(insert_data).execute()
        
        if getattr(response, 'error', None):
            logger.error(f"Error saving translated story line view: {response.error}")
            return False
            
        logger.info(f"Successfully saved {language_code} translation for story line view {story_line_view_id}")
        return True
        
    except APIError as e:
        logger.error(f"APIError saving translated story line view: {e.message} - {e.details}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error saving translated story line view: {e}", exc_info=True)
        return False

# --- END: Story Line View Internationalization functions ---

# --- BEGIN: Image Metadata Storage functions ---

async def save_cluster_image_metadata(cluster_id: int, image_url: str, original_url: str, author: str = None, source: str = None) -> int:
    """
    Saves metadata for a cluster image to the cluster_images table.
    
    Args:
        cluster_id (int): The ID of the cluster this image belongs to
        image_url (str): The URL of the uploaded image in Supabase storage
        original_url (str): The original URL where the image was found
        author (str, optional): The author/creator of the image
        source (str, optional): The source/website where the image was found
        
    Returns:
        int: The ID of the created record, or 0 if creation failed
    """
    if not _check_supabase_client():
        logger.warning("Cannot save cluster image metadata: Supabase client missing.")
        return 0
        
    try:
        logger.info(f"Saving metadata for cluster image. Cluster ID: {cluster_id}")
        
        # Create record with current UTC timestamp
        now = datetime.now(timezone.utc).isoformat()
        record = {
            "created_at": now,
            "cluster_id": cluster_id,
            "image_url": image_url,
            "original_url": original_url
        }
        
        # Add optional fields if provided
        if author:
            record["author"] = author
        if source:
            record["source"] = source
            
        # Insert the record
        response = supabase.table("cluster_images").insert(record).execute()
        
        # Extract and return the ID
        if response and response.data and len(response.data) > 0:
            image_id = response.data[0].get("id", 0)
            logger.info(f"Successfully saved cluster image metadata with ID: {image_id}")
            return image_id
        else:
            logger.warning("No data returned when saving cluster image metadata")
            return 0
            
    except Exception as e:
        logger.error(f"Error saving cluster image metadata: {e}", exc_info=True)
        return 0
        
async def save_article_image_metadata(article_id: int, image_url: str, original_url: str, author: str = None, source: str = None) -> int:
    """
    Saves metadata for an article image to the article_images table.
    
    Args:
        article_id (int): The ID of the article this image belongs to
        image_url (str): The URL of the uploaded image in Supabase storage
        original_url (str): The original URL where the image was found
        author (str, optional): The author/creator of the image
        source (str, optional): The source/website where the image was found
        
    Returns:
        int: The ID of the created record, or 0 if creation failed
    """
    if not _check_supabase_client():
        logger.warning("Cannot save article image metadata: Supabase client missing.")
        return 0
        
    try:
        logger.info(f"Saving metadata for article image. Article ID: {article_id}")
        
        # Create record with current UTC timestamp
        now = datetime.now(timezone.utc).isoformat()
        record = {
            "created_at": now,
            "article_id": article_id,
            "image_url": image_url,
            "original_url": original_url
        }
        
        # Add optional fields if provided
        if author:
            record["author"] = author
        if source:
            record["source"] = source
            
        # Insert the record
        response = supabase.table("article_images").insert(record).execute()
        
        # Extract and return the ID
        if response and response.data and len(response.data) > 0:
            image_id = response.data[0].get("id", 0)
            logger.info(f"Successfully saved article image metadata with ID: {image_id}")
            return image_id
        else:
            logger.warning("No data returned when saving article image metadata")
            return 0
            
    except Exception as e:
        logger.error(f"Error saving article image metadata: {e}", exc_info=True)
        return 0

# --- END: Image Metadata Storage functions ---

async def archive_old_articles() -> Dict:
    """
    Archives news articles that are older than 18 days by setting their status to 'ARCHIVED'.
    
    Returns:
        Dict: Statistics about the archival process (total, archived, errors).
    """
    if not _check_supabase_client(): 
        return {"total": 0, "archived": 0, "errors": 1, "message": "Supabase client not initialized"}
    
    stats = {"total": 0, "archived": 0, "errors": 0}
    page, page_size, total_count = 0, 1000, None
    archive_threshold = datetime.now(timezone.utc) - timedelta(days=18)
    archive_threshold_str = archive_threshold.isoformat()
    
    try:
        logger.info(f"Starting archival of news articles older than {archive_threshold_str}...")
        
        while True:
            logger.info(f"Fetching NewsArticles page {page + 1}...")
            current_count_method = 'exact' if page == 0 else None
            
            # Select articles that are older than the threshold and not already archived
            response = supabase.table("NewsArticles") \
                .select("id, created_at, status", count=current_count_method) \
                .lt("created_at", archive_threshold_str) \
                .not_.eq("status", "ARCHIVED") \
                .range(page * page_size, (page + 1) * page_size - 1) \
                .execute()
                
            if hasattr(response, 'error') and response.error:
                logger.error(f"Error fetching NewsArticles page {page + 1}: {response.error}")
                stats["errors"] += 1
                break
                
            if page == 0:
                total_count = getattr(response, 'count', 0) if hasattr(response, 'count') else len(response.data)
                if total_count is None: 
                    total_count = 0
                logger.info(f"Total articles to archive: {total_count}")
                stats["total"] = total_count
                
                if not response.data and total_count == 0:
                    logger.info("No articles found to archive.")
                    return stats
                    
            if not response.data:
                logger.info("No more articles found to archive.")
                break
                
            articles = response.data
            
            # Archive articles in batch
            try:
                article_ids = [article["id"] for article in articles]
                update_resp = supabase.table("NewsArticles") \
                    .update({"status": "ARCHIVED"}) \
                    .in_("id", article_ids) \
                    .execute()
                
                if getattr(update_resp, 'error', None):
                    logger.error(f"Error archiving articles {article_ids}: {update_resp.error}")
                    stats["errors"] += len(article_ids)
                else:
                    logger.info(f"Set {len(article_ids)} articles to ARCHIVED status.")
                    stats["archived"] += len(article_ids)
                    
            except Exception as e_upd:
                logger.error(f"Exception archiving articles {article_ids}: {e_upd}")
                stats["errors"] += len(article_ids)
                    
            page += 1
            
        logger.info(f"Article archival complete. Statistics: {stats}")
        return stats
        
    except Exception as e:
        logger.error(f"Error in archive_old_articles: {e}", exc_info=True)
        stats["errors"] += 1
        return stats