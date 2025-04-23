import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, List, Set, Optional # Added Optional

# --- Project Imports ---
# Assuming these files are in the same directory or configured in PYTHONPATH
try:
    import database
    import cluster_logic
    import cluster_content_generator
    from articleImage import ImageSearcher
except ImportError as e:
    logging.critical(f"Failed to import required modules: {e}. Check file locations and PYTHONPATH.")
    # Depending on setup, you might need to add parent dir to sys.path
    import sys, os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    import database
    import cluster_logic
    import cluster_content_generator
    from articleImage import ImageSearcher


# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)

# --- Configuration ---
# Time windows for fetching data
RECENT_ARTICLE_WINDOW_HOURS: int = 48 # How far back to look for SourceArticles with clusters
RECENT_STORY_WINDOW_DAYS: int = 7      # How far back to look for existing ClusterStories to compare against

# Logic thresholds
SIMILARITY_THRESHOLD: float = 0.5 # Jaccard threshold for considering a cluster an UPDATE candidate

# Processing controls
CLUSTER_PROCESSING_DELAY_SECONDS: float = 3.0 # Delay between processing each cluster to avoid rate limits


async def process_cluster(
    cluster_info: Dict,
    image_searcher: Optional[ImageSearcher], # Allow None if initialization fails
    is_update: bool = False
) -> bool:
    """
    Processes a single cluster: fetches content, generates story, finds images, and saves.

    Args:
        cluster_info: Dict containing cluster details
                      (e.g., {'cluster_id': str, 'article_ids': Set[int]}) or
                      (e.g., {'cluster_id': str, 'article_ids': Set[int], 'existing_story_id': str} for updates)
        image_searcher: An initialized ImageSearcher instance, or None.
        is_update: Boolean flag indicating if this is an update operation.

    Returns:
        True if processing and saving were successful, False otherwise.
    """
    cluster_id = cluster_info.get('cluster_id')
    article_ids_set: Optional[Set[int]] = cluster_info.get('article_ids')

    if not cluster_id or not article_ids_set:
        logging.error(f"Invalid cluster_info received: Missing cluster_id or article_ids. Info: {cluster_info}")
        return False

    article_ids_list = sorted(list(article_ids_set)) # Use sorted list for consistency

    logging.info(f"--- Starting processing for Cluster ID: {cluster_id} ({len(article_ids_list)} articles) | Update: {is_update} ---")

    try:
        # 1. Fetch Source Content
        logging.info(f"Step 1: Fetching source content for {len(article_ids_list)} articles.")
        content_map = await database.get_source_articles_content(article_ids_list)
        # Ensure all requested IDs are present in the map (with empty string if not found)
        source_contents_english = [content_map.get(aid, '') for aid in article_ids_list]
        # Filter out only those that are actually empty after fetch attempt
        valid_source_contents = [content for content in source_contents_english if content]

        if not valid_source_contents:
            logging.error(f"Cluster {cluster_id}: No valid English content found for any source articles after fetch. Skipping.")
            return False
        logging.info(f"Found non-empty content for {len(valid_source_contents)}/{len(article_ids_list)} articles.")

        # 2. Synthesize English Story
        logging.info(f"Step 2: Synthesizing English story for cluster {cluster_id}.")
        english_story = await cluster_content_generator.synthesize_english_story(valid_source_contents)
        if not english_story:
            logging.error(f"Cluster {cluster_id}: Failed to synthesize English story. Skipping.")
            return False
        logging.info(f"English story synthesized. Headline: {english_story.get('headline', '')[:60]}...")

        # 3. Translate Synthesized Story to German
        logging.info(f"Step 3: Translating synthesized story to German for cluster {cluster_id}.")
        german_story = await cluster_content_generator.translate_synthesized_story(english_story)
        if not german_story:
            logging.error(f"Cluster {cluster_id}: Failed to translate synthesized story to German. Skipping.")
            # Decide if you want to proceed without German translation or fail completely
            return False # Fail for now if translation fails
        logging.info("Synthesized story translated to German.")

        # 4. Find Images
        image1_url, image2_url, image3_url = "", "", "" # Initialize defaults
        if image_searcher:
            logging.info(f"Step 4: Searching for images for cluster {cluster_id}.")
            # Use synthesized content for richer search context
            search_query_content = english_story.get('content', '') + " " + english_story.get('headline', '')
            if not search_query_content.strip():
                logging.warning(f"Cluster {cluster_id}: English content/headline is empty, cannot search for images effectively.")
            else:
                images = await image_searcher.search_images(search_query_content, num_images=3, content=english_story.get('content', ''))
                if not images:
                    logging.warning(f"Cluster {cluster_id}: No images found by ImageSearcher.")
                else:
                    logging.info(f"Found {len(images)} images for cluster {cluster_id}.")
                    image1_url = images[0].get('url', '') if len(images) > 0 else ""
                    image2_url = images[1].get('url', '') if len(images) > 1 else ""
                    image3_url = images[2].get('url', '') if len(images) > 2 else ""
        else:
             logging.warning(f"Cluster {cluster_id}: ImageSearcher not initialized, skipping image search.")

        # 5. Prepare Data for Database
        logging.info(f"Step 5: Preparing data for database for cluster {cluster_id}.")
        current_time_iso = datetime.now(timezone.utc).isoformat()

        # Ensure required fields from LLM output exist, default to empty string if not
        headline_eng = english_story.get("headline", "")
        summary_eng = english_story.get("summary", "")
        body_eng = english_story.get("content", "")
        headline_ger = german_story.get("headline", "")
        summary_ger = german_story.get("summary", "")
        body_ger = german_story.get("content", "")

        story_data = {
            "cluster_id": cluster_id,
            "source_article_ids": article_ids_list, # Save as sorted list
            "headline_english": headline_eng,
            "headline_german": headline_ger,
            "summary_english": summary_eng,
            "summary_german": summary_ger,
            "body_english": body_eng,
            "body_german": body_ger,
            "image1_url": image1_url,
            "image2_url": image2_url,
            "image3_url": image3_url,
            "status": "PUBLISHED", # Or "NEW" / "UPDATED" based on is_update? Needs defining.
            "updated_at": current_time_iso # Always update this timestamp
        }

        # 6. Save to Database (Insert or Update)
        logging.info(f"Step 6: Saving cluster story {cluster_id} to database (Update={is_update}).")
        save_successful = False
        if is_update:
            existing_story_id = cluster_info.get('existing_story_id')
            if not existing_story_id:
                 logging.error(f"Cluster {cluster_id}: Marked as UPDATE but missing existing_story_id. Cannot update.")
                 return False
            # Set status to UPDATED or keep as PUBLISHED?
            # story_data['status'] = "UPDATED"
            save_successful = await database.update_cluster_story(existing_story_id, story_data)
        else:
            story_data["created_at"] = current_time_iso # Add created_at for new stories
            # story_data['status'] = "NEW" # Set initial status for new stories
            new_story_id = await database.insert_cluster_story(story_data)
            save_successful = new_story_id is not None

        if save_successful:
            logging.info(f"Successfully {'updated' if is_update else 'inserted'} cluster story for cluster ID: {cluster_id}")
            return True
        else:
            logging.error(f"Failed to {'update' if is_update else 'insert'} cluster story for cluster ID: {cluster_id}")
            return False

    except Exception as e:
        logging.error(f"An unexpected error occurred while processing cluster {cluster_id}: {e}", exc_info=True)
        return False
    finally:
        logging.info(f"--- Finished processing for Cluster ID: {cluster_id} ---")


async def run_cluster_pipeline():
    """
    Main orchestration function for the cluster story pipeline.
    """
    logging.info("===== Starting Cluster Story Pipeline Run =====")

    start_time = datetime.now(timezone.utc)
    processed_clusters_count = 0
    created_stories_count = 0
    updated_stories_count = 0
    failed_clusters_count = 0
    image_searcher: Optional[ImageSearcher] = None # Initialize as None

    try:
        # Initialize Image Searcher (handle potential failure)
        try:
            logging.info("Initializing Image Searcher...")
            image_searcher = ImageSearcher(use_llm=True) # Use LLM for image query optimization
            logging.info("Image Searcher initialized.")
        except Exception as e:
            logging.error(f"Failed to initialize ImageSearcher: {e}. Image search will be skipped.", exc_info=True)
            image_searcher = None # Ensure it's None if init fails

        # 1. Identify Relevant Clusters
        logging.info(f"Fetching recent clusters (last {RECENT_ARTICLE_WINDOW_HOURS} hours)...")
        recent_clusters = await database.fetch_recent_clusters(
            time_window_hours=RECENT_ARTICLE_WINDOW_HOURS
        )
        if recent_clusters is None: # Check explicitly for None in case of critical DB error
             logging.critical("Database function fetch_recent_clusters returned None. Aborting pipeline.")
             return
        if not recent_clusters:
            logging.info("No recent clusters found matching criteria. Pipeline run finished.")
            return
        logging.info(f"Found {len(recent_clusters)} recent clusters.")

        # 2. Fetch Recent Cluster Stories for Matching
        logging.info(f"Fetching recent cluster stories (last {RECENT_STORY_WINDOW_DAYS} days)...")
        existing_stories = await database.fetch_recent_cluster_stories(
            time_window_days=RECENT_STORY_WINDOW_DAYS
        )
        if existing_stories is None: # Check explicitly for None
             logging.critical("Database function fetch_recent_cluster_stories returned None. Aborting pipeline.")
             return
        logging.info(f"Found {len(existing_stories)} existing stories for matching.")

        # 3. Match Clusters and Categorize
        logging.info("Matching and categorizing clusters...")
        clusters_to_create, clusters_to_update = cluster_logic.match_and_categorize_clusters(
            recent_clusters,
            existing_stories,
            similarity_threshold=SIMILARITY_THRESHOLD
        )
        total_to_process = len(clusters_to_create) + len(clusters_to_update)
        logging.info(f"Categorization result: {len(clusters_to_create)} NEW, {len(clusters_to_update)} UPDATE. Total to process: {total_to_process}")

        if total_to_process == 0:
             logging.info("No clusters require creation or update based on matching logic. Pipeline run finished.")
             return

        # 4. Process Clusters for Creation
        if clusters_to_create:
            logging.info(f"\n--- Processing {len(clusters_to_create)} clusters for CREATION ---")
            for i, cluster_info in enumerate(clusters_to_create):
                task_start_time = datetime.now(timezone.utc)
                logging.info(f"Processing NEW cluster {i+1}/{len(clusters_to_create)} (ID: {cluster_info.get('cluster_id')})...")
                processed_clusters_count += 1
                success = await process_cluster(cluster_info, image_searcher, is_update=False)
                if success:
                    created_stories_count += 1
                else:
                    failed_clusters_count += 1
                task_duration = datetime.now(timezone.utc) - task_start_time
                logging.info(f"Finished processing NEW cluster {i+1}. Success: {success}. Duration: {task_duration}")
                # Optional delay
                if i < len(clusters_to_create) - 1 or clusters_to_update: # Add delay if more clusters follow
                    logging.info(f"Waiting {CLUSTER_PROCESSING_DELAY_SECONDS}s before next cluster...")
                    await asyncio.sleep(CLUSTER_PROCESSING_DELAY_SECONDS)

        # 5. Process Clusters for Update
        if clusters_to_update:
            logging.info(f"\n--- Processing {len(clusters_to_update)} clusters for UPDATE ---")
            for i, cluster_info in enumerate(clusters_to_update):
                task_start_time = datetime.now(timezone.utc)
                logging.info(f"Processing UPDATE cluster {i+1}/{len(clusters_to_update)} (ID: {cluster_info.get('cluster_id')})...")
                processed_clusters_count += 1
                success = await process_cluster(cluster_info, image_searcher, is_update=True)
                if success:
                    updated_stories_count += 1
                else:
                    failed_clusters_count += 1
                task_duration = datetime.now(timezone.utc) - task_start_time
                logging.info(f"Finished processing UPDATE cluster {i+1}. Success: {success}. Duration: {task_duration}")
                # Optional delay
                if i < len(clusters_to_update) - 1: # Add delay if more update clusters follow
                    logging.info(f"Waiting {CLUSTER_PROCESSING_DELAY_SECONDS}s before next cluster...")
                    await asyncio.sleep(CLUSTER_PROCESSING_DELAY_SECONDS)

    except asyncio.CancelledError:
         logging.info("Pipeline run cancelled.")
    except Exception as e:
        logging.critical(f"A critical error occurred during the pipeline execution: {e}", exc_info=True)
    finally:
        end_time = datetime.now(timezone.utc)
        duration = end_time - start_time
        logging.info("\n===== Cluster Story Pipeline Run Summary =====")
        logging.info(f"Start Time: {start_time.isoformat()}")
        logging.info(f"End Time:   {end_time.isoformat()}")
        logging.info(f"Duration: {duration}")
        logging.info(f"Clusters Processed Attempted: {processed_clusters_count}")
        logging.info(f"  New Stories Created:        {created_stories_count}")
        logging.info(f"  Existing Stories Updated:   {updated_stories_count}")
        logging.info(f"  Clusters Failed:            {failed_clusters_count}")
        logging.info("============================================")

if __name__ == "__main__":
    # Ensure the event loop is managed correctly
    try:
        # Python 3.7+
        asyncio.run(run_cluster_pipeline())
    except KeyboardInterrupt:
        logging.info("Pipeline interrupted by user.")
    except Exception as e:
         logging.critical(f"Pipeline failed to run due to an unhandled error in __main__: {e}", exc_info=True)