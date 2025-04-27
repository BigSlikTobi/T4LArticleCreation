import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, List, Set, Optional

try:
    import database
    import cluster_logic
    import cluster_content_generator
    from articleImage import ImageSearcher
except ImportError as e:
    logging.critical(f"Failed to import required modules: {e}. Check file locations and PYTHONPATH.")
    import sys, os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    import database
    import cluster_logic
    import cluster_content_generator
    from articleImage import ImageSearcher

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)

RECENT_ARTICLE_WINDOW_HOURS: int = 48
RECENT_STORY_WINDOW_DAYS: int = 7
SIMILARITY_THRESHOLD: float = 0.5
CLUSTER_PROCESSING_DELAY_SECONDS: float = 3.0


async def process_cluster(
    cluster_info: Dict,
    image_searcher: Optional[ImageSearcher],
    is_update: bool = False
) -> bool:
    """
    Processes a single cluster: fetches content, generates/updates story,
    finds images (only if new), and saves.
    """
    latest_cluster_id = cluster_info.get('latest_cluster_id')
    article_ids_set: Optional[Set[int]] = cluster_info.get('article_ids')
    existing_story_id = cluster_info.get('existing_story_id') # Needed for updates

    if not latest_cluster_id or not article_ids_set:
        logger.error(f"Invalid cluster_info received: Missing latest_cluster_id or article_ids. Info: {cluster_info}")
        return False
    if is_update and not existing_story_id:
        logger.error(f"Invalid cluster_info for UPDATE: Missing existing_story_id. Info: {cluster_info}")
        return False

    article_ids_list = sorted(list(article_ids_set))

    logger.info(f"--- Starting processing for Cluster (Ephemeral ID): {latest_cluster_id} ({len(article_ids_list)} articles) | Update: {is_update} | Target Story: {existing_story_id if is_update else 'N/A'} ---")

    # --- Variables to hold final story components ---
    final_headline_eng = ""
    final_headline_ger = ""
    final_summary_eng = ""
    final_summary_ger = ""
    final_body_eng = ""
    final_body_ger = ""
    final_image1_url = ""
    final_image2_url = ""
    final_image3_url = ""
    # ---

    try:
        # --- Fetch Existing Data ONLY if updating ---
        existing_data = None
        if is_update:
            logger.info(f"Step 0: Fetching existing story details for update (Story ID: {existing_story_id}).")
            existing_data = await database.fetch_cluster_story_details(existing_story_id)
            if not existing_data:
                logger.error(f"Cluster {latest_cluster_id}: Failed to fetch existing story {existing_story_id} details. Skipping update.")
                return False
            # Pre-populate final values with existing ones
            final_headline_eng = existing_data.get("headline_english", "")
            final_headline_ger = existing_data.get("headline_german", "")
            final_image1_url = existing_data.get("image1_url", "")
            final_image2_url = existing_data.get("image2_url", "")
            final_image3_url = existing_data.get("image3_url", "")
            logger.info(f"Existing headlines and images fetched for story {existing_story_id}.")
        # --- End Fetch Existing Data ---

        # 1. Fetch Source Content (Always Needed)
        logger.info(f"Step 1: Fetching source content for {len(article_ids_list)} articles.")
        content_map = await database.get_source_articles_content(article_ids_list)
        valid_source_contents = [content_map.get(aid, '') for aid in article_ids_list if content_map.get(aid)]
        if not valid_source_contents:
            logger.error(f"Cluster {latest_cluster_id}: No valid English content found. Skipping.")
            return False
        logger.info(f"Found non-empty content for {len(valid_source_contents)}/{len(article_ids_list)} articles.")

        # 2. Synthesize English Story Content (Summary & Body - Always Regenerate)
        #    Headline generation is skipped if updating
        logger.info(f"Step 2: Synthesizing English story content for cluster {latest_cluster_id}.")
        english_story_parts = await cluster_content_generator.synthesize_english_story(valid_source_contents)
        if not english_story_parts:
            logger.error(f"Cluster {latest_cluster_id}: Failed to synthesize English content. Skipping.")
            return False
        # Store the newly synthesized summary and body
        final_summary_eng = english_story_parts.get("summary", "")
        final_body_eng = english_story_parts.get("content", "")
        # Use generated headline ONLY IF creating new
        if not is_update:
            final_headline_eng = english_story_parts.get("headline", "")
        logger.info(f"English content synthesized. New Summary: {final_summary_eng[:60]}...")

        # 3. Translate Synthesized Content (Summary & Body - Always Regenerate)
        #    Headline translation is skipped if updating
        logger.info(f"Step 3: Translating synthesized content to German for cluster {latest_cluster_id}.")
        # Prepare data for translation (use final headline, even if it's the old one)
        data_to_translate = {
            "headline": final_headline_eng, # Pass headline for context, even if not saved
            "summary": final_summary_eng,
            "content": final_body_eng
        }
        german_story_parts = await cluster_content_generator.translate_synthesized_story(data_to_translate)
        if not german_story_parts:
            logger.error(f"Cluster {latest_cluster_id}: Failed to translate synthesized content. Skipping.")
            return False
        # Store the newly translated summary and body
        final_summary_ger = german_story_parts.get("summary", "")
        final_body_ger = german_story_parts.get("content", "")
        # Use translated headline ONLY IF creating new
        if not is_update:
            final_headline_ger = german_story_parts.get("headline", "")
        logger.info("Synthesized content translated to German.")

        # 4. Find Images (ONLY IF Creating New)
        if not is_update:
            if image_searcher:
                logger.info(f"Step 4: Searching for images for new cluster {latest_cluster_id}.")
                search_query_content = final_body_eng + " " + final_headline_eng # Use generated content/headline
                if not search_query_content.strip():
                    logger.warning(f"Cluster {latest_cluster_id}: English content/headline empty, skipping image search.")
                else:
                    images = await image_searcher.search_images(search_query_content, num_images=3, content=final_body_eng)
                    if not images:
                        logger.warning(f"Cluster {latest_cluster_id}: No images found.")
                    else:
                        logger.info(f"Found {len(images)} images for cluster {latest_cluster_id}.")
                        # Assign to final variables
                        final_image1_url = images[0].get('url', '') if len(images) > 0 else ""
                        final_image2_url = images[1].get('url', '') if len(images) > 1 else ""
                        final_image3_url = images[2].get('url', '') if len(images) > 2 else ""
            else:
                 logger.warning(f"Cluster {latest_cluster_id}: ImageSearcher not initialized, skipping image search.")
        else:
            logger.info(f"Step 4: Skipping image search (Update operation). Reusing existing images.")

        # 5. Prepare Data for Database
        logger.info(f"Step 5: Preparing final data for database for cluster {latest_cluster_id}.")
        current_time_iso = datetime.now(timezone.utc).isoformat()

        # Use the final variables determined above
        story_data = {
            "cluster_id": latest_cluster_id,
            "source_article_ids": article_ids_list,
            "headline_english": final_headline_eng,
            "headline_german": final_headline_ger,
            "summary_english": final_summary_eng,
            "summary_german": final_summary_ger,
            "body_english": final_body_eng,
            "body_german": final_body_ger,
            "image1_url": final_image1_url,
            "image2_url": final_image2_url,
            "image3_url": final_image3_url,
            "status": "PUBLISHED", # Or "UPDATED" / "NEW"
            "updated_at": current_time_iso
        }
        # Add created_at only if inserting
        if not is_update:
             story_data["created_at"] = current_time_iso
             # story_data['status'] = "NEW"

        # 6. Save to Database (Insert or Update)
        logger.info(f"Step 6: Saving cluster story {latest_cluster_id} to database (Update={is_update}).")
        save_successful = False
        if is_update:
            # story_data['status'] = "UPDATED" # Optional status change
            save_successful = await database.update_cluster_story(existing_story_id, story_data)
        else:
            new_story_id = await database.insert_cluster_story(story_data)
            save_successful = new_story_id is not None

        if save_successful:
            logger.info(f"Successfully {'updated' if is_update else 'inserted'} cluster story for cluster ID: {latest_cluster_id}")
            return True
        else:
            logger.error(f"Failed to {'update' if is_update else 'insert'} cluster story for cluster ID: {latest_cluster_id}")
            return False

    except Exception as e:
        logger.error(f"An unexpected error occurred while processing cluster {latest_cluster_id}: {e}", exc_info=True)
        return False
    finally:
        logger.info(f"--- Finished processing for Cluster (Ephemeral ID): {latest_cluster_id} ---")


# --- run_cluster_pipeline function remains the same ---
async def run_cluster_pipeline():
    """
    Main orchestration function for the cluster story pipeline.
    (Code identical to previous version)
    """
    logger.info("===== Starting Cluster Story Pipeline Run =====") # Use logger

    start_time = datetime.now(timezone.utc)
    processed_clusters_count = 0
    created_stories_count = 0
    updated_stories_count = 0
    failed_clusters_count = 0
    image_searcher: Optional[ImageSearcher] = None

    try:
        try:
            logger.info("Initializing Image Searcher...")
            image_searcher = ImageSearcher(use_llm=True)
            logger.info("Image Searcher initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize ImageSearcher: {e}. Image search will be skipped.", exc_info=True)
            image_searcher = None

        # 1. Identify Relevant Clusters
        logger.info(f"Fetching recent clusters (last {RECENT_ARTICLE_WINDOW_HOURS} hours)...")
        recent_clusters = await database.fetch_recent_clusters(
            time_window_hours=RECENT_ARTICLE_WINDOW_HOURS
        )
        if recent_clusters is None:
             logger.critical("Database function fetch_recent_clusters returned None. Aborting pipeline.")
             return
        if not recent_clusters:
            logger.info("No recent clusters found matching criteria. Pipeline run finished.")
            return
        logger.info(f"Found {len(recent_clusters)} recent clusters.")

        # 2. Fetch Recent Cluster Stories for Matching
        logger.info(f"Fetching recent cluster stories (last {RECENT_STORY_WINDOW_DAYS} days)...")
        existing_stories = await database.fetch_recent_cluster_stories(
            time_window_days=RECENT_STORY_WINDOW_DAYS
        )
        if existing_stories is None:
             logger.critical("Database function fetch_recent_cluster_stories returned None. Aborting pipeline.")
             return
        logger.info(f"Found {len(existing_stories)} existing stories for matching.")

        # 3. Match Clusters and Categorize
        logger.info("Matching and categorizing clusters...")
        clusters_to_create, clusters_to_update = cluster_logic.match_and_categorize_clusters(
            recent_clusters,
            existing_stories,
            similarity_threshold=SIMILARITY_THRESHOLD
        )
        total_to_process = len(clusters_to_create) + len(clusters_to_update)
        logger.info(f"Categorization result: {len(clusters_to_create)} NEW, {len(clusters_to_update)} UPDATE. Total to process: {total_to_process}")

        if total_to_process == 0:
             logger.info("No clusters require creation or update based on matching logic. Pipeline run finished.")
             return

        # 4. Process Clusters for Creation
        if clusters_to_create:
            logger.info(f"\n--- Processing {len(clusters_to_create)} clusters for CREATION ---")
            for i, cluster_info in enumerate(clusters_to_create):
                task_start_time = datetime.now(timezone.utc)
                log_cluster_id = cluster_info.get('latest_cluster_id', 'UNKNOWN')
                logger.info(f"Processing NEW cluster {i+1}/{len(clusters_to_create)} (Ephemeral ID: {log_cluster_id})...")
                processed_clusters_count += 1
                success = await process_cluster(cluster_info, image_searcher, is_update=False)
                if success:
                    created_stories_count += 1
                else:
                    failed_clusters_count += 1
                task_duration = datetime.now(timezone.utc) - task_start_time
                logger.info(f"Finished processing NEW cluster {i+1}. Success: {success}. Duration: {task_duration}")
                if i < len(clusters_to_create) - 1 or clusters_to_update:
                    logger.info(f"Waiting {CLUSTER_PROCESSING_DELAY_SECONDS}s before next cluster...")
                    await asyncio.sleep(CLUSTER_PROCESSING_DELAY_SECONDS)

        # 5. Process Clusters for Update
        if clusters_to_update:
            logger.info(f"\n--- Processing {len(clusters_to_update)} clusters for UPDATE ---")
            for i, cluster_info in enumerate(clusters_to_update):
                task_start_time = datetime.now(timezone.utc)
                log_cluster_id = cluster_info.get('latest_cluster_id', 'UNKNOWN')
                log_existing_id = cluster_info.get('existing_story_id', 'UNKNOWN')
                logger.info(f"Processing UPDATE cluster {i+1}/{len(clusters_to_update)} (Ephemeral ID: {log_cluster_id}, Target Story: {log_existing_id})...")
                processed_clusters_count += 1
                success = await process_cluster(cluster_info, image_searcher, is_update=True)
                if success:
                    updated_stories_count += 1
                else:
                    failed_clusters_count += 1
                task_duration = datetime.now(timezone.utc) - task_start_time
                logger.info(f"Finished processing UPDATE cluster {i+1}. Success: {success}. Duration: {task_duration}")
                if i < len(clusters_to_update) - 1:
                    logger.info(f"Waiting {CLUSTER_PROCESSING_DELAY_SECONDS}s before next cluster...")
                    await asyncio.sleep(CLUSTER_PROCESSING_DELAY_SECONDS)

    except asyncio.CancelledError:
         logger.info("Pipeline run cancelled.")
    except Exception as e:
        logger.critical(f"A critical error occurred during the pipeline execution: {e}", exc_info=True)
    finally:
        end_time = datetime.now(timezone.utc)
        duration = end_time - start_time
        logger.info("\n===== Cluster Story Pipeline Run Summary =====")
        logger.info(f"Start Time: {start_time.isoformat()}")
        logger.info(f"End Time:   {end_time.isoformat()}")
        logger.info(f"Duration: {duration}")
        logger.info(f"Clusters Processed Attempted: {processed_clusters_count}")
        logger.info(f"  New Stories Created:        {created_stories_count}")
        logger.info(f"  Existing Stories Updated:   {updated_stories_count}")
        logger.info(f"  Clusters Failed:            {failed_clusters_count}")
        logger.info("============================================")


if __name__ == "__main__":
    logger = logging.getLogger() # Get root logger
    try:
        asyncio.run(run_cluster_pipeline())
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user.")
    except Exception as e:
         logger.critical(f"Pipeline failed to run due to an unhandled error in __main__: {e}", exc_info=True)
