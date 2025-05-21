import asyncio
import logging
from typing import List, Dict, Optional
from uuid import UUID # For type hinting cluster_article_id

import database
import cluster_article_generator
from articleImage import ImageSearcher # Import ImageSearcher

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(module)s - %(message)s',
    force=True
)

async def process_single_cluster(
    cluster_id: str,
    cluster_status: str,
    image_searcher: ImageSearcher, # Pass ImageSearcher instance
    language: str = "English"
):
    """
    Processes a single cluster: synthesizes content, finds images, and updates database.
    """
    logger.info(f"Processing cluster {cluster_id} with status '{cluster_status}'.")

    source_articles = await database.fetch_source_articles_for_cluster(cluster_id)
    if not source_articles:
        logger.warning(f"No source articles found for cluster {cluster_id}. Skipping.")
        return

    logger.info(f"Found {len(source_articles)} source articles for cluster {cluster_id}.")
    source_article_db_ids = [sa['id'] for sa in source_articles]

    synthesized_data: Optional[Dict] = None
    # This will store the ID of the record in 'cluster_articles' table
    processed_cluster_article_id: Optional[UUID] = None 

    if cluster_status == 'NEW':
        logger.info(f"Generating NEW synthesized article for cluster {cluster_id}.")
        synthesized_data = await cluster_article_generator.generate_cluster_article(
            source_articles_data=source_articles,
            language=language
        )
        if synthesized_data and synthesized_data.get('headline') and synthesized_data.get('content'):
            new_ca_id = await database.insert_cluster_article(
                cluster_id=cluster_id,
                source_article_ids=source_article_db_ids,
                article_data=synthesized_data
            )
            if new_ca_id:
                logger.info(f"Successfully inserted new synthesized article (ID: {new_ca_id}) for cluster {cluster_id}.")
                processed_cluster_article_id = new_ca_id
                # Mark content processed for text part
                # await database.mark_cluster_content_processed(cluster_id) # Moved further down
            else:
                logger.error(f"Failed to insert new synthesized article for cluster {cluster_id}.")
                return # Stop if text content fails
        else:
            logger.error(f"Failed to generate or validate synthesized data for NEW cluster {cluster_id}. Response: {synthesized_data}")
            return # Stop if text content fails

    elif cluster_status == 'UPDATED':
        logger.info(f"Generating UPDATED synthesized article for cluster {cluster_id}.")
        existing_synthesized_article_db_data = await database.get_existing_cluster_article(cluster_id)

        if not existing_synthesized_article_db_data:
            logger.warning(f"Cluster {cluster_id} is 'UPDATED' but no existing synthesized article found. Treating as NEW.")
            synthesized_data = await cluster_article_generator.generate_cluster_article(
                source_articles_data=source_articles,
                language=language
            )
            if synthesized_data and synthesized_data.get('headline') and synthesized_data.get('content'):
                new_ca_id = await database.insert_cluster_article(
                    cluster_id=cluster_id,
                    source_article_ids=source_article_db_ids,
                    article_data=synthesized_data
                )
                if new_ca_id:
                    logger.info(f"Successfully inserted (fallback) new synthesized article (ID: {new_ca_id}) for 'UPDATED' cluster {cluster_id}.")
                    processed_cluster_article_id = new_ca_id
                    # await database.mark_cluster_content_processed(cluster_id) # Moved
                else:
                    logger.error(f"Failed to insert (fallback) new synthesized article for 'UPDATED' cluster {cluster_id}.")
                    return # Stop if text content fails
            else:
                logger.error(f"Failed to generate or validate (fallback) synthesized data for UPDATED cluster {cluster_id}. Response: {synthesized_data}")
                return # Stop if text content fails
        else: # Existing article found, proceed with update
            logger.info(f"Found existing synthesized article (ID: {existing_synthesized_article_db_data['id']}) for cluster {cluster_id} to update.")
            processed_cluster_article_id = UUID(str(existing_synthesized_article_db_data['id'])) # Store existing ID
            
            synthesized_data = await cluster_article_generator.generate_cluster_article(
                source_articles_data=source_articles,
                previous_combined_article_data=existing_synthesized_article_db_data,
                language=language
            )
            if synthesized_data and synthesized_data.get('headline') and synthesized_data.get('content'):
                success = await database.update_cluster_article(
                    cluster_article_id=processed_cluster_article_id,
                    source_article_ids=source_article_db_ids,
                    article_data=synthesized_data
                )
                if success:
                    logger.info(f"Successfully updated synthesized text content for cluster article {processed_cluster_article_id}.")
                    # await database.mark_cluster_content_processed(cluster_id) # Moved
                else:
                    logger.error(f"Failed to update synthesized text content for cluster article {processed_cluster_article_id}.")
                    return # Stop if text content update fails
            else:
                logger.error(f"Failed to generate or validate synthesized data for UPDATED cluster {cluster_id}. Response: {synthesized_data}")
                return # Stop if text content fails
    else:
        logger.warning(f"Unknown cluster status '{cluster_status}' for cluster {cluster_id}. Skipping.")
        return

    # --- Image Processing Step (applies if text content was successful) ---
    if processed_cluster_article_id and synthesized_data:
        logger.info(f"Attempting to find images for cluster article {processed_cluster_article_id} (Cluster: {cluster_id}).")
        
        # Create search query from synthesized content
        # Use headline and a part of the content for better image search context
        search_query_text = (
            (synthesized_data.get('headline', "") or "").replace("<h1>", "").replace("</h1>", "") + " " +
            (synthesized_data.get('summary', "") or "").replace("<p>", "").replace("</p>", "")
        ).strip()
        
        # If summary is short, add some main content
        if len(search_query_text.split()) < 10 :
             main_content_snippet = (synthesized_data.get('content', "") or "")
             # Basic HTML tag removal for snippet
             main_content_snippet = main_content_snippet.replace("<div>", "").replace("</div>", "")
             main_content_snippet = main_content_snippet.replace("<h2>", "").replace("</h2>", "")
             main_content_snippet = main_content_snippet.replace("<h3>", "").replace("</h3>", "")
             main_content_snippet = main_content_snippet.replace("<p>", "").replace("</p>", "")
             search_query_text += " " + " ".join(main_content_snippet.split()[:50]) # Add first 50 words of content
             search_query_text = search_query_text.strip()


        if not search_query_text:
            logger.warning(f"Search query for images is empty for cluster article {processed_cluster_article_id}. Skipping image search.")
        else:
            logger.info(f"Image search query: '{search_query_text[:200]}...'")
            try:
                # Fetch 2 images. Pass synthesized content for better ranking context.
                images = await image_searcher.search_images(
                    query=search_query_text, 
                    num_images=2,
                    content=synthesized_data.get('content', '') # Pass full content for ranking
                )
                
                image1_url: Optional[str] = None
                image2_url: Optional[str] = None

                if images:
                    logger.info(f"Found {len(images)} images for cluster article {processed_cluster_article_id}.")
                    image1_url = images[0].get('url') if len(images) > 0 else None
                    image2_url = images[1].get('url') if len(images) > 1 else None
                    
                    if image1_url or image2_url: # Only update if at least one image is found
                        img_update_success = await database.update_cluster_article_images(
                            cluster_article_id=processed_cluster_article_id,
                            image_url=image1_url,
                            image2_url=image2_url
                        )
                        if img_update_success:
                            logger.info(f"Successfully updated images for cluster article {processed_cluster_article_id}.")
                        else:
                            logger.error(f"Failed to update images for cluster article {processed_cluster_article_id}.")
                    else:
                        logger.info(f"No valid image URLs to update for cluster article {processed_cluster_article_id}.")
                else:
                    logger.warning(f"No images found for cluster article {processed_cluster_article_id} using query: '{search_query_text[:100]}...'")
            except Exception as e_img:
                logger.error(f"Error during image search/processing for cluster article {processed_cluster_article_id}: {e_img}", exc_info=True)
        
        # Now that text and (attempted) image processing are done, mark the cluster as content processed.
        # This means even if images fail, as long as text is there, isContent becomes true.
        # If images are a HARD requirement, this call should be conditional on image success.
        logger.info(f"Finalizing processing for cluster {cluster_id}. Marking content as processed.")
        await database.mark_cluster_content_processed(cluster_id)

    elif not processed_cluster_article_id:
        logger.error(f"Skipping image processing and finalization for cluster {cluster_id} due to earlier text processing failure.")


async def run_cluster_processing_pipeline():
    logger.info("Starting Cluster Processing Pipeline...")

    # Initialize ImageSearcher once
    try:
        image_searcher = ImageSearcher(use_llm=True) # Or False based on preference/quota
        logger.info("ImageSearcher initialized successfully.")
    except Exception as e:
        logger.critical(f"Failed to initialize ImageSearcher: {e}. Image processing will be skipped.", exc_info=True)
        # Decide if pipeline should halt or continue without images
        # For now, let's allow it to continue for text processing if image_searcher fails
        image_searcher = None # type: ignore 

    # Process 'NEW' clusters
    logger.info("--- Processing NEW clusters ---")
    new_clusters = await database.fetch_clusters_to_process('NEW')
    if not new_clusters:
        logger.info("No 'NEW' clusters to process.")
    else:
        logger.info(f"Found {len(new_clusters)} 'NEW' clusters to process.")
        for i, cluster_data in enumerate(new_clusters):
            cluster_id = cluster_data.get('cluster_id')
            if cluster_id:
                logger.info(f"Processing NEW cluster {i+1}/{len(new_clusters)}: {cluster_id}")
                if image_searcher: # Only proceed if image_searcher is available
                    await process_single_cluster(cluster_id, 'NEW', image_searcher)
                else:
                    logger.warning(f"Skipping image search for cluster {cluster_id} as ImageSearcher failed to initialize.")
                    # Call a version of process_single_cluster that doesn't do images or handle it inside
                    # For simplicity, current process_single_cluster will log warning if image_searcher is None (not passed)
                    # but we are passing it, so it needs to be handled.
                    # Let's assume if image_searcher is None, we still do text processing.
                    # The `process_single_cluster` needs to be robust to a None image_searcher if we proceed this way.
                    # However, for now, the logic passes it, and if it's None, the image block inside process_single_cluster will effectively be skipped.
                    # The above instantiation of ImageSearcher should be passed.
                    await process_single_cluster(cluster_id, 'NEW', image_searcher) # image_searcher could be None here

                if i < len(new_clusters) - 1:
                     logger.info("Waiting 5 seconds before next cluster...")
                     await asyncio.sleep(5) 
            else:
                logger.warning(f"Found cluster data without 'cluster_id': {cluster_data}")

    # Process 'UPDATED' clusters
    logger.info("--- Processing UPDATED clusters ---")
    updated_clusters = await database.fetch_clusters_to_process('UPDATED')
    if not updated_clusters:
        logger.info("No 'UPDATED' clusters to process.")
    else:
        logger.info(f"Found {len(updated_clusters)} 'UPDATED' clusters to process.")
        for i, cluster_data in enumerate(updated_clusters):
            cluster_id = cluster_data.get('cluster_id')
            if cluster_id:
                logger.info(f"Processing UPDATED cluster {i+1}/{len(updated_clusters)}: {cluster_id}")
                if image_searcher:
                     await process_single_cluster(cluster_id, 'UPDATED', image_searcher)
                else:
                    logger.warning(f"Skipping image search for cluster {cluster_id} as ImageSearcher failed to initialize.")
                    await process_single_cluster(cluster_id, 'UPDATED', image_searcher) # image_searcher could be None

                if i < len(updated_clusters) - 1:
                     logger.info("Waiting 5 seconds before next cluster...")
                     await asyncio.sleep(5)
            else:
                logger.warning(f"Found cluster data without 'cluster_id': {cluster_data}")

    logger.info("Cluster Processing Pipeline finished.")

if __name__ == "__main__":
    db_ready = database.supabase is not None
    llm_ready = (
        cluster_article_generator.gemini_model is not None and
        "Error:" not in cluster_article_generator.multi_source_synthesis_prompt_template
    )
    # ImageSearcher readiness could also be checked if it's critical for startup
    # image_searcher_ready = ... (check if API keys are present for it)

    if db_ready and llm_ready: # Add image_searcher_ready if critical
        asyncio.run(run_cluster_processing_pipeline())
    else:
        if not db_ready: logger.critical("Database client not initialized. Halting pipeline.")
        if not llm_ready: logger.critical("LLM model or prompts not initialized correctly. Halting pipeline.")
        logger.critical("Cluster pipeline cannot start due to initialization errors.")