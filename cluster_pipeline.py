import asyncio
import logging
from typing import List, Dict, Optional
from uuid import UUID 

import database
import cluster_article_generator
from articleImage import ImageSearcher
import cluster_translator # Import the new translator module

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(module)s - %(message)s',
    force=True
)

# Define target languages for the pipeline
TARGET_TRANSLATION_LANGUAGES = ["de"] # Example: German. Add more codes like "es", "fr"

async def process_single_cluster(
    cluster_id: str,
    cluster_status: str,
    image_searcher: ImageSearcher,
    language: str = "English" # Primary synthesis language
):
    logger.info(f"Processing cluster {cluster_id} with status '{cluster_status}'.")

    source_articles = await database.fetch_source_articles_for_cluster(cluster_id)
    if not source_articles:
        logger.warning(f"No source articles found for cluster {cluster_id}. Skipping.")
        return

    logger.info(f"Found {len(source_articles)} source articles for cluster {cluster_id}.")
    source_article_db_ids = [sa['id'] for sa in source_articles]

    synthesized_english_data: Optional[Dict] = None # Will hold the English synthesized content
    processed_cluster_article_id: Optional[UUID] = None 

    # --- 1. English Synthesis (NEW or UPDATE) ---
    if cluster_status == 'NEW':
        logger.info(f"Generating NEW English synthesized article for cluster {cluster_id}.")
        synthesized_english_data = await cluster_article_generator.generate_cluster_article(
            source_articles_data=source_articles,
            language=language # Should be 'English'
        )
        if synthesized_english_data and synthesized_english_data.get('headline') and synthesized_english_data.get('content'):
            new_ca_id = await database.insert_cluster_article(
                cluster_id=cluster_id,
                source_article_ids=source_article_db_ids,
                article_data=synthesized_english_data # This is English data
            )
            if new_ca_id:
                logger.info(f"Successfully inserted new English synthesized article (ID: {new_ca_id}) for cluster {cluster_id}.")
                processed_cluster_article_id = new_ca_id
            else:
                logger.error(f"Failed to insert new English synthesized article for cluster {cluster_id}.")
                return 
        else:
            logger.error(f"Failed to generate English synthesized data for NEW cluster {cluster_id}. Response: {synthesized_english_data}")
            return 

    elif cluster_status == 'UPDATED':
        logger.info(f"Generating UPDATED English synthesized article for cluster {cluster_id}.")
        existing_synthesized_article_db_data = await database.get_existing_cluster_article(cluster_id)

        if not existing_synthesized_article_db_data:
            logger.warning(f"Cluster {cluster_id} is 'UPDATED' but no existing English article found. Treating as NEW.")
            synthesized_english_data = await cluster_article_generator.generate_cluster_article(
                source_articles_data=source_articles,
                language=language
            )
            if synthesized_english_data and synthesized_english_data.get('headline') and synthesized_english_data.get('content'):
                new_ca_id = await database.insert_cluster_article(
                    cluster_id=cluster_id,
                    source_article_ids=source_article_db_ids,
                    article_data=synthesized_english_data
                )
                if new_ca_id:
                    logger.info(f"Successfully inserted (fallback) new English article (ID: {new_ca_id}) for 'UPDATED' cluster {cluster_id}.")
                    processed_cluster_article_id = new_ca_id
                else:
                    logger.error(f"Failed to insert (fallback) new English article for 'UPDATED' cluster {cluster_id}.")
                    return
            else:
                logger.error(f"Failed to generate (fallback) English data for UPDATED cluster {cluster_id}. Response: {synthesized_english_data}")
                return
        else: 
            logger.info(f"Found existing English article (ID: {existing_synthesized_article_db_data['id']}) for cluster {cluster_id} to update.")
            processed_cluster_article_id = UUID(str(existing_synthesized_article_db_data['id']))
            
            synthesized_english_data = await cluster_article_generator.generate_cluster_article(
                source_articles_data=source_articles,
                previous_combined_article_data=existing_synthesized_article_db_data,
                language=language
            )
            if synthesized_english_data and synthesized_english_data.get('headline') and synthesized_english_data.get('content'):
                success = await database.update_cluster_article(
                    cluster_article_id=processed_cluster_article_id,
                    source_article_ids=source_article_db_ids,
                    article_data=synthesized_english_data # English data
                )
                if not success:
                    logger.error(f"Failed to update English synthesized text content for cluster article {processed_cluster_article_id}.")
                    return 
                logger.info(f"Successfully updated English synthesized text content for cluster article {processed_cluster_article_id}.")
            else:
                logger.error(f"Failed to generate English synthesized data for UPDATED cluster {cluster_id}. Response: {synthesized_english_data}")
                return
    else:
        logger.warning(f"Unknown cluster status '{cluster_status}' for cluster {cluster_id}. Skipping.")
        return

    # --- 2. Image Processing (applies if English content was successful) ---
    if processed_cluster_article_id and synthesized_english_data:
        logger.info(f"Attempting to find images for cluster article {processed_cluster_article_id} (Cluster: {cluster_id}).")
        search_query_text = ((synthesized_english_data.get('headline', "") or "").replace("<h1>", "").replace("</h1>", "") + " " + (synthesized_english_data.get('summary', "") or "").replace("<p>", "").replace("</p>", "")).strip()
        if len(search_query_text.split()) < 10 :
             main_content_snippet = (synthesized_english_data.get('content', "") or "").replace("<div>", "").replace("</div>", "").replace("<h2>", "").replace("</h2>", "").replace("<h3>", "").replace("</h3>", "").replace("<p>", "").replace("</p>", "")
             search_query_text += " " + " ".join(main_content_snippet.split()[:50]); search_query_text = search_query_text.strip()
        if not search_query_text:
            logger.warning(f"Search query for images is empty for cluster article {processed_cluster_article_id}. Skipping image search.")
        else:
            logger.info(f"Image search query: '{search_query_text[:200]}...'")
            try:
                images = await image_searcher.search_images(query=search_query_text, num_images=2, content=synthesized_english_data.get('content', ''))
                image1_url, image2_url = (images[0].get('url') if images else None), (images[1].get('url') if len(images) > 1 else None)
                if image1_url or image2_url:
                    if await database.update_cluster_article_images(processed_cluster_article_id, image1_url, image2_url):
                        logger.info(f"Successfully updated images for cluster article {processed_cluster_article_id}.")
                    else: logger.error(f"Failed to update images for cluster article {processed_cluster_article_id}.")
                else: logger.info(f"No valid image URLs to update for cluster article {processed_cluster_article_id}.")
            except Exception as e_img: logger.error(f"Error during image search for {processed_cluster_article_id}: {e_img}", exc_info=True)
        
        # --- 3. Translation Step (applies if English content was successful) ---
        logger.info(f"Attempting translations for cluster article {processed_cluster_article_id}.")
        for lang_code in TARGET_TRANSLATION_LANGUAGES:
            logger.info(f"Processing translation to {lang_code} for cluster article {processed_cluster_article_id}.")
            await cluster_translator.process_and_store_translation(
                cluster_article_id=str(processed_cluster_article_id), # Ensure it's string for DB
                english_data=synthesized_english_data, # Pass the synthesized English data
                target_language_code=lang_code
            )
            # Small delay between language translations for the same article if needed
            await asyncio.sleep(1) 

        # --- 4. Finalize: Mark content processed ---
        logger.info(f"Finalizing processing for cluster {cluster_id}. Marking content as processed.")
        await database.mark_cluster_content_processed(cluster_id)

    elif not processed_cluster_article_id:
        logger.error(f"Skipping image/translation and finalization for cluster {cluster_id} due to earlier text processing failure.")


async def run_cluster_processing_pipeline():
    logger.info("Starting Cluster Processing Pipeline...")
    image_searcher: Optional[ImageSearcher] = None
    try:
        image_searcher = ImageSearcher(use_llm=True) 
        logger.info("ImageSearcher initialized successfully.")
    except Exception as e:
        logger.critical(f"Failed to initialize ImageSearcher: {e}. Image processing will be affected.", exc_info=True)
        # Pipeline can continue for text and translation if ImageSearcher fails to init (logs warning)

    # Process 'NEW' clusters
    logger.info("--- Processing NEW clusters ---")
    new_clusters = await database.fetch_clusters_to_process('NEW')
    if not new_clusters: logger.info("No 'NEW' clusters to process.")
    else:
        logger.info(f"Found {len(new_clusters)} 'NEW' clusters to process.")
        for i, cluster_data in enumerate(new_clusters):
            cluster_id = cluster_data.get('cluster_id')
            if cluster_id:
                logger.info(f"Processing NEW cluster {i+1}/{len(new_clusters)}: {cluster_id}")
                await process_single_cluster(cluster_id, 'NEW', image_searcher) # type: ignore
                if i < len(new_clusters) - 1: logger.info("Waiting 5 seconds before next cluster..."); await asyncio.sleep(5) 
            else: logger.warning(f"Found cluster data without 'cluster_id': {cluster_data}")

    # Process 'UPDATED' clusters
    logger.info("--- Processing UPDATED clusters ---")
    updated_clusters = await database.fetch_clusters_to_process('UPDATED')
    if not updated_clusters: logger.info("No 'UPDATED' clusters to process.")
    else:
        logger.info(f"Found {len(updated_clusters)} 'UPDATED' clusters to process.")
        for i, cluster_data in enumerate(updated_clusters):
            cluster_id = cluster_data.get('cluster_id')
            if cluster_id:
                logger.info(f"Processing UPDATED cluster {i+1}/{len(updated_clusters)}: {cluster_id}")
                await process_single_cluster(cluster_id, 'UPDATED', image_searcher) # type: ignore
                if i < len(updated_clusters) - 1: logger.info("Waiting 5 seconds before next cluster..."); await asyncio.sleep(5)
            else: logger.warning(f"Found cluster data without 'cluster_id': {cluster_data}")

    logger.info("Cluster Processing Pipeline finished.")

if __name__ == "__main__":
    db_ready = database.supabase is not None
    synthesis_llm_ready = (cluster_article_generator.gemini_model is not None and "Error:" not in cluster_article_generator.multi_source_synthesis_prompt_template)
    translation_llm_ready = (cluster_translator.gemini_translator_model is not None and "Error:" not in cluster_translator.component_translation_prompt_template)
    
    # ImageSearcher readiness could also be part of this check if critical
    # For now, pipeline proceeds even if ImageSearcher fails to init (logs warning)

    if db_ready and synthesis_llm_ready and translation_llm_ready:
        asyncio.run(run_cluster_processing_pipeline())
    else:
        if not db_ready: logger.critical("Database client not initialized.")
        if not synthesis_llm_ready: logger.critical("Synthesis LLM model or prompts not initialized correctly.")
        if not translation_llm_ready: logger.critical("Translation LLM model or prompts not initialized correctly.")
        logger.critical("Cluster pipeline cannot start due to critical initialization errors.")