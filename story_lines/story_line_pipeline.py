# story_lines/story_line_pipeline.py

import asyncio
import logging
import os
from typing import List, Dict, Any, Union
from uuid import UUID

from dotenv import load_dotenv

# Add parent directory to import path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the modular components
from story_lines.timeline_generator import (
    load_prompts as load_generator_prompts,
    initialize_llm as initialize_generator_llm,
    generate_timeline_for_cluster
)
from story_lines.timeline_translator import (
    load_prompts as load_translator_prompts,
    initialize_llm as initialize_translator_llm,
    translate_timeline
)
from story_lines.viewpoint_generator import (
    load_viewpoint_prompts,
    initialize_viewpoint_llm,
    determine_viewpoints
)
from story_lines.deep_dive_generator import ( # Added import
    load_deep_dive_prompts,
    initialize_deep_dive_llm,
    generate_deep_dive_for_viewpoint
)
from story_lines.deep_dive_translator import (  # Added import for deep dive translation
    load_prompts as load_deep_dive_translator_prompts,
    initialize_llm as initialize_deep_dive_translator_llm,
    translate_deep_dive
)
from story_lines.article_fetcher import fetch_complete_cluster_data
from database import fetch_all_cluster_ids, save_multiple_story_line_views, get_untranslated_story_line_views

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    force=True
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


async def translate_saved_deep_dives(cluster_id: str) -> int:
    """
    Translate all untranslated deep dive articles for a specific cluster to German.
    
    Args:
        cluster_id: The cluster ID to translate deep dives for
        
    Returns:
        int: Number of deep dives successfully translated
    """
    try:
        # Get all untranslated story line views
        all_untranslated_views = await get_untranslated_story_line_views(language_code='de')
        
        # Filter for the specific cluster
        untranslated_views = [
            view for view in all_untranslated_views 
            if str(view.get('cluster_id')) == str(cluster_id)
        ]
        
        if not untranslated_views:
            logger.info(f"PIPELINE: No untranslated deep dives found for cluster {cluster_id}")
            return 0
            
        logger.info(f"PIPELINE: Found {len(untranslated_views)} deep dives to translate for cluster {cluster_id}")
        
        translation_count = 0
        for view in untranslated_views:
            story_line_view_id = view.get('id')
            view_name = view.get('view', 'Unknown View')
            
            # Prepare deep dive data
            deep_dive_data = {
                "headline": view.get('headline', ''),
                "content": view.get('content', ''),
                "introduction": view.get('introduction', ''),
                "view": view.get('view', ''),
                "justification": view.get('justification', '')
            }
            
            if story_line_view_id and any(deep_dive_data.values()):
                logger.info(f"PIPELINE: Translating deep dive '{view_name}' (ID: {story_line_view_id})")
                success, _ = await translate_deep_dive(story_line_view_id, deep_dive_data, language_code='de')
                
                if success:
                    translation_count += 1
                    logger.info(f"PIPELINE: Successfully translated deep dive '{view_name}'")
                else:
                    logger.warning(f"PIPELINE: Failed to translate deep dive '{view_name}'")
                    
                # Small delay between translations
                await asyncio.sleep(1)
            else:
                logger.warning(f"PIPELINE: Skipping translation for deep dive with missing data: {view}")
                
        logger.info(f"PIPELINE: Translated {translation_count}/{len(untranslated_views)} deep dives for cluster {cluster_id}")
        return translation_count
        
    except Exception as e:
        logger.error(f"PIPELINE: Error during deep dive translation for cluster {cluster_id}: {e}", exc_info=True)
        return 0


async def process_cluster_end_to_end(cluster_id_str: str) -> Dict[str, Any]: # Ensure cluster_id is str
    """
    Process a single cluster through the entire pipeline:
    1. Generate English timeline
    2. Translate to German
    3. Determine viewpoints for deep dive
    4. Generate deep-dive articles for each viewpoint
    
    Args:
        cluster_id_str: The ID of the cluster to process (as a string)
    
    Returns:
        Dict with results of each step
    """
    result = {
        "cluster_id": cluster_id_str,
        "timeline_generation_success": False,
        "timeline_translation_success": False,
        "viewpoint_determination_success": False,
        "deep_dive_generation_count": 0,
        "story_line_views_saved": 0,
        "deep_dive_translations_count": 0,
        "timeline_id": None,
        "viewpoints": [],
        "deep_dive_articles": [] # To store generated deep dive content (optional for now)
    }
    
    # Step 1: Generate English timeline
    logger.info(f"PIPELINE: Starting timeline generation for cluster {cluster_id_str}")
    timeline_id_obj, timeline_data = await generate_timeline_for_cluster(cluster_id_str)
    
    if not timeline_id_obj or not timeline_data:
        logger.error(f"PIPELINE: Failed to generate timeline for cluster {cluster_id_str}")
    else:
        result["timeline_generation_success"] = True
        result["timeline_id"] = str(timeline_id_obj) # Store as string
    
        # Step 2: Translate to German (only if timeline generation was successful)
        logger.info(f"PIPELINE: Starting German translation for timeline {result['timeline_id']} (cluster {cluster_id_str})")
        translation_success, _ = await translate_timeline(
            timeline_id=result['timeline_id'],
            timeline_data=timeline_data,
            language_code='de'
        )
        result["timeline_translation_success"] = translation_success

    # Step 3: Determine Viewpoints
    logger.info(f"PIPELINE: Fetching cluster data for viewpoint determination for cluster {cluster_id_str}")
    # Ensure cluster_id passed to fetch_complete_cluster_data is in the correct type (str or UUID)
    # The function itself handles str(cluster_id) if it needs to.
    cluster_full_data = await fetch_complete_cluster_data(cluster_id_str)

    if cluster_full_data:
        logger.info(f"PIPELINE: Determining viewpoints for cluster {cluster_id_str}")
        viewpoints = await determine_viewpoints(cluster_full_data)
        if viewpoints:
            result["viewpoint_determination_success"] = True
            result["viewpoints"] = viewpoints
            
            print(f"\n--- Viewpoints for Cluster {cluster_id_str} ---")
            for i, vp in enumerate(viewpoints):
                print(f"  Viewpoint {i+1}: {vp.get('name', 'N/A')}")
                print(f"    Justification: {vp.get('justification', 'N/A')}")
            print("--- End Viewpoints ---\n")

            # Step 4: Generate Deep-Dive Articles for each viewpoint
            if result["viewpoint_determination_success"]:
                logger.info(f"PIPELINE: Starting deep-dive article generation for {len(viewpoints)} viewpoints in cluster {cluster_id_str}")
                for i, viewpoint in enumerate(viewpoints):
                    logger.info(f"PIPELINE: Generating deep dive {i+1}/{len(viewpoints)} for viewpoint: '{viewpoint.get('name')}'")
                    deep_dive_content = await generate_deep_dive_for_viewpoint(
                        cluster_data=cluster_full_data,
                        viewpoint=viewpoint,
                        cluster_id=cluster_id_str
                    )
                    if deep_dive_content:
                        result["deep_dive_generation_count"] += 1
                        result["deep_dive_articles"].append(
                            {"viewpoint_name": viewpoint.get("name"), **deep_dive_content}
                        )
                    else:
                        logger.error(f"PIPELINE: Failed to generate deep dive for viewpoint '{viewpoint.get('name')}' in cluster {cluster_id_str}")
                    
                    if i < len(viewpoints) -1: # Add delay between deep dives for same cluster
                        logger.info("PIPELINE: Waiting 1 second before next deep-dive generation...")
                        await asyncio.sleep(1)
                
                # Step 5: Save story line views to database
                if result["deep_dive_generation_count"] > 0:
                    logger.info(f"PIPELINE: === SAVING STORY LINE VIEWS TO DATABASE ===")
                    logger.info(f"PIPELINE: Saving {result['deep_dive_generation_count']} story line views to database for cluster {cluster_id_str}")
                    
                    # Prepare viewpoints and articles for database saving
                    viewpoints_for_db = viewpoints[:result["deep_dive_generation_count"]]  # Match the number of successful articles
                    articles_for_db = [article for article in result["deep_dive_articles"]]
                    
                    try:
                        save_stats = await save_multiple_story_line_views(
                            cluster_id=cluster_id_str,
                            viewpoints=viewpoints_for_db,
                            deep_dive_articles=articles_for_db
                        )
                        
                        result["story_line_views_saved"] = save_stats["saved"]
                        
                        if save_stats["saved"] > 0:
                            logger.info(f"PIPELINE: Successfully saved {save_stats['saved']} story line views for cluster {cluster_id_str}")
                        
                        if save_stats["failed"] > 0:
                            logger.warning(f"PIPELINE: Failed to save {save_stats['failed']} story line views for cluster {cluster_id_str}")
                            
                    except Exception as e:
                        logger.error(f"PIPELINE: Error saving story line views for cluster {cluster_id_str}: {e}", exc_info=True)
                
                # Step 6: Translate deep dive articles to German
                logger.info(f"PIPELINE: Starting deep dive translation for cluster {cluster_id_str}")
                translation_count = await translate_saved_deep_dives(cluster_id_str)
                result["deep_dive_translations_count"] = translation_count
                        
        else:
            logger.warning(f"PIPELINE: No viewpoints determined for cluster {cluster_id_str}, skipping deep-dive generation.")
    else:
        logger.warning(f"PIPELINE: Could not fetch full data for cluster {cluster_id_str}, skipping viewpoint and deep-dive generation.")
    
    logger.info(f"PIPELINE: Completed end-to-end processing for cluster {cluster_id_str}")
    return result


async def main():
    """Main function that runs the complete pipeline for all clusters"""
    try:
        # Step 1: Initialize LLMs and Prompts
        logger.info("PIPELINE: Initializing LLMs and loading prompts...")
        
        logger.info("PIPELINE: Initializing Timeline Generator resources")
        load_generator_prompts()
        await initialize_generator_llm(model_type="default")
        
        logger.info("PIPELINE: Initializing Timeline Translator resources")
        load_translator_prompts()
        await initialize_translator_llm(model_type="flash")
        
        logger.info("PIPELINE: Initializing Viewpoint Generator resources")
        load_viewpoint_prompts()
        await initialize_viewpoint_llm(model_type="flash")

        logger.info("PIPELINE: Initializing Deep-Dive Generator resources")
        load_deep_dive_prompts()
        await initialize_deep_dive_llm(model_type="default") # Using default for quality generation, grounding enabled by default
        
        logger.info("PIPELINE: Initializing Deep-Dive Translator resources")
        load_deep_dive_translator_prompts()
        await initialize_deep_dive_translator_llm(model_type="flash")
        
        # Step 2: Get clusters to process
        cluster_ids_list_objs = await fetch_all_cluster_ids() # Returns List[Union[str, UUID]]
        
        if not cluster_ids_list_objs:
            logger.info("PIPELINE: No clusters found that need processing")
            return
            
        logger.info(f"PIPELINE: Found {len(cluster_ids_list_objs)} clusters to process")
        
        # Step 3: Process each cluster end-to-end
        all_results = []
        for i, cluster_id_obj in enumerate(cluster_ids_list_objs):
            cluster_id_str = str(cluster_id_obj) # Convert UUID to string if necessary
            if cluster_id_str:
                logger.info(f"PIPELINE: Processing cluster {i+1}/{len(cluster_ids_list_objs)}: {cluster_id_str}")
                single_result = await process_cluster_end_to_end(cluster_id_str)
                all_results.append(single_result)
                if i < len(cluster_ids_list_objs) - 1:
                    logger.info("PIPELINE: Waiting 5 seconds before next cluster to manage load/rate limits...")
                    await asyncio.sleep(5) 
            else:
                logger.warning("PIPELINE: Encountered an empty or None cluster_id in the list.")
                
        # Summary statistics
        total_clusters = len(all_results)
        successful_timeline_generation = sum(1 for r in all_results if r["timeline_generation_success"])
        successful_timeline_translation = sum(1 for r in all_results if r["timeline_translation_success"])
        successful_viewpoint_determination = sum(1 for r in all_results if r["viewpoint_determination_success"])
        total_deep_dives_generated = sum(r["deep_dive_generation_count"] for r in all_results)
        total_story_line_views_saved = sum(r["story_line_views_saved"] for r in all_results)
        total_deep_dive_translations = sum(r["deep_dive_translations_count"] for r in all_results)
        
        logger.info(f"\n--- PIPELINE SUMMARY ---")
        logger.info(f"Total clusters processed: {total_clusters}")
        logger.info(f"  Successful timeline generations: {successful_timeline_generation}/{total_clusters}")
        logger.info(f"  Successful timeline translations: {successful_timeline_translation}/{total_clusters}")
        logger.info(f"  Successful viewpoint determinations: {successful_viewpoint_determination}/{total_clusters}")
        logger.info(f"  Total deep-dive articles generated: {total_deep_dives_generated}")
        logger.info(f"  Total story line views saved to database: {total_story_line_views_saved}")
        logger.info(f"  Total deep-dive articles translated: {total_deep_dive_translations}")
        logger.info(f"--- END PIPELINE SUMMARY ---")
        
        return all_results
            
    except Exception as e:
        logger.critical(f"PIPELINE FAILURE: An unexpected error occurred: {e}", exc_info=True)
        return None


if __name__ == "__main__":
    asyncio.run(main())