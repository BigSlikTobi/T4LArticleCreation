# story_lines/story_line_writer.py

import asyncio
import logging
import os
import sys
from typing import Dict, List, Optional, Union
from uuid import UUID

# Add parent directory to import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from database import save_multiple_story_line_views
from story_lines.viewpoint_generator import (
    load_viewpoint_prompts,
    initialize_viewpoint_llm,
    determine_viewpoints
)
from story_lines.deep_dive_generator import (
    load_deep_dive_prompts,
    initialize_deep_dive_llm,
    generate_deep_dive_for_viewpoint
)
from story_lines.article_fetcher import fetch_complete_cluster_data, validate_cluster_data_for_analysis

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    force=True
)
logger = logging.getLogger(__name__)


async def initialize_story_line_writer():
    """Initialize all components needed for story line writing"""
    logger.info("Initializing Story Line Writer components...")
    
    try:
        # Load prompts
        load_viewpoint_prompts()
        load_deep_dive_prompts()
        
        # Initialize LLMs
        await initialize_viewpoint_llm(model_type="flash")  # Good for classification/extraction
        await initialize_deep_dive_llm(model_type="default")  # Better for generation with grounding
        
        logger.info("Story Line Writer initialization complete.")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize Story Line Writer: {e}", exc_info=True)
        return False


async def process_cluster_for_story_lines(cluster_id: Union[str, UUID]) -> Dict[str, any]:
    """
    Process a single cluster to generate viewpoints, deep dive articles, and save to database.
    
    Args:
        cluster_id: The cluster ID to process
        
    Returns:
        Dict with processing results and statistics
    """
    result = {
        "cluster_id": str(cluster_id),
        "success": False,
        "viewpoints_generated": 0,
        "articles_generated": 0,
        "articles_saved": 0,
        "error": None
    }
    
    try:
        logger.info(f"Processing cluster {cluster_id} for story lines...")
        
        # Step 1: Fetch cluster data
        cluster_data = await fetch_complete_cluster_data(cluster_id)
        if not cluster_data:
            result["error"] = "Failed to fetch cluster data"
            return result
        
        # Step 1.5: Validate cluster data
        if not validate_cluster_data_for_analysis(cluster_data):
            result["error"] = "Cluster data insufficient for analysis"
            return result
        
        # Step 2: Determine viewpoints
        viewpoints = await determine_viewpoints(cluster_data)
        if not viewpoints:
            result["error"] = "No viewpoints generated"
            return result
        
        result["viewpoints_generated"] = len(viewpoints)
        logger.info(f"Generated {len(viewpoints)} viewpoints for cluster {cluster_id}")
        
        # Step 3: Generate deep dive articles for each viewpoint
        deep_dive_articles = []
        for i, viewpoint in enumerate(viewpoints):
            logger.info(f"Generating deep dive {i+1}/{len(viewpoints)} for viewpoint: '{viewpoint.get('name')}'")
            
            article = await generate_deep_dive_for_viewpoint(
                cluster_data=cluster_data,
                viewpoint=viewpoint,
                cluster_id=str(cluster_id)
            )
            
            if article:
                deep_dive_articles.append(article)
            else:
                logger.warning(f"Failed to generate deep dive for viewpoint: '{viewpoint.get('name')}'")
        
        result["articles_generated"] = len(deep_dive_articles)
        
        if not deep_dive_articles:
            result["error"] = "No deep dive articles generated"
            return result
        
        # Step 4: Save to database
        save_stats = await save_multiple_story_line_views(
            cluster_id=cluster_id,
            viewpoints=viewpoints[:len(deep_dive_articles)],  # Match the length
            deep_dive_articles=deep_dive_articles
        )
        
        result["articles_saved"] = save_stats["saved"]
        result["success"] = save_stats["saved"] > 0
        
        if save_stats["failed"] > 0:
            result["error"] = f"Some articles failed to save: {save_stats['failed']}/{save_stats['total']}"
        
        logger.info(f"Cluster {cluster_id} processing complete. Generated: {result['articles_generated']}, Saved: {result['articles_saved']}")
        
    except Exception as e:
        logger.error(f"Error processing cluster {cluster_id}: {e}", exc_info=True)
        result["error"] = str(e)
    
    return result


async def process_multiple_clusters(cluster_ids: List[Union[str, UUID]]) -> Dict[str, any]:
    """
    Process multiple clusters for story line generation.
    
    Args:
        cluster_ids: List of cluster IDs to process
        
    Returns:
        Dict with overall statistics and individual results
    """
    overall_stats = {
        "total_clusters": len(cluster_ids),
        "successful_clusters": 0,
        "failed_clusters": 0,
        "total_viewpoints": 0,
        "total_articles_generated": 0,
        "total_articles_saved": 0,
        "individual_results": []
    }
    
    logger.info(f"Processing {len(cluster_ids)} clusters for story line generation...")
    
    for i, cluster_id in enumerate(cluster_ids):
        logger.info(f"\n--- Processing cluster {i+1}/{len(cluster_ids)}: {cluster_id} ---")
        
        result = await process_cluster_for_story_lines(cluster_id)
        overall_stats["individual_results"].append(result)
        
        if result["success"]:
            overall_stats["successful_clusters"] += 1
        else:
            overall_stats["failed_clusters"] += 1
            logger.warning(f"Cluster {cluster_id} failed: {result.get('error', 'Unknown error')}")
        
        overall_stats["total_viewpoints"] += result["viewpoints_generated"]
        overall_stats["total_articles_generated"] += result["articles_generated"]
        overall_stats["total_articles_saved"] += result["articles_saved"]
    
    logger.info(f"\n=== STORY LINE GENERATION COMPLETE ===")
    logger.info(f"Total clusters processed: {overall_stats['total_clusters']}")
    logger.info(f"Successful: {overall_stats['successful_clusters']}")
    logger.info(f"Failed: {overall_stats['failed_clusters']}")
    logger.info(f"Total viewpoints generated: {overall_stats['total_viewpoints']}")
    logger.info(f"Total articles generated: {overall_stats['total_articles_generated']}")
    logger.info(f"Total articles saved: {overall_stats['total_articles_saved']}")
    
    return overall_stats


if __name__ == "__main__":
    async def test_story_line_writer():
        """Test the story line writer with mock data"""
        logger.info("Testing Story Line Writer...")
        
        # Initialize components
        success = await initialize_story_line_writer()
        if not success:
            logger.error("Failed to initialize components. Exiting test.")
            return
        
        # Test with a mock cluster ID (replace with actual cluster ID for real testing)
        test_cluster_id = "test-cluster-123"
        
        # You can also test with multiple clusters:
        # test_cluster_ids = ["cluster-1", "cluster-2", "cluster-3"]
        # results = await process_multiple_clusters(test_cluster_ids)
        
        result = await process_cluster_for_story_lines(test_cluster_id)
        
        print(f"\nTest Result: {result}")
        
        logger.info("Story Line Writer test completed.")
    
    asyncio.run(test_story_line_writer())
