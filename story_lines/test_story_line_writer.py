#!/usr/bin/env python3
"""
Test script for the story line writer functionality.
This script demonstrates how to use the story line writer to generate
viewpoints and deep dive articles from cluster data.
"""

import asyncio
import logging
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from story_lines.story_line_writer import (
    initialize_story_line_writer,
    process_cluster_for_story_lines,
    process_multiple_clusters
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_single_cluster():
    """Test processing a single cluster for story lines"""
    print("=== Testing Single Cluster Processing ===")
    
    # Replace with actual cluster ID from your database
    test_cluster_id = "your-cluster-id-here"
    
    # Initialize the story line writer
    logger.info("Initializing story line writer...")
    success = await initialize_story_line_writer()
    if not success:
        logger.error("Failed to initialize story line writer. Exiting test.")
        return
    
    # Process the cluster
    logger.info(f"Processing cluster: {test_cluster_id}")
    result = await process_cluster_for_story_lines(test_cluster_id)
    
    # Display results
    print(f"\n--- PROCESSING RESULTS ---")
    print(f"Cluster ID: {result['cluster_id']}")
    print(f"Success: {result['success']}")
    print(f"Viewpoints Generated: {result['viewpoints_generated']}")
    print(f"Articles Generated: {result['articles_generated']}")
    print(f"Articles Saved: {result['articles_saved']}")
    if result.get('error'):
        print(f"Error: {result['error']}")
    
    return result


async def test_multiple_clusters():
    """Test processing multiple clusters for story lines"""
    print("\n=== Testing Multiple Cluster Processing ===")
    
    # Replace with actual cluster IDs from your database
    test_cluster_ids = [
        "cluster-id-1",
        "cluster-id-2", 
        "cluster-id-3"
    ]
    
    # Initialize the story line writer
    logger.info("Initializing story line writer...")
    success = await initialize_story_line_writer()
    if not success:
        logger.error("Failed to initialize story line writer. Exiting test.")
        return
    
    # Process multiple clusters
    logger.info(f"Processing {len(test_cluster_ids)} clusters")
    results = await process_multiple_clusters(test_cluster_ids)
    
    # Display overall results
    print(f"\n--- OVERALL RESULTS ---")
    print(f"Total Clusters: {results['total_clusters']}")
    print(f"Successful: {results['successful_clusters']}")
    print(f"Failed: {results['failed_clusters']}")
    print(f"Total Viewpoints: {results['total_viewpoints']}")
    print(f"Total Articles Generated: {results['total_articles_generated']}")
    print(f"Total Articles Saved: {results['total_articles_saved']}")
    
    # Display individual results
    print(f"\n--- INDIVIDUAL CLUSTER RESULTS ---")
    for result in results['individual_results']:
        print(f"Cluster {result['cluster_id']}: "
              f"Success={result['success']}, "
              f"VP={result['viewpoints_generated']}, "
              f"Gen={result['articles_generated']}, "
              f"Saved={result['articles_saved']}")
        if result.get('error'):
            print(f"  Error: {result['error']}")
    
    return results


async def main():
    """Main test function"""
    print("Story Line Writer Test Suite")
    print("=" * 50)
    
    try:
        # Test single cluster processing
        await test_single_cluster()
        
        # Test multiple cluster processing  
        await test_multiple_clusters()
        
        print("\n" + "=" * 50)
        print("Test suite completed!")
        
    except Exception as e:
        logger.error(f"Test suite failed with error: {e}", exc_info=True)


if __name__ == "__main__":
    print("Story Line Writer Test")
    print("Note: Replace the test cluster IDs with actual ones from your database")
    print("Make sure your .env file is configured with the required API keys")
    print()
    
    # Run the test suite
    asyncio.run(main())
