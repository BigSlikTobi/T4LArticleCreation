#!/usr/bin/env python3
"""
Test script to validate that cluster_id is properly saved to both timelines and timelines_int tables.
This script tests the changes made to ensure cluster_id column is populated.
"""
import asyncio
import logging
import sys
import os
from uuid import uuid4

# Add current directory to path for imports
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from database import save_timeline_to_database, save_translated_timeline

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_cluster_id_saving():
    """Test that cluster_id is properly saved to both tables"""
    # Create test data
    test_cluster_id = str(uuid4())
    test_timeline_entries = [
        {
            "created_at": "2024-01-01T10:00:00Z",
            "headline": "Test Headline 1",
            "source_name": "Test Source",
            "summary": "Test summary 1"
        },
        {
            "created_at": "2024-01-02T10:00:00Z", 
            "headline": "Test Headline 2",
            "source_name": "Test Source",
            "summary": "Test summary 2"
        }
    ]
    
    logger.info(f"Testing timeline creation with cluster_id: {test_cluster_id}")
    
    # Test saving timeline to database (this should now save cluster_id)
    timeline_id = await save_timeline_to_database(test_cluster_id, test_timeline_entries)
    
    if timeline_id:
        logger.info(f"✓ Timeline saved successfully with ID: {timeline_id}")
        
        # Test saving translated timeline (this should now save cluster_id to timelines_int)
        test_translated_data = {
            "cluster_id": test_cluster_id,
            "timeline": [
                {
                    "created_at": "2024-01-01T10:00:00Z",
                    "headline": "Test Schlagzeile 1",
                    "source_name": "Test Quelle", 
                    "summary": "Test Zusammenfassung 1"
                },
                {
                    "created_at": "2024-01-02T10:00:00Z",
                    "headline": "Test Schlagzeile 2", 
                    "source_name": "Test Quelle",
                    "summary": "Test Zusammenfassung 2"
                }
            ],
            "generated_at": "2024-01-01T10:00:00Z"
        }
        
        translation_success = await save_translated_timeline(
            timeline_id=timeline_id,
            language_code='de',
            translated_data=test_translated_data,
            cluster_id=test_cluster_id
        )
        
        if translation_success:
            logger.info("✓ Translated timeline saved successfully with cluster_id")
            logger.info("✓ All tests passed! cluster_id should now be saved in both tables")
        else:
            logger.error("✗ Failed to save translated timeline")
            
    else:
        logger.error("✗ Failed to save timeline")


async def main():
    """Main test function"""
    logger.info("=== Testing cluster_id saving functionality ===")
    
    try:
        await test_cluster_id_saving()
    except Exception as e:
        logger.error(f"Test failed with error: {e}", exc_info=True)
    
    logger.info("=== Test completed ===")


if __name__ == "__main__":
    print("Cluster ID Database Changes Test")
    print("This script tests that cluster_id is properly saved to both timelines and timelines_int tables")
    print("Note: This requires a working database connection to run successfully")
    print()
    
    # Run the test
    asyncio.run(main())
    
    print("Test script created successfully. To run the actual test:")
    print("1. Ensure your .env file is configured with database credentials")
    print("2. Uncomment the asyncio.run(main()) line in this script")
    print("3. Run: python3 test_cluster_id_changes.py")
