#!/usr/bin/env python3
"""
Test script to verify that the personal summary generator can save summaries
when there is content to process.
"""

import asyncio
import logging
import sys
import os
from datetime import datetime, timezone

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import save_generated_update

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_save_summary():
    """Test that we can save a generated summary"""
    test_user_id = "00b4b7d6-eabe-4179-955b-3b8a8ab32e95"
    test_entity_id = "KC"
    test_entity_type = "team"
    test_content = "This is a test summary for the Kansas City Chiefs generated on " + datetime.now().isoformat()
    test_article_ids = ["123", "456"]
    test_stat_ids = []
    
    logger.info(f"Testing save of summary for user {test_user_id}, entity {test_entity_id}")
    
    success = await save_generated_update(
        user_id=test_user_id,
        entity_id=test_entity_id,
        entity_type=test_entity_type,
        update_content=test_content,
        source_article_ids=test_article_ids,
        source_stat_ids=test_stat_ids
    )
    
    if success:
        logger.info("✅ Successfully saved test summary!")
        return True
    else:
        logger.error("❌ Failed to save test summary")
        return False

async def main():
    success = await test_save_summary()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    asyncio.run(main())
