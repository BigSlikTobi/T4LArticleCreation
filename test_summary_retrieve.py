#!/usr/bin/env python3
"""
Test script to verify that the personal summary generator can retrieve previously saved summaries.
"""

import asyncio
import logging
import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import get_previous_summary_for_entity, get_last_update_timestamp

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_retrieve_summary():
    """Test that we can retrieve a previously saved summary"""
    test_user_id = "00b4b7d6-eabe-4179-955b-3b8a8ab32e95"
    test_entity_id = "KC"
    
    logger.info(f"Testing retrieval of summary for user {test_user_id}, entity {test_entity_id}")
    
    # Test getting the last update timestamp
    timestamp = await get_last_update_timestamp(test_user_id, test_entity_id)
    if timestamp:
        logger.info(f"✅ Found last update timestamp: {timestamp}")
    else:
        logger.warning("⚠️ No last update timestamp found")
    
    # Test getting the previous summary
    summary = await get_previous_summary_for_entity(test_user_id, test_entity_id)
    if summary:
        logger.info(f"✅ Found previous summary: {summary[:100]}...")
        return True
    else:
        logger.warning("⚠️ No previous summary found")
        return False

async def main():
    success = await test_retrieve_summary()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    asyncio.run(main())
