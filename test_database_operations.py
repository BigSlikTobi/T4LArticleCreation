#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import asyncio
from datetime import datetime, timezone
from database import (
    fetch_new_articles_for_entity, 
    save_generated_update,
    get_previous_summary_for_entity
)

async def test_database_operations():
    """Test the database operations without LLM"""
    
    user_id = "00b4b7d6-eabe-4179-955b-3b8a8ab32e95"
    entity_id = "KC"
    entity_type = "team"
    
    print("=== Database Operations Test ===")
    
    # Step 1: Fetch articles from a longer time period (30 days)
    since_timestamp = "2025-03-01T00:00:00+00:00"  # Go back far enough to catch the KC article
    print(f"🔍 Fetching KC articles since {since_timestamp}")
    
    articles = await fetch_new_articles_for_entity(entity_id, entity_type, since_timestamp)
    print(f"✅ Found {len(articles)} articles for {entity_id}")
    
    if not articles:
        print("❌ No articles found - cannot test summary generation")
        return
    
    # Display found articles
    for i, article in enumerate(articles[:3]):  # Show first 3
        print(f"   📰 Article {i+1}: ID {article['id']}, Published: {article['publishedAt']}")
        print(f"      Headline: {article['headline'][:80]}...")
        if 'Content' in article and article['Content']:
            print(f"      Content preview: {article['Content'][:100]}...")
    
    # Step 2: Check for previous summary
    print(f"\n🔍 Checking for previous summary...")
    previous_summary = await get_previous_summary_for_entity(user_id, entity_id)
    print(f"✅ Previous summary: {'Found' if previous_summary else 'None'}")
    if previous_summary:
        print(f"   Previous summary: {previous_summary[:100]}...")
    
    # Step 3: Create a mock summary (instead of using LLM)
    print(f"\n📝 Creating test summary...")
    
    headlines = [article['headline'] for article in articles[:3]]
    mock_summary = f"Recent updates for Kansas City Chiefs: {'; '.join(headlines[:2])}. This covers the latest news as of {datetime.now().strftime('%Y-%m-%d')}."
    
    print(f"✅ Test summary created: {mock_summary[:100]}...")
    
    # Step 4: Save the summary
    print(f"\n💾 Saving generated summary...")
    article_ids = [article['id'] for article in articles[:3]]
    
    success = await save_generated_update(
        user_id=user_id,
        entity_id=entity_id,
        entity_type=entity_type,
        update_content=mock_summary,
        source_article_ids=article_ids
    )
    
    if success:
        print(f"✅ Summary saved successfully!")
        print(f"   📊 Used {len(article_ids)} source articles")
        print(f"   📝 Summary length: {len(mock_summary)} characters")
        print(f"   🔗 Article IDs: {article_ids}")
        
        # Step 5: Verify we can retrieve it
        print(f"\n🔍 Verifying saved summary...")
        retrieved_summary = await get_previous_summary_for_entity(user_id, entity_id)
        
        if retrieved_summary:
            print(f"✅ Successfully retrieved saved summary!")
            print(f"   Retrieved: {retrieved_summary[:100]}...")
            if retrieved_summary == mock_summary:
                print("✅ Retrieved summary matches saved summary!")
            else:
                print("⚠️  Retrieved summary differs from saved summary")
        else:
            print("❌ Could not retrieve saved summary")
    else:
        print(f"❌ Failed to save summary")

if __name__ == "__main__":
    asyncio.run(test_database_operations())
