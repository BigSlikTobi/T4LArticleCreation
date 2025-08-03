#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import asyncio
from datetime import datetime, timezone
from database import (
    fetch_user_preferences,
    save_generated_update,
    supabase
)

async def create_test_scenario():
    """Create a test scenario to validate individual entity summary generation"""
    
    user_id = "00b4b7d6-eabe-4179-955b-3b8a8ab32e95"
    
    print("=== Creating Test Scenario for Individual Entity Summaries ===")
    
    # Step 1: Clear existing test data
    print("🧹 Clearing existing generated_updates...")
    supabase.table("generated_updates").delete().eq("user_id", user_id).execute()
    
    # Step 2: Check user preferences
    print("👤 Checking user preferences...")
    preferences = await fetch_user_preferences(user_id)
    print(f"Found {len(preferences)} preferences:")
    for pref in preferences:
        print(f"   - {pref['entity_type']}: {pref['entity_id']}")
    
    # Step 3: Create test summaries for each entity
    print("\n📝 Creating individual test summaries for each entity...")
    
    for i, pref in enumerate(preferences):
        entity_id = pref['entity_id']
        entity_type = pref['entity_type']
        
        # Create unique summary content for each entity
        summary_content = f"Test summary #{i+1} for {entity_type} {entity_id} - Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        print(f"   💾 Saving summary for {entity_type} {entity_id}...")
        success = await save_generated_update(
            user_id=user_id,
            entity_id=entity_id,
            entity_type=entity_type,
            update_content=summary_content,
            source_article_ids=[1000 + i],  # Mock article IDs
            source_stat_ids=[2000 + i] if entity_type == "player" else []
        )
        
        if success:
            print(f"   ✅ Successfully saved summary for {entity_type} {entity_id}")
        else:
            print(f"   ❌ Failed to save summary for {entity_type} {entity_id}")
    
    # Step 4: Verify the results
    print("\n🔍 Verifying individual summaries in database...")
    response = supabase.table("generated_updates").select("*").eq("user_id", user_id).order("created_at", desc=True).execute()
    
    print(f"Found {len(response.data)} summaries in database:")
    for update in response.data:
        print(f"   📄 Update ID: {update['update_id']}")
        print(f"      Content: {update['update_content'][:80]}...")
        print(f"      Source Articles: {update['source_article_ids']}")
        print(f"      Source Stats: {update['source_stat_ids']}")
        print(f"      Created: {update['created_at']}")
        print()
    
    # Step 5: Verify entity tracking
    print("🔍 Verifying entity tracking...")
    for pref in preferences:
        entity_id = pref['entity_id']
        entity_type = pref['entity_type']
        
        # Check if we can find this entity's summary
        response = supabase.table("generated_updates").select("*").eq("user_id", user_id).filter("source_article_ids", "cs", f'["{entity_id}"]').execute()
        
        if response.data:
            print(f"   ✅ Found summary for {entity_type} {entity_id}")
            print(f"      Content: {response.data[0]['update_content'][:60]}...")
        else:
            print(f"   ❌ No summary found for {entity_type} {entity_id}")
    
    print("\n✅ Test scenario setup complete!")
    print("Now the personal_summary_generator.py should detect these existing summaries")
    print("and skip generating new ones unless there's new content since these timestamps.")

if __name__ == "__main__":
    asyncio.run(create_test_scenario())
