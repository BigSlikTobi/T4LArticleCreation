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
from LLMSetup import initialize_model

async def test_end_to_end_summary():
    """Test the complete personalized summary generation workflow"""
    
    user_id = "00b4b7d6-eabe-4179-955b-3b8a8ab32e95"
    entity_id = "KC"
    entity_type = "team"
    
    print("=== End-to-End Personalized Summary Test ===")
    
    # Step 1: Fetch articles from a longer time period (30 days)
    since_timestamp = "2025-03-01T00:00:00+00:00"  # Go back far enough to catch the KC article
    print(f"🔍 Fetching KC articles since {since_timestamp}")
    
    articles = await fetch_new_articles_for_entity(entity_id, entity_type, since_timestamp)
    print(f"✅ Found {len(articles)} articles for {entity_id}")
    
    if not articles:
        print("❌ No articles found - cannot test summary generation")
        return
    
    # Display found articles
    for article in articles[:3]:  # Show first 3
        print(f"   📰 ID: {article['id']}, Published: {article['publishedAt']}")
        print(f"      Headline: {article['headline'][:80]}...")
    
    # Step 2: Check for previous summary
    print(f"\n🔍 Checking for previous summary...")
    previous_summary = await get_previous_summary_for_entity(user_id, entity_id)
    print(f"✅ Previous summary: {'Found' if previous_summary else 'None'}")
    
    # Step 3: Generate a new summary using LLM
    print(f"\n🤖 Generating personalized summary...")
    
    try:
        # Initialize the Gemini model
        model_config = initialize_model("gemini", "default", grounding_enabled=True)
        model_name = model_config["model_name"]
        model = model_config["model"]
        
        print(f"🤖 Using model: {model_name}")
        
        # Create a simple prompt
        prompt = f"""
        Create a personalized summary for a Kansas City Chiefs fan based on these recent articles:
        
        {chr(10).join([f"- {article['headline']}" for article in articles[:3]])}
        
        Previous summary: {previous_summary or "None"}
        
        Generate a 2-3 sentence update focusing on what's new and relevant for a Chiefs fan.
        """
        
        response = model.generate_content(
            prompt,
            config={
                'temperature': 0.7,
                'max_output_tokens': 1000,
                'tools': model_config.get("tools") if model_config.get("tools") else None
            }
        )
        summary_content = response.text
        
        print(f"✅ Generated summary: {summary_content[:100]}...")
        
        # Step 4: Save the summary
        print(f"\n💾 Saving generated summary...")
        article_ids = [article['id'] for article in articles[:3]]
        
        success = await save_generated_update(
            user_id=user_id,
            entity_id=entity_id,
            entity_type=entity_type,
            update_content=summary_content,
            source_article_ids=article_ids
        )
        
        if success:
            print(f"✅ Summary saved successfully!")
            print(f"   📊 Used {len(article_ids)} source articles")
            print(f"   📝 Summary length: {len(summary_content)} characters")
        else:
            print(f"❌ Failed to save summary")
            
    except Exception as e:
        print(f"❌ Error generating summary: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_end_to_end_summary())
