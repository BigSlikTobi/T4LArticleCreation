#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from database import supabase

def reset_test_for_demo():
    """Reset our test to demonstrate working functionality"""
    try:
        # First, find the most recent KC article
        print('=== Finding most recent KC article ===')
        response = supabase.table('SourceArticles').select('id, headline, publishedAt, article_entity_links!inner(entity_id, entity_type)').eq('article_entity_links.entity_id', 'KC').eq('article_entity_links.entity_type', 'team').order('publishedAt', desc=True).limit(1).execute()
        
        if not response.data:
            print("No KC articles found!")
            return
            
        latest_kc_article = response.data[0]
        print(f"Latest KC article: ID {latest_kc_article['id']}, Published: {latest_kc_article['publishedAt']}")
        print(f"Headline: {latest_kc_article['headline']}")
        
        # Delete our test entries to reset
        print('\n=== Deleting test generated_updates ===')
        response = supabase.table('generated_updates').delete().eq('user_id', '00b4b7d6-eabe-4179-955b-3b8a8ab32e95').execute()
        print(f"Deleted test updates")
        
        print('\n=== Test reset complete ===')
        print("Now the personal_summary_generator.py should generate a summary for KC")
        print(f"It will find the KC article from {latest_kc_article['publishedAt']} and create a summary")
        
    except Exception as e:
        print(f'Error: {e}')
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    reset_test_for_demo()
