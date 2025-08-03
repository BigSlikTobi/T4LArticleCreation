#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from database import supabase

def check_recent_data():
    try:
        print('=== Most Recent SourceArticles (DESC order) ===')
        response = supabase.table('SourceArticles').select('id, headline, publishedAt').order('publishedAt', desc=True).limit(10).execute()
        print(f"Found {len(response.data)} articles")
        for article in response.data or []:
            print(f'ID: {article["id"]}, Published: {article["publishedAt"]}, Headline: {article["headline"][:80]}...')
        
        print('\n=== Recent KC Team Articles (DESC order) ===')
        response = supabase.table('SourceArticles').select('id, headline, publishedAt, article_entity_links!inner(entity_id, entity_type)').eq('article_entity_links.entity_id', 'KC').eq('article_entity_links.entity_type', 'team').order('publishedAt', desc=True).limit(5).execute()
        
        print(f"Found {len(response.data)} KC articles")
        for article in response.data or []:
            print(f'ID: {article["id"]}, Published: {article["publishedAt"]}, Headline: {article["headline"][:80]}...')
        
        print('\n=== Our test timestamp comparison ===')
        response = supabase.table('generated_updates').select('created_at, source_article_ids').eq('user_id', '00b4b7d6-eabe-4179-955b-3b8a8ab32e95').order('created_at', desc=True).limit(1).execute()
        if response.data:
            test_timestamp = response.data[0]["created_at"]
            print(f'Last update: {test_timestamp}')
            print(f'Source article IDs: {response.data[0]["source_article_ids"]}')
            
            # Now check if there are any KC articles AFTER this timestamp
            print(f'\n=== KC Articles AFTER {test_timestamp} ===')
            response = supabase.table('SourceArticles').select('id, headline, publishedAt, article_entity_links!inner(entity_id, entity_type)').eq('article_entity_links.entity_id', 'KC').eq('article_entity_links.entity_type', 'team').gt('publishedAt', test_timestamp).order('publishedAt', desc=True).execute()
            
            print(f"Found {len(response.data)} KC articles after test timestamp")
            for article in response.data or []:
                print(f'ID: {article["id"]}, Published: {article["publishedAt"]}, Headline: {article["headline"][:80]}...')
        
    except Exception as e:
        print(f'Error: {e}')
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_recent_data()
