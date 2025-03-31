import asyncio
import json
from datetime import datetime
from typing import Dict, List
from database import (
    fetch_unprocessed_team_articles, 
    mark_team_article_as_processed, 
    insert_processed_team_article
)
from englishArticle import generate_english_article
from germanArticle import generate_german_article
from articleImage import ImageSearcher

async def process_single_team_article(article: Dict, image_searcher: ImageSearcher, debug: bool = False) -> Dict:
    """
    Process a single team article through the complete pipeline:
    1. Generate English version
    2. Generate German version
    3. Search for relevant images
    4. Validate all required content is present
    5. Save to TeamNewsArticles database if valid (unless debug=True)
    6. Mark source article as processed only if saved successfully
    """
    article_id = article['id']
    
    # Check for content in multiple possible field names
    main_content = article.get('Content') or article.get('content') or article.get('ContentOriginal') or article.get('contentOriginal') or ''
    
    if not main_content:
        print(f"WARNING: Article ID {article_id} has no content. Available fields: {list(article.keys())}")
    
    # Get the source field which links to TeamNewsSource
    source_id = article.get('source')
    if source_id:
        print(f"Article is from TeamNewsSource ID: {source_id}")
    else:
        print("WARNING: Article has no source ID to link to TeamNewsSource")
    
    isArticleCreated = False
    
    print(f"\n{'='*80}")
    print(f"Processing Team Article ID: {article_id}")
    print(f"Original Headline: {article.get('headline', '')}")
    print(f"Content Length: {len(main_content)} characters")
    print(f"{'='*80}\n")
    
    try:
        # Step 1: Generate English Article
        print("Step 1: Generating English Article...")
        english_result = await generate_english_article(main_content, verbose=True)
        if not english_result:
            raise Exception("Failed to generate English article")
        english_headline = english_result.get('headline', '').replace('<h1>', '').replace('</h1>', '')
        english_content = english_result.get('content', '')
        
        print(f"\nEnglish Article Generated:")
        print(f"Headline: {english_headline}")
        print(f"Content Length: {len(english_content)}")

        # Step 2: Generate German Article
        print("\nStep 2: Generating German Article...")
        german_result = await generate_german_article(
            english_result.get('headline', ''),
            english_result.get('content', '')
        )
        if not german_result:
            raise Exception("Failed to generate German article")
        german_headline = german_result.get('headline', '').replace('<h1>', '').replace('</h1>', '')
        german_content = german_result.get('content', '')
        
        print(f"\nGerman Article Generated:")
        print(f"Headline: {german_headline}")
        print(f"Content Length: {len(german_content)}")

        # Step 3: Search for Images using the enhanced LLM-based approach
        print("\nStep 3: Searching for Images using LLM optimization...")
        # Use the full article content for better context instead of just the headline
        full_content = english_content + " " + english_headline
        
        # Pass the content to search_images for improved ranking
        images = await image_searcher.search_images(full_content, content=english_content)
        
        if not images:
            print("Warning: LLM-based image search returned no results, trying fallback with headline only...")
            # Fallback to using just the headline if full content search fails
            images = await image_searcher.search_images(english_headline, content=english_headline)
            
        if not images:
            print("Warning: No images found after fallback attempt")
            images = []
        
        print(f"Found {len(images)} images")
        for idx, image in enumerate(images, 1):
            print(f"Image {idx}:")
            print(f"- Title: {image['title']}")
            print(f"- URL: {image['url']}")
            print(f"- Dimensions: {image['width']}x{image['height']}")

        # Step 4: Validate that all required content is present
        print("\nStep 4: Validating article content...")
        image1_url = images[0]['url'] if len(images) > 0 else ""
        
        # Extract summaries and clean HTML tags
        english_summary = english_result.get('summary', '').replace('<p>', '').replace('</p>', '')
        german_summary = german_result.get('summary', '').replace('<p>', '').replace('</p>', '')
        
        # Check if any required content is missing
        if not english_headline or not german_headline or not english_content or not german_content or not image1_url or not english_summary or not german_summary:
            missing_parts = []
            if not english_headline: missing_parts.append("English headline")
            if not german_headline: missing_parts.append("German headline")
            if not english_summary: missing_parts.append("English summary")
            if not german_summary: missing_parts.append("German summary")
            if not english_content: missing_parts.append("English content")
            if not german_content: missing_parts.append("German content") 
            if not image1_url: missing_parts.append("Image 1")
            
            print(f"Error: Article is missing required content: {', '.join(missing_parts)}")
            print("Article will not be saved to database")
            
            # Compile results for return but don't save to database
            result = {
                'article_id': article_id,
                'original': {
                    'headline': article.get('headline', ''),
                    'source': article.get('source'),
                    'url': article.get('url', '')
                },
                'english': {
                    'headline': english_headline,
                    'summary': english_summary,
                    'content': english_content
                },
                'german': {
                    'headline': german_headline,
                    'summary': german_summary,
                    'content': german_content
                },
                'images': images,
                'isArticleCreated': isArticleCreated
            }
            
            print(f"\n{'-'*80}")
            print("Team Article Processing Complete - NOT SAVED (missing content)")
            print(f"{'-'*80}\n")
            
            return result

        # Create article data dict - Fixed field names to match database expectations
        db_article = {
            "id": article_id,  # Include original id for marking as processed
            "created_at": datetime.utcnow().isoformat(),
            "headlineEnglish": english_headline,
            "headlineGerman": german_headline,
            "summaryEnglish": english_summary,
            "summaryGerman": german_summary,
            "contentEnglish": english_content,
            "contentGerman": german_content,
            "image1": image1_url,
            "image2": images[1]['url'] if len(images) > 1 else "",
            "image3": images[2]['url'] if len(images) > 2 else "",
            "sourceArticle": article_id,
            "source": source_id,  # Add the source ID from TeamSourceArticles to look up the team
            "team": article.get('team'),  # Use the team ID directly if available
            "status": "NEW"
        }
        
        # Step 5: Insert into TeamNewsArticles database or show debug output
        if debug:
            print("\nDEBUG MODE: Article would be saved with the following data:")
            print(json.dumps(db_article, indent=2))
            isArticleCreated = True
        else:
            print("\nStep 5: Inserting article into TeamNewsArticles database...")
            success = await insert_processed_team_article(db_article)
            if not success:
                raise Exception("Failed to insert article into TeamNewsArticles database")
            
            # Article was successfully created and saved
            isArticleCreated = True
            
            # Step 6: Only mark the source article as processed if successfully saved
            print("\nStep 6: Marking source article as processed...")
            success = await mark_team_article_as_processed(article_id)
            if not success:
                raise Exception("Failed to mark team source article as processed")

        # Compile all results for output
        result = {
            'article_id': article_id,
            'original': {
                'headline': article.get('headline', ''),
                'source': article.get('source'),
                'url': article.get('url', '')
            },
            'english': english_result,
            'german': german_result,
            'images': images,
            'isArticleCreated': isArticleCreated,
            'debug_mode': debug
        }
        
        print(f"\n{'-'*80}")
        if debug:
            print("Team Article Processing Complete - DEBUG MODE (not saved to database)")
        else:
            print("Team Article Processing Complete - Successfully saved to database")
        print(f"{'-'*80}\n")
        
        return result
    except Exception as e:
        print(f"Error processing team article {article_id}: {str(e)}")
        return {
            'article_id': article_id,
            'error': str(e),
            'isArticleCreated': isArticleCreated,
            'debug_mode': debug
        }

async def run_team_pipeline(debug: bool = False):
    """
    Main team pipeline function that:
    1. Fetches team articles to process
    2. Processes each article sequentially through all steps
    3. Reports processing results
    
    Args:
        debug (bool): If True, results will be displayed in the terminal
                      instead of being saved to the database
    """
    print("Starting Team Article Processing Pipeline...")
    if debug:
        print("DEBUG MODE ENABLED: Results will be displayed but not saved to database")
    
    # Initialize image searcher with LLM-based search enabled
    image_searcher = ImageSearcher(use_llm=True)
    
    # Fetch unprocessed team articles
    articles = await fetch_unprocessed_team_articles()
    print(f"Found {len(articles)} team articles to process")
    
    # Print the first article structure to debug field names
    if articles and debug:
        print(f"\nSample article structure from TeamSourceArticles:")
        print(f"Available fields: {list(articles[0].keys())}")
        for key, value in articles[0].items():
            value_preview = str(value)[:100] + "..." if isinstance(value, str) and len(value) > 100 else value
            print(f"- {key}: {value_preview}")
        print("\n")
    
    # Process each article sequentially
    processed_count = 0
    saved_count = 0
    failed_count = 0
    
    for article in articles:
        try:
            result = await process_single_team_article(article, image_searcher, debug=debug)
            processed_count += 1
            
            if result and result.get('isArticleCreated', False):
                saved_count += 1
            else:
                failed_count += 1
            
            # Add a small delay between articles to avoid rate limits
            await asyncio.sleep(2)
            
        except Exception as e:
            print(f"Error processing team article {article.get('id')}: {e}")
            failed_count += 1
    
    # Report final processing statistics
    print(f"\nTeam Pipeline complete!")
    print(f"Team articles processed: {processed_count}")
    if debug:
        print(f"Team articles processed in DEBUG mode (not saved): {saved_count}")
    else:
        print(f"Team articles saved to database: {saved_count}")
    print(f"Team articles failed or rejected: {failed_count}")

if __name__ == "__main__":
    # Set debug=True to display results in terminal instead of saving to database
    # Set debug=False for normal operation that saves to database
    asyncio.run(run_team_pipeline(debug=False))