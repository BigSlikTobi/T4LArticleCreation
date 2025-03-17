import asyncio
import json
from datetime import datetime
from typing import Dict, List
from database import fetch_unprocessed_articles, mark_article_as_processed, insert_processed_article
from englishArticle import generate_english_article
from germanArticle import generate_german_article
from articleImage import ImageSearcher

async def process_single_article(article: Dict, image_searcher: ImageSearcher) -> Dict:
    """
    Process a single article through the complete pipeline:
    1. Generate English version
    2. Generate German version
    3. Search for relevant images
    4. Validate all required content is present
    5. Save to NewsArticles database if valid
    6. Mark source article as processed only if saved successfully
    """
    article_id = article['id']
    main_content = article.get('Content', '')
    isArticleCreated = False
    
    print(f"\n{'='*80}")
    print(f"Processing Article ID: {article_id}")
    print(f"Original Headline: {article.get('headline', '')}")
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
        
        # Check if any required content is missing
        if not english_headline or not german_headline or not english_content or not german_content or not image1_url:
            missing_parts = []
            if not english_headline: missing_parts.append("English headline")
            if not german_headline: missing_parts.append("German headline")
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
                'english': english_result,
                'german': german_result,
                'images': images,
                'isArticleCreated': isArticleCreated
            }
            
            print(f"\n{'-'*80}")
            print("Article Processing Complete - NOT SAVED (missing content)")
            print(f"{'-'*80}\n")
            
            return result

        # Step 5: Insert into NewsArticles database if all required content is present
        db_article = {
            "created_at": datetime.utcnow().isoformat(),
            "headlineEnglish": english_headline,
            "headlineGerman": german_headline,
            "ContentEnglish": english_content,
            "ContentGerman": german_content,
            "Image1": image1_url,
            "Image2": images[1]['url'] if len(images) > 1 else "",
            "Image3": images[2]['url'] if len(images) > 2 else "",
            "SourceArticle": article_id
        }
        
        success = await insert_processed_article(db_article)
        if not success:
            raise Exception("Failed to insert article into NewsArticles database")
        
        # Article was successfully created and saved
        isArticleCreated = True
        
        # Step 6: Only mark the source article as processed if successfully saved
        success = await mark_article_as_processed(article_id)
        if not success:
            raise Exception("Failed to mark source article as processed")

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
            'isArticleCreated': isArticleCreated
        }

        print(f"\n{'-'*80}")
        print("Article Processing Complete - Successfully saved to database")
        print(f"{'-'*80}\n")
        
        return result
    except Exception as e:
        print(f"Error processing article {article_id}: {str(e)}")
        return {
            'article_id': article_id,
            'error': str(e),
            'isArticleCreated': isArticleCreated
        }

async def run_pipeline():
    """
    Main pipeline function that:
    1. Fetches articles to process
    2. Processes each article sequentially through all steps
    3. Reports processing results
    """
    print("Starting Article Processing Pipeline...")
    
    # Initialize image searcher with LLM-based search enabled
    image_searcher = ImageSearcher(use_llm=True)
    
    # Fetch unprocessed articles using centralized database logic
    articles = await fetch_unprocessed_articles()
    print(f"Found {len(articles)} articles to process")
    
    # Process each article sequentially
    processed_count = 0
    saved_count = 0
    failed_count = 0
    
    for article in articles:
        try:
            result = await process_single_article(article, image_searcher)
            processed_count += 1
            
            if result and result.get('isArticleCreated', False):
                saved_count += 1
            else:
                failed_count += 1
            
            # Add a small delay between articles to avoid rate limits
            await asyncio.sleep(2)
            
        except Exception as e:
            print(f"Error processing article {article.get('id')}: {e}")
            failed_count += 1
    
    # Report final processing statistics
    print(f"\nPipeline complete!")
    print(f"Articles processed: {processed_count}")
    print(f"Articles saved to database: {saved_count}")
    print(f"Articles failed or rejected: {failed_count}")

if __name__ == "__main__":
    asyncio.run(run_pipeline())