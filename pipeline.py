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
    4. Save to NewsArticles database
    5. Mark source article as processed
    """
    article_id = article['id']
    main_content = article.get('Content', '')
    
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
        
        print(f"\nEnglish Article Generated:")
        print(f"Headline: {english_headline}")
        print(f"Content Length: {len(english_result.get('content', ''))}")

        # Step 2: Generate German Article
        print("\nStep 2: Generating German Article...")
        german_result = await generate_german_article(
            english_result.get('headline', ''),
            english_result.get('content', '')
        )
        if not german_result:
            raise Exception("Failed to generate German article")
        german_headline = german_result.get('headline', '').replace('<h1>', '').replace('</h1>', '')
        
        print(f"\nGerman Article Generated:")
        print(f"Headline: {german_headline}")
        print(f"Content Length: {len(german_result.get('content', ''))}")

        # Step 3: Search for Images using the enhanced LLM-based approach
        print("\nStep 3: Searching for Images using LLM optimization...")
        # Use the full article content for better context instead of just the headline
        full_content = english_result.get('content', '') + " " + english_headline
        images = await image_searcher.search_images(full_content)
        
        if not images:
            print("Warning: LLM-based image search returned no results, trying fallback with headline only...")
            # Fallback to using just the headline if full content search fails
            images = await image_searcher.search_images(english_headline)
            
        if not images:
            print("Warning: No images found after fallback attempt")
            images = []
        
        print(f"Found {len(images)} images")
        for idx, image in enumerate(images, 1):
            print(f"Image {idx}:")
            print(f"- Title: {image['title']}")
            print(f"- URL: {image['url']}")
            print(f"- Dimensions: {image['width']}x{image['height']}")

        # Step 4: Insert into NewsArticles database
        db_article = {
            "created_at": datetime.utcnow().isoformat(),
            "headlineEnglish": english_headline,
            "headlineGerman": german_headline,
            "ContentEnglish": english_result.get('content', ''),
            "ContentGerman": german_result.get('content', ''),
            "Image1": images[0]['url'] if len(images) > 0 else "",
            "Image2": images[1]['url'] if len(images) > 1 else "",
            "Image3": images[2]['url'] if len(images) > 2 else "",
            "SourceArticle": article_id
        }
        
        success = await insert_processed_article(db_article)
        if not success:
            raise Exception("Failed to insert article into NewsArticles database")

        # Step 5: Only mark the source article as processed if all previous steps succeeded
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
            'images': images
        }

        print(f"\n{'-'*80}")
        print("Article Processing Complete!")
        print(f"{'-'*80}\n")
        
        return result
    except Exception as e:
        print(f"Error processing article {article_id}: {str(e)}")
        return None

async def run_pipeline():
    """
    Main pipeline function that:
    1. Fetches articles to process
    2. Processes each article sequentially through all steps
    3. Saves the results
    """
    print("Starting Article Processing Pipeline...")
    
    # Initialize image searcher with LLM-based search enabled
    image_searcher = ImageSearcher(use_llm=True)
    
    # Fetch unprocessed articles using centralized database logic
    articles = await fetch_unprocessed_articles()
    print(f"Found {len(articles)} articles to process")
    
    # Process each article sequentially
    results = {}
    for article in articles:
        try:
            result = await process_single_article(article, image_searcher)
            if result:  # Only save results if processing was successful
                results[str(result['article_id'])] = result
            
            # Add a small delay between articles to avoid rate limits
            await asyncio.sleep(2)
            
        except Exception as e:
            print(f"Error processing article {article.get('id')}: {e}")
    
    # Save all results
    output_file = "processed_articles.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\nPipeline complete! Processed {len(results)} articles")
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    asyncio.run(run_pipeline())