import asyncio
import json  # Added import
from datetime import datetime
from typing import Dict, List
from database import fetch_unprocessed_articles, mark_article_as_processed, insert_processed_article, fetch_teams, save_article_image_metadata
from englishArticle import generate_english_article
from germanArticle import generate_german_article
from articleImage import ImageSearcher
from team_classifier import classify_team  # Changed import to use simple classifier

async def process_single_article(article: Dict, image_searcher: ImageSearcher, teams_data: List[Dict]) -> Dict:
    """
    Process a single article through the complete pipeline:
        1. Generate English version
        2. Generate German version
        3. Search for relevant images
        4. Classify article by team
        5. Validate all required content is present
        6. Save to NewsArticles database if valid
        7. Mark source article as processed only if saved successfully

    Args:
        article (Dict): The source article record to process.
        image_searcher (ImageSearcher): The image searcher instance for finding images.
        teams_data (List[Dict]): List of team records for classification.

    Returns:
        Dict: A dictionary with processing results, including article content, images, and status flags.
    """
    article_id = article['id']
    
    # Create a structured main_content with headline and content
    main_content = {
        "headline": article.get('headline', ''),
        "content": article.get('Content', '')
    }
    
    # Convert to JSON string
    main_content = json.dumps(main_content)
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
        # Create structured main_content for German translation
        german_input = {
            "headline": english_result.get('headline', ''),
            "content": english_result.get('content', '')
        }
        german_result = await generate_german_article(json.dumps(german_input))
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

        # Step 4: Classify article by team
        print("\nStep 4: Classifying article by team...")
        team_name, confidence = await classify_team(english_headline, english_content)
        team_assignment = None
        team_id = None
        
        if team_name and confidence > 0.8 and team_name.lower() != "unknown":
            # Find matching team in database
            for team in teams_data:
                db_team_name = team.get('fullName', '').lower()
                if db_team_name and (
                    team_name.lower() in db_team_name or 
                    db_team_name in team_name.lower()
                ):
                    team_id = team['id']
                    team_assignment = team_id
                    print(f"Article classified as about team: {team.get('fullName')} (ID: {team_id}) with confidence {confidence:.2f}")
                    break
            
            if not team_id:
                print(f"Could not match team '{team_name}' to any team in database")
        else:
            print(f"Team classification confidence too low ({confidence:.2f}) or no team matched")

        # Step 5: Validate that all required content is present
        print("\nStep 5: Validating article content...")
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
                'isArticleCreated': isArticleCreated,
                'team_classification': {
                    'team_id': team_id,
                    'confidence': confidence
                }
            }
            
            print(f"\n{'-'*80}")
            print("Article Processing Complete - NOT SAVED (missing content)")
            print(f"{'-'*80}\n")
            
            return result

        # Step 6: Insert into NewsArticles database if all required content is present
        db_article = {
            "created_at": datetime.utcnow().isoformat(),
            "headlineEnglish": english_headline,
            "headlineGerman": german_headline,
            "SummaryEnglish": english_summary,
            "SummaryGerman": german_summary,
            "ContentEnglish": english_content,
            "ContentGerman": german_content,
            "Image1": image1_url,
            "Image2": images[1]['url'] if len(images) > 1 else "",
            "Image3": images[2]['url'] if len(images) > 2 else "",
            "SourceArticle": article_id,
            "team": team_assignment,
            "status": "NEW"
        }
        
        # Insert the article into the database
        new_article_id = await insert_processed_article(db_article)
        if not new_article_id:
            raise Exception("Failed to insert article into NewsArticles database")
        
        # Article was successfully created and saved
        isArticleCreated = True
        
        # Save image metadata to article_image table
        for idx, image in enumerate(images[:3]):
            if isinstance(image, dict) and 'url' in image: # Added check for robustness
                await save_article_image_metadata(
                    article_id=new_article_id,
                    image_url=image['url'], # This is the Supabase URL
                    original_url=image.get('original_url', image['url']), # URL from the web
                    author=image.get('author'),
                    source=image.get('page_url') # This should be the website URL (e.g., nfl.com)
                )
            else:
                print(f"Warning: Skipping invalid image data structure for article_id {new_article_id}. Image data: {image}")
        
        # Step 7: Only mark the source article as processed if successfully saved
        success = await mark_article_as_processed(article_id, article_created=isArticleCreated)
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
            'isArticleCreated': isArticleCreated,
            'team_classification': {
                'team_id': team_id,
                'confidence': confidence,
                'assigned': team_assignment is not None
            }
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

    Returns:
        None
    """
    print("Starting Article Processing Pipeline...")
    
    # Initialize image searcher with LLM-based search enabled
    image_searcher = ImageSearcher(use_llm=True)
    
    # Fetch unprocessed articles using centralized database logic
    articles = await fetch_unprocessed_articles()
    print(f"Found {len(articles)} articles to process")
    
    # Fetch teams data for classification
    teams_data = await fetch_teams()
    if not teams_data:
        print("Warning: No teams data found for classification. Team assignment will be skipped.")
    else:
        print(f"Loaded {len(teams_data)} teams for classification")
    
    # Process each article sequentially
    processed_count = 0
    saved_count = 0
    failed_count = 0
    team_assigned_count = 0
    
    for article in articles:
        try:
            result = await process_single_article(article, image_searcher, teams_data)
            processed_count += 1
            
            if result and result.get('isArticleCreated', False):
                saved_count += 1
                if result.get('team_classification', {}).get('assigned', False):
                    team_assigned_count += 1
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
    print(f"Articles with team assigned: {team_assigned_count}")
    print(f"Articles failed or rejected: {failed_count}")

if __name__ == "__main__":
    asyncio.run(run_pipeline())