import sys
import os
import asyncio
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import supabase

async def batch_update_updatedby() -> dict:
    """
    Batch processes all existing articles in NewsArticles table to update their UpdatedBy field
    based on ArticleVector table data.
    
    Returns:
        Dict: Statistics about the operation (total processed, updated count, errors)
    """
    try:
        stats = {"total": 0, "processed": 0, "updated": 0, "errors": 0}
        
        # Fetch all articles from NewsArticles table
        response = supabase.table("NewsArticles").select("id,SourceArticle").execute()
        
        if not response.data:
            print("No articles found in NewsArticles table")
            return stats
            
        articles = response.data
        stats["total"] = len(articles)
        print(f"Processing {len(articles)} articles...")
        
        # Process each article
        for article in articles:
            try:
                stats["processed"] += 1
                article_id = article["id"]
                source_article_id = article["SourceArticle"]
                
                # Check ArticleVector table for updates
                vector_response = supabase.table("ArticleVector").select("update").eq("SourceArticle", source_article_id).execute()
                
                if vector_response.data and vector_response.data[0].get("update"):
                    updated_articles = vector_response.data[0]["update"]
                    if updated_articles:
                        # This article updates other articles - need to update their UpdatedBy field
                        for updated_source_id in updated_articles:
                            update_response = supabase.table("NewsArticles").update(
                                {"UpdatedBy": article_id}
                            ).eq("SourceArticle", updated_source_id).execute()
                            
                            if update_response.data:
                                stats["updated"] += 1
                                print(f"Updated article with SourceArticle {updated_source_id} to be updated by {article_id}")
                            else:
                                print(f"Warning: Could not update UpdatedBy for article with SourceArticle {updated_source_id}")
                
                if stats["processed"] % 10 == 0:  # Progress update every 10 articles
                    print(f"Progress: {stats['processed']}/{stats['total']} articles processed")
                    
            except Exception as e:
                print(f"Error processing article {article.get('id', 'unknown')}: {e}")
                stats["errors"] += 1
                
        print("\nBatch processing complete!")
        print(f"Total articles processed: {stats['processed']}")
        print(f"Articles updated: {stats['updated']}")
        print(f"Errors encountered: {stats['errors']}")
        return stats
        
    except Exception as e:
        print(f"Error in batch processing: {e}")
        return {"total": 0, "processed": 0, "updated": 0, "errors": 1}

async def main():
    """Main entry point for the batch update process"""
    print("Starting batch update of UpdatedBy field...")
    stats = await batch_update_updatedby()
    
    print("\nFinal Summary:")
    print(f"Total articles: {stats['total']}")
    print(f"Articles processed: {stats['processed']}")
    print(f"Articles updated: {stats['updated']}")
    print(f"Errors encountered: {stats['errors']}")

if __name__ == "__main__":
    asyncio.run(main())