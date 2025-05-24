"""
Article fetcher module for deep dive analysis system.
Provides reusable functions to fetch both source articles and cluster article content.
"""

import logging
from typing import List, Dict, Optional, Union, Tuple
from uuid import UUID
from database import fetch_source_articles_for_cluster, get_existing_cluster_article

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    force=True 
)
logger = logging.getLogger(__name__)


async def fetch_complete_cluster_data(cluster_id: Union[str, UUID]) -> Optional[Dict]:
    """
    Fetches complete cluster data including both source articles and cluster article content.
    
    Args:
        cluster_id: The cluster ID to fetch data for
        
    Returns:
        Dict containing:
        - cluster_id: The cluster ID
        - source_articles: List of source articles with headline, content, created_at, source name
        - cluster_article: Dict with cluster article headline, summary, content (None if not exists)
        - article_count: Number of source articles
    """
    try:
        logger.info(f"Fetching complete cluster data for cluster {cluster_id}")
        
        # Fetch source articles
        source_articles = await fetch_source_articles_for_cluster(cluster_id)
        if not source_articles:
            logger.warning(f"No source articles found for cluster {cluster_id}")
            return None
            
        # Fetch cluster article content
        cluster_article = await get_existing_cluster_article(cluster_id)
        
        result = {
            "cluster_id": str(cluster_id),
            "source_articles": source_articles,
            "cluster_article": cluster_article,
            "article_count": len(source_articles)
        }
        
        logger.info(f"Successfully fetched data for cluster {cluster_id}: {len(source_articles)} source articles, "
                   f"cluster article {'found' if cluster_article else 'not found'}")
        
        return result
        
    except Exception as e:
        logger.error(f"Error fetching complete cluster data for {cluster_id}: {e}", exc_info=True)
        return None


def format_source_articles_for_analysis(source_articles: List[Dict]) -> str:
    """
    Formats source articles for analysis prompts.
    
    Args:
        source_articles: List of source article dictionaries
        
    Returns:
        Formatted string with source articles
    """
    if not source_articles:
        return "No source articles available."
    
    formatted_articles = []
    for i, article in enumerate(source_articles, 1):
        source_name = ""
        if article.get('NewsSource') and isinstance(article['NewsSource'], dict):
            source_name = article['NewsSource'].get('Name', 'Unknown Source')
        elif article.get('NewsSource') and isinstance(article['NewsSource'], str):
            source_name = article['NewsSource']
        else:
            source_name = 'Unknown Source'
            
        headline = article.get('headline', 'No headline')
        content = article.get('Content', 'No content')
        created_at = article.get('created_at', 'Unknown date')
        
        formatted_article = (
            f"--- Source Article {i} ---\n"
            f"Source: {source_name}\n"
            f"Date: {created_at}\n"
            f"Headline: {headline}\n"
            f"Content: {content}\n"
            f"--- End Source Article {i} ---"
        )
        formatted_articles.append(formatted_article)
    
    return "\n\n".join(formatted_articles)


def extract_cluster_article_content(cluster_article: Optional[Dict]) -> Tuple[str, str, str]:
    """
    Extracts headline, summary, and content from cluster article data.
    
    Args:
        cluster_article: Cluster article dictionary or None
        
    Returns:
        Tuple of (headline, summary, content) - empty strings if not available
    """
    if not cluster_article:
        return "", "", ""
    
    # Clean HTML tags for analysis (keep content but remove HTML formatting)
    headline = cluster_article.get('headline', '')
    summary = cluster_article.get('summary', '') 
    content = cluster_article.get('content', '')
    
    # Remove HTML tags but keep the text content
    import re
    
    def clean_html(text: str) -> str:
        if not text:
            return ""
        # Remove HTML tags but keep the content
        clean_text = re.sub(r'<[^>]+>', '', text)
        # Clean up extra whitespace
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
        return clean_text
    
    return clean_html(headline), clean_html(summary), clean_html(content)


def validate_cluster_data_for_analysis(cluster_data: Dict) -> bool:
    """
    Validates that cluster data contains sufficient content for analysis.
    
    Args:
        cluster_data: Complete cluster data dictionary
        
    Returns:
        True if data is sufficient for analysis, False otherwise
    """
    if not cluster_data:
        logger.warning("Cluster data is None")
        return False
        
    source_articles = cluster_data.get('source_articles', [])
    cluster_article = cluster_data.get('cluster_article')
    
    # Need at least one source article
    if not source_articles:
        logger.warning("No source articles available for analysis")
        return False
    
    # Check if we have meaningful content in source articles
    has_content = False
    for article in source_articles:
        if article.get('Content') and len(article['Content'].strip()) > 50:
            has_content = True
            break
    
    if not has_content:
        logger.warning("Source articles don't contain sufficient content for analysis")
        return False
    
    # Cluster article is optional but preferred
    if not cluster_article:
        logger.info("No cluster article available, will analyze only source articles")
    
    return True


async def fetch_multiple_clusters_data(cluster_ids: List[Union[str, UUID]]) -> List[Dict]:
    """
    Fetches complete data for multiple clusters.
    
    Args:
        cluster_ids: List of cluster IDs to fetch
        
    Returns:
        List of cluster data dictionaries (excluding None results)
    """
    results = []
    
    for cluster_id in cluster_ids:
        try:
            cluster_data = await fetch_complete_cluster_data(cluster_id)
            if cluster_data and validate_cluster_data_for_analysis(cluster_data):
                results.append(cluster_data)
            else:
                logger.warning(f"Skipping cluster {cluster_id} - insufficient data for analysis")
        except Exception as e:
            logger.error(f"Error processing cluster {cluster_id}: {e}")
            continue
    
    logger.info(f"Successfully fetched data for {len(results)} out of {len(cluster_ids)} clusters")
    return results


if __name__ == "__main__":
    import asyncio
    
    async def test_article_fetcher():
        """Test the article fetcher functionality"""
        print("Testing Article Fetcher...")
        
        # This would need a real cluster_id for testing
        # For now, just test the formatting functions
        
        mock_source_articles = [
            {
                'headline': 'Test Headline 1',
                'Content': 'Test content for article 1',
                'created_at': '2023-01-01T10:00:00Z',
                'NewsSource': {'Name': 'Test Source 1'}
            },
            {
                'headline': 'Test Headline 2', 
                'Content': 'Test content for article 2',
                'created_at': '2023-01-02T10:00:00Z',
                'NewsSource': {'Name': 'Test Source 2'}
            }
        ]
        
        formatted = format_source_articles_for_analysis(mock_source_articles)
        print("Formatted source articles:")
        print(formatted[:200] + "...")
        
        mock_cluster_article = {
            'headline': '<h1>Test Cluster Headline</h1>',
            'summary': '<p>Test cluster summary</p>',
            'content': '<div><p>Test cluster content</p></div>'
        }
        
        headline, summary, content = extract_cluster_article_content(mock_cluster_article)
        print(f"\nExtracted cluster content:")
        print(f"Headline: {headline}")
        print(f"Summary: {summary}")
        print(f"Content: {content}")
        
        print("\nArticle fetcher test completed successfully!")
    
    asyncio.run(test_article_fetcher())
