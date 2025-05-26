import asyncio
import logging
import os
from typing import Union
from uuid import UUID

import yaml
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Assuming database.py, LLMSetup.py, and post_processing.py are in the parent directory
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from database import fetch_all_cluster_ids, fetch_source_articles_for_cluster, save_timeline_to_database
from LLMSetup import initialize_model as llm_setup_initialize_model
from post_processing import remove_citations_from_text

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    force=True
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# --- Constants ---
PROMPTS_FILE_PATH = os.path.join(os.path.dirname(__file__), '..', 'prompts.yml')

# --- LLM and Prompts ---
llm_model_info = {}
prompts = {}


def load_prompts():
    """Load required prompts from the prompts.yml file"""
    global prompts
    try:
        with open(PROMPTS_FILE_PATH, 'r') as f:
            prompts = yaml.safe_load(f)
        
        if 'summarize_article_prompt' not in prompts:
            logger.error("Required 'summarize_article_prompt' not found in prompts.yml")
            raise ValueError("Required prompt not found: summarize_article_prompt")
            
        logger.info("Prompts loaded successfully for story line generator.")
    except FileNotFoundError:
        logger.error(f"Prompts file not found at {PROMPTS_FILE_PATH}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML in prompts file: {e}")
        raise
    except ValueError as e:
        logger.error(str(e))
        raise


async def initialize_llm(model_type="default"):
    """Initialize the LLM model for summarization"""
    global llm_model_info
    try:
        # Use LLMSetup.py to get the model
        model_info = llm_setup_initialize_model(provider="gemini", model_type=model_type, grounding_enabled=False)
        
        llm_model_info["model_name"] = model_info["model_name"]
        llm_model_info["model"] = model_info["model"]  # This is the client.models object
        llm_model_info["tools"] = model_info["tools"]
        llm_model_info["grounding_enabled"] = model_info["grounding_enabled"]
        
        logger.info(f"LLM ready for story line generator: {llm_model_info['model_name']} (using LLMSetup pattern)")
        return llm_model_info
    except Exception as e:
        logger.error(f"Failed to initialize LLM in story_line_generator: {e}", exc_info=True)
        raise


async def summarize_article_with_llm(headline: str, content: str) -> str:
    """Generate a summary for a news article using the LLM"""
    # Check if essential parts of llm_model_info are initialized
    if not llm_model_info or \
       "model_name" not in llm_model_info or \
       "model" not in llm_model_info:
        logger.error("LLM (model_name or model) is not initialized in llm_model_info.")
        raise ConnectionError("LLM model info not available for summarization.")

    if not headline or not content or content == "N/A":
        logger.warning(f"Headline or content is empty or N/A for headline: '{headline}', returning 'Content not available for summarization.'")
        return "Content not available for summarization."

    prompt_template = prompts.get('summarize_article_prompt')
    if not prompt_template:
        logger.error("Summarize prompt template is not loaded.")
        return "Error: Summarization prompt not found."

    formatted_prompt = prompt_template.format(headline=headline, content=content)

    try:
        # Use the same pattern as englishArticle.py
        response = await asyncio.to_thread(
            llm_model_info["model"].generate_content,
            model=llm_model_info["model_name"],
            contents=formatted_prompt,
            config=types.GenerateContentConfig(
                temperature=0.4,
                max_output_tokens=2048,
                tools=llm_model_info.get("tools", [])
            )
        )

        if response and hasattr(response, 'text') and response.text:
            summary = response.text.strip()
            # Post-process to remove "Summary: " prefix if present
            if summary.lower().startswith("summary:"):
                summary = summary[len("summary:"):].strip()
            return remove_citations_from_text(summary)
        elif hasattr(response, 'parts') and response.parts:
            # Handle cases where response might be structured with parts
            full_text = "".join(part.text for part in response.parts if hasattr(part, 'text'))
            summary = full_text.strip()
            if summary.lower().startswith("summary:"):
                summary = summary[len("summary:"):].strip()
            return remove_citations_from_text(summary)
        else:
            logger.warning(f"LLM response for summarization was empty or in unexpected format for headline: {headline}")
            return "Summary could not be generated."
    except Exception as e:
        logger.error(f"Error during LLM summarization for headline '{headline}': {e}")
        return f"Error generating summary: {e}"


async def generate_timeline_for_cluster(cluster_id: Union[str, UUID]) -> tuple:
    """
    Process a cluster to generate a story line timeline
    
    Returns:
        tuple: (timeline_id, timeline_json)
    """
    logger.info(f"Processing cluster: {cluster_id}")
    source_articles = await fetch_source_articles_for_cluster(cluster_id)

    if not source_articles:
        logger.warning(f"No source articles found for cluster {cluster_id}. Skipping.")
        return None, None

    timeline_entries = []
    for article in source_articles:
        headline = article.get('headline', 'N/A')
        content = article.get('Content', 'N/A')  # Capital C as per database schema
        created_at = article.get('created_at', 'N/A')
        
        # Extract source name from nested NewsSource object
        source_name = 'Unknown'
        if article.get('NewsSource') and isinstance(article['NewsSource'], dict):
            source_name = article['NewsSource'].get('Name', 'Unknown')

        # Ensure created_at is a string for JSON serialization
        if not isinstance(created_at, str):
            created_at = str(created_at)

        summary = await summarize_article_with_llm(headline, content)

        timeline_entries.append({
            "created_at": created_at,
            "headline": headline,
            "source_name": source_name,
            "summary": summary
        })

    # Create the timeline JSON structure
    timeline_json = {
        "cluster_id": str(cluster_id),
        "timeline": timeline_entries
    }

    # Save timeline to database
    timeline_id = await save_timeline_to_database(cluster_id, timeline_entries)
    
    if not timeline_id:
        logger.error(f"Failed to save timeline for cluster {cluster_id} to database")
        return None, None
    
    logger.info(f"Timeline for cluster {cluster_id} saved to database with ID: {timeline_id}")
    return timeline_id, timeline_json


async def main():
    """Main function to process all clusters and generate story lines"""
    try:
        load_prompts()
        await initialize_llm(model_type="default") # Use default model for generation
    except Exception as e:
        logger.critical(f"Initialization failed: {e}")
        return

    cluster_ids_list = await fetch_all_cluster_ids() 

    if not cluster_ids_list:
        logger.info("No cluster IDs found to process.")
        return

    logger.info(f"Found {len(cluster_ids_list)} clusters to process.")

    # Process clusters sequentially
    results = []
    for cluster_id in cluster_ids_list:
        if cluster_id:
            result = await generate_timeline_for_cluster(cluster_id)
            if result[0]:  # If timeline_id is not None
                results.append(result)
        else:
            logger.warning("Encountered an empty or None cluster_id.")
            
    logger.info(f"Processed {len(results)} clusters successfully.")
    return results


if __name__ == "__main__":
    asyncio.run(main())
