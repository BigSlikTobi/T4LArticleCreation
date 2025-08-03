#!/usr/bin/env python3
"""
Personal Summary Generator for Sprint 3, Epic 1, Task 1

This script implements the personalized content engine that generates rolling summaries 
for each user's specific preferences. For each user and each of their preferred entities 
(players/teams), it gathers new stats and articles since the last update and generates 
a personalized rolling summary using the LLM.

Key Features:
- Loops through all users and their preferences
- Determines "new" content since last update for each entity
- Uses rolling summary prompt to provide context-aware updates
- Saves results to generated_updates table with proper entity tracking
"""

import asyncio
import logging
import os
import sys
import yaml
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple
from google.genai import types

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from LLMSetup import initialize_model
from database import (
    fetch_all_users, fetch_user_preferences, get_last_update_timestamp,
    fetch_new_articles_for_entity, fetch_new_stats_for_player,
    get_previous_summary_for_entity, save_generated_update,
    _check_supabase_client
)

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    force=True
)
logger = logging.getLogger(__name__)

# --- Constants ---
PROMPTS_FILE_PATH = os.path.join(os.path.dirname(__file__), "prompts.yml")
DEFAULT_LOOKBACK_DAYS = 7  # Default lookback period if no previous update

# --- Global Variables ---
llm_model_info: Dict = {}
prompts: Dict = {}
rolling_summary_prompt_template: str = ""


def load_prompts():
    """Load required prompts from the prompts.yml file"""
    global prompts, rolling_summary_prompt_template
    
    try:
        with open(PROMPTS_FILE_PATH, "r", encoding="utf-8") as f:
            prompts = yaml.safe_load(f)
        
        if not prompts or 'rolling_summary_prompt' not in prompts:
            raise ValueError("CRITICAL: 'rolling_summary_prompt' not found in prompts.yml.")
        
        rolling_summary_prompt_template = prompts['rolling_summary_prompt']
        logger.info("Successfully loaded rolling summary prompt from prompts.yml")
        
    except FileNotFoundError:
        logger.critical(f"CRITICAL: {PROMPTS_FILE_PATH} not found. Personal summary generation will fail.")
        rolling_summary_prompt_template = "Error: Prompts file not found."
        raise
    except Exception as e:
        logger.critical(f"Error loading prompts: {e}")
        raise


async def initialize_llm(model_type="default"):
    """Initialize the LLM model for personalized summary generation"""
    global llm_model_info
    
    try:
        # Use default model for personalized summaries with grounding for current context
        model_info_dict = initialize_model(provider="gemini", model_type=model_type, grounding_enabled=True)
        llm_model_info.update(model_info_dict)
        logger.info(f"LLM ready for Personal Summary Generator: {llm_model_info.get('model_name')} (Grounding: {llm_model_info.get('grounding_enabled')})")
        
    except Exception as e:
        logger.error(f"Failed to initialize LLM in Personal Summary Generator: {e}", exc_info=True)
        raise


def format_new_articles_for_prompt(articles: List[Dict]) -> str:
    """
    Formats new articles for inclusion in the rolling summary prompt.
    
    Args:
        articles (List[Dict]): List of article records from database.
        
    Returns:
        str: Formatted string of article titles and snippets.
    """
    if not articles:
        return "No new articles found."
    
    formatted_articles = []
    for article in articles:
        title = article.get('title', 'Untitled')
        # Get first 200 characters of article text as snippet
        content = article.get('article_text', '')
        snippet = content[:200] + "..." if len(content) > 200 else content
        publishedAt = article.get('publishedAt', 'Unknown date')

        formatted_articles.append(f"• {title} ({publishedAt}): {snippet}")

    return "\n".join(formatted_articles)


def format_new_stats_for_prompt(stats: List[Dict]) -> str:
    """
    Formats new player statistics for inclusion in the rolling summary prompt.
    
    Args:
        stats (List[Dict]): List of player stats records from database.
        
    Returns:
        str: Formatted string of key statistics.
    """
    if not stats:
        return "No new statistics found."
    
    formatted_stats = []
    for stat in stats:
        game_info = stat.get('games', {})
        game_date = game_info.get('gameday', 'Unknown date')
        week = game_info.get('week', '?')
        season = game_info.get('season', '?')
        
        # Extract key statistics
        key_stats = []
        if stat.get('passing_yards'):
            key_stats.append(f"{stat['passing_yards']} pass yds")
        if stat.get('passing_tds'):
            key_stats.append(f"{stat['passing_tds']} pass TDs")
        if stat.get('rushing_yards'):
            key_stats.append(f"{stat['rushing_yards']} rush yds")
        if stat.get('rushing_tds'):
            key_stats.append(f"{stat['rushing_tds']} rush TDs")
        if stat.get('receiving_yards'):
            key_stats.append(f"{stat['receiving_yards']} rec yds")
        if stat.get('receiving_tds'):
            key_stats.append(f"{stat['receiving_tds']} rec TDs")
        if stat.get('fantasy_points_ppr'):
            key_stats.append(f"{stat['fantasy_points_ppr']} fantasy pts")
        
        stats_text = ", ".join(key_stats) if key_stats else "No significant stats"
        formatted_stats.append(f"• Week {week}, {season} ({game_date}): {stats_text}")
    
    return "\n".join(formatted_stats)


async def generate_rolling_summary(
    previous_summary: Optional[str],
    new_articles: List[Dict],
    new_stats: List[Dict],
    entity_id: str,
    entity_type: str
) -> Optional[str]:
    """
    Generates a rolling summary using the LLM with previous context and new information.
    
    Args:
        previous_summary (Optional[str]): The previous summary for this entity.
        new_articles (List[Dict]): New articles about the entity.
        new_stats (List[Dict]): New statistics for the entity (if player).
        entity_id (str): The entity ID.
        entity_type (str): The entity type ('player' or 'team').
        
    Returns:
        Optional[str]: The generated rolling summary or None if generation failed.
    """
    try:
        # Format the new information
        new_articles_text = format_new_articles_for_prompt(new_articles)
        new_stats_text = format_new_stats_for_prompt(new_stats) if entity_type == "player" else "N/A (Team entity)"
        
        # Use empty string for first-time summaries
        previous_summary_text = previous_summary or "This is the first update for this entity."
        
        # Format the prompt
        prompt = rolling_summary_prompt_template.format(
            previous_summary_text=previous_summary_text,
            new_stats_text=new_stats_text,
            new_article_titles_and_snippets=new_articles_text
        )
        
        logger.info(f"Generating rolling summary for entity {entity_id} ({entity_type})")
        logger.debug(f"Prompt preview: {prompt[:200]}...")
        
        # Generate content using the LLM
        if not llm_model_info.get("model"):
            logger.error("LLM model not initialized")
            return None
        
        model = llm_model_info["model"]
        model_name = llm_model_info["model_name"]
        tools = llm_model_info.get("tools", [])
        
        # Call the LLM with correct API pattern
        response = await asyncio.to_thread(
            model.generate_content,
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.7,
                max_output_tokens=1000,
                tools=tools if tools else []
            )
        )
        
        if not response or not response.text:
            logger.error(f"Empty response from LLM for entity {entity_id}")
            return None
        
        generated_summary = response.text.strip()
        logger.info(f"Successfully generated rolling summary for entity {entity_id} ({len(generated_summary)} characters)")
        
        return generated_summary
        
    except Exception as e:
        logger.error(f"Error generating rolling summary for entity {entity_id}: {e}", exc_info=True)
        return None


async def get_default_since_timestamp() -> str:
    """
    Gets the default "since" timestamp for entities with no previous updates.
    
    Returns:
        str: ISO timestamp for DEFAULT_LOOKBACK_DAYS ago.
    """
    default_date = datetime.now(timezone.utc) - timedelta(days=DEFAULT_LOOKBACK_DAYS)
    return default_date.isoformat()


async def process_user_entity_preference(user_id: str, preference: Dict) -> bool:
    """
    Processes a single user's preference for a specific entity.
    
    Args:
        user_id (str): The user ID.
        preference (Dict): The user preference record containing entity_id and entity_type.
        
    Returns:
        bool: True if processing was successful, False otherwise.
    """
    entity_id = preference.get('entity_id')
    entity_type = preference.get('entity_type')
    
    if not entity_id or not entity_type:
        logger.warning(f"Invalid preference for user {user_id}: missing entity_id or entity_type")
        return False
    
    try:
        logger.info(f"Processing preference for user {user_id}: {entity_type} {entity_id}")
        
        # Step 1: Get the timestamp of the last update for this user-entity combination
        last_update_timestamp = await get_last_update_timestamp(user_id, entity_id)
        
        # Step 2: Determine the "since" timestamp
        if last_update_timestamp:
            since_timestamp = last_update_timestamp
            logger.info(f"Found previous update at {since_timestamp}, looking for content since then")
        else:
            since_timestamp = await get_default_since_timestamp()
            logger.info(f"No previous update found, looking for content since {since_timestamp} (default {DEFAULT_LOOKBACK_DAYS} days)")
        
        # Step 3: Fetch new articles about this entity
        new_articles = await fetch_new_articles_for_entity(entity_id, entity_type, since_timestamp)
        logger.info(f"Found {len(new_articles)} new articles for {entity_type} {entity_id}")
        
        # Step 4: Fetch new stats if this is a player
        new_stats = []
        if entity_type == "player":
            new_stats = await fetch_new_stats_for_player(entity_id, since_timestamp)
            logger.info(f"Found {len(new_stats)} new stat records for player {entity_id}")
        
        # Step 5: Check if there's any new content to process
        if not new_articles and not new_stats:
            logger.info(f"No new content found for user {user_id}, entity {entity_id}. Skipping update.")
            return True  # Not an error, just no new content
        
        # Step 6: Get the previous summary for context
        previous_summary = await get_previous_summary_for_entity(user_id, entity_id)
        if previous_summary:
            logger.info(f"Found previous summary for context ({len(previous_summary)} characters)")
        else:
            logger.info("No previous summary found, this will be a fresh summary")
        
        # Step 7: Generate the rolling summary
        rolling_summary = await generate_rolling_summary(
            previous_summary=previous_summary,
            new_articles=new_articles,
            new_stats=new_stats,
            entity_id=entity_id,
            entity_type=entity_type
        )
        
        if not rolling_summary:
            logger.error(f"Failed to generate rolling summary for user {user_id}, entity {entity_id}")
            return False
        
        # Step 8: Save the generated update to the database
        source_article_ids = [str(article.get('id')) for article in new_articles if article.get('id')]
        source_stat_ids = [str(stat.get('stat_id')) for stat in new_stats if stat.get('stat_id')]
        
        success = await save_generated_update(
            user_id=user_id,
            entity_id=entity_id,
            entity_type=entity_type,
            update_content=rolling_summary,
            source_article_ids=source_article_ids,
            source_stat_ids=source_stat_ids
        )
        
        if success:
            logger.info(f"Successfully saved rolling summary for user {user_id}, entity {entity_id}")
            return True
        else:
            logger.error(f"Failed to save rolling summary for user {user_id}, entity {entity_id}")
            return False
            
    except Exception as e:
        logger.error(f"Exception processing preference for user {user_id}, entity {entity_id}: {e}", exc_info=True)
        return False


async def process_user_preferences(user_id: str) -> Dict[str, int]:
    """
    Processes all preferences for a single user.
    
    Args:
        user_id (str): The user ID to process.
        
    Returns:
        Dict[str, int]: Statistics about the processing (total, successful, failed).
    """
    stats = {"total": 0, "successful": 0, "failed": 0}
    
    try:
        # Fetch all preferences for this user
        preferences = await fetch_user_preferences(user_id)
        
        if not preferences:
            logger.info(f"No preferences found for user {user_id}")
            return stats
        
        stats["total"] = len(preferences)
        logger.info(f"Processing {stats['total']} preferences for user {user_id}")
        
        # Process each preference
        for preference in preferences:
            success = await process_user_entity_preference(user_id, preference)
            
            if success:
                stats["successful"] += 1
            else:
                stats["failed"] += 1
        
        logger.info(f"Completed processing for user {user_id}: {stats['successful']}/{stats['total']} successful")
        return stats
        
    except Exception as e:
        logger.error(f"Exception processing preferences for user {user_id}: {e}", exc_info=True)
        stats["failed"] = stats["total"]
        return stats


async def run_personal_summary_generation() -> Dict[str, int]:
    """
    Main function that runs the personalized summary generation for all users.
    
    Returns:
        Dict[str, int]: Overall statistics about the generation process.
    """
    overall_stats = {
        "total_users": 0,
        "users_processed": 0,
        "total_preferences": 0,
        "successful_summaries": 0,
        "failed_summaries": 0,
        "errors": 0
    }
    
    try:
        logger.info("Starting personalized summary generation process...")
        
        # Step 1: Fetch all users
        users = await fetch_all_users()
        
        if not users:
            logger.warning("No users found in the database")
            return overall_stats
        
        overall_stats["total_users"] = len(users)
        logger.info(f"Found {overall_stats['total_users']} users to process")
        
        # Step 2: Process each user
        for user in users:
            user_id = user.get('user_id')
            
            if not user_id:
                logger.warning("Found user record without user_id, skipping")
                overall_stats["errors"] += 1
                continue
            
            try:
                logger.info(f"Processing user {user_id}")
                
                # Process all preferences for this user
                user_stats = await process_user_preferences(user_id)
                
                # Update overall statistics
                overall_stats["users_processed"] += 1
                overall_stats["total_preferences"] += user_stats["total"]
                overall_stats["successful_summaries"] += user_stats["successful"]
                overall_stats["failed_summaries"] += user_stats["failed"]
                
                logger.info(f"User {user_id} completed: {user_stats['successful']}/{user_stats['total']} preferences processed successfully")
                
            except Exception as e:
                logger.error(f"Error processing user {user_id}: {e}", exc_info=True)
                overall_stats["errors"] += 1
        
        # Step 3: Log final summary
        logger.info("Personalized summary generation completed!")
        logger.info(f"Final stats: {overall_stats}")
        
        return overall_stats
        
    except Exception as e:
        logger.error(f"Critical error in personal summary generation: {e}", exc_info=True)
        overall_stats["errors"] += 1
        return overall_stats


async def main():
    """Main entry point for the personal summary generator"""
    try:
        # Initialize components
        logger.info("Initializing Personal Summary Generator...")
        
        # Check database connection
        if not _check_supabase_client():
            logger.critical("Database client not available. Cannot proceed.")
            return
        
        # Load prompts
        load_prompts()
        
        # Initialize LLM
        await initialize_llm()
        
        # Verify initialization
        if not rolling_summary_prompt_template or "Error:" in rolling_summary_prompt_template:
            logger.critical("Rolling summary prompt not loaded correctly. Cannot proceed.")
            return
        
        if not llm_model_info.get("model"):
            logger.critical("LLM model not initialized correctly. Cannot proceed.")
            return
        
        logger.info("Initialization completed successfully")
        
        # Run the main generation process
        stats = await run_personal_summary_generation()
        
        # Print final summary
        print("\n" + "="*50)
        print("PERSONAL SUMMARY GENERATION COMPLETED")
        print("="*50)
        print(f"Total Users: {stats['total_users']}")
        print(f"Users Processed: {stats['users_processed']}")
        print(f"Total Preferences: {stats['total_preferences']}")
        print(f"Successful Summaries: {stats['successful_summaries']}")
        print(f"Failed Summaries: {stats['failed_summaries']}")
        print(f"Errors: {stats['errors']}")
        print("="*50)
        
        # Exit with appropriate code
        if stats["errors"] > 0 or stats["failed_summaries"] > 0:
            logger.warning("Process completed with some errors or failures")
            sys.exit(1)
        else:
            logger.info("Process completed successfully")
            sys.exit(0)
            
    except Exception as e:
        logger.critical(f"Critical error in main: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
