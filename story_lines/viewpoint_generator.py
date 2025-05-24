# story_lines/viewpoint_generator.py

import asyncio
import json
import logging
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

import yaml
from dotenv import load_dotenv
from google.genai import types as genai_types

# Add parent directory to import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from LLMSetup import initialize_model as llm_setup_initialize_model
from post_processing import remove_citations_from_text # Though likely not needed for viewpoint names/justifications
from story_lines.article_fetcher import (
    format_source_articles_for_analysis,
    extract_cluster_article_content
)


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
llm_model_info: Dict = {}
prompts: Dict = {}
viewpoint_prompt_template: str = ""

def load_viewpoint_prompts():
    """Load required prompts from the prompts.yml file"""
    global prompts, viewpoint_prompt_template
    try:
        with open(PROMPTS_FILE_PATH, 'r', encoding="utf-8") as f:
            prompts = yaml.safe_load(f)

        if 'determine_viewpoints_prompt' not in prompts:
            logger.error("Required 'determine_viewpoints_prompt' not found in prompts.yml")
            raise ValueError("Required prompt not found: determine_viewpoints_prompt")
        viewpoint_prompt_template = prompts['determine_viewpoints_prompt']
        logger.info("Prompts loaded successfully for ViewpointGenerator.")
    except FileNotFoundError:
        logger.error(f"Prompts file not found at {PROMPTS_FILE_PATH}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML in prompts file: {e}")
        raise
    except ValueError as e:
        logger.error(str(e))
        raise

async def initialize_viewpoint_llm(model_type="flash"): # flash is good for classification/extraction
    """Initialize the LLM model for viewpoint determination"""
    global llm_model_info
    try:
        model_info_dict = llm_setup_initialize_model(provider="gemini", model_type=model_type, grounding_enabled=False) # Grounding less critical here
        llm_model_info.update(model_info_dict)
        logger.info(f"LLM ready for ViewpointGenerator: {llm_model_info.get('model_name')}")
    except Exception as e:
        logger.error(f"Failed to initialize LLM in ViewpointGenerator: {e}", exc_info=True)
        raise

def _prepare_story_context(cluster_data: Dict) -> str:
    """Prepares the story context string for the LLM prompt."""
    source_articles = cluster_data.get('source_articles', [])
    cluster_article_obj = cluster_data.get('cluster_article') # Renamed to avoid conflict

    context_parts = []

    if cluster_article_obj:
        headline, summary, content = extract_cluster_article_content(cluster_article_obj)
        if headline:
            context_parts.append(f"Synthesized Article Headline: {headline}")
        if summary:
            context_parts.append(f"Synthesized Article Summary: {summary}")
        # Optionally add a snippet of synthesized content if needed, but headline and summary are often enough
        # if content:
        #     context_parts.append(f"Synthesized Article Content Snippet: {content[:500]}...") # Keep it brief
        
        # Also include some source material context
        if source_articles:
            context_parts.append("\nKey Source Articles Overview:")
            # Take top 2-3 source articles formatted
            formatted_sources = format_source_articles_for_analysis(source_articles[:3]) # Limit for brevity
            context_parts.append(formatted_sources)

    elif source_articles:
        context_parts.append("Source Articles Overview:")
        # Format all source articles (or a limited number if too many)
        formatted_sources = format_source_articles_for_analysis(source_articles[:5]) # Limit for brevity
        context_parts.append(formatted_sources)
    else:
        return "No specific story content available to analyze."

    return "\n\n".join(context_parts)


async def determine_viewpoints(cluster_data: Dict) -> List[Dict]:
    """
    Determines relevant viewpoints for a given cluster's story.

    Args:
        cluster_data: A dictionary containing 'source_articles' and optionally 'cluster_article'.

    Returns:
        A list of dictionaries, where each dictionary has "name" and "justification".
    """
    if not llm_model_info.get("model") or not viewpoint_prompt_template:
        logger.error("Viewpoint LLM or prompt not initialized.")
        return []

    story_context = _prepare_story_context(cluster_data)
    if "No specific story content available" in story_context:
        logger.warning(f"Skipping viewpoint generation for cluster {cluster_data.get('cluster_id')} due to lack of content.")
        return []

    # Truncate story_context if it's too long to avoid LLM input limits
    # A typical limit for Gemini Flash input is around 30k tokens, but we should be more conservative.
    # Let's aim for roughly 10k-15k characters as a soft cap for the context.
    max_context_length = 15000 
    if len(story_context) > max_context_length:
        logger.warning(f"Story context for cluster {cluster_data.get('cluster_id')} is too long ({len(story_context)} chars). Truncating to {max_context_length} chars.")
        story_context = story_context[:max_context_length] + "...\n[CONTEXT TRUNCATED]"


    full_prompt = viewpoint_prompt_template.format(story_context=story_context)

    logger.info(f"Attempting to determine viewpoints for cluster: {cluster_data.get('cluster_id')}")
    # logger.debug(f"Viewpoint generation prompt context:\n{story_context[:500]}...") # Log snippet of context

    try:
        response = await asyncio.to_thread(
            llm_model_info["model"].generate_content,
            model=llm_model_info["model_name"],
            contents=full_prompt,
            config=genai_types.GenerateContentConfig(
                temperature=0.3, # Lower temperature for more focused, less creative viewpoints
                max_output_tokens=1024, # Should be enough for 3-5 viewpoints in JSON
                # tools=llm_model_info.get("tools", []) # Grounding tools not typically needed here
            )
        )

        if response and hasattr(response, 'text') and response.text:
            raw_text = response.text.strip()
            
            # Extract JSON part
            # Remove potential markdown code block fences
            if raw_text.startswith("```json"):
                raw_text = raw_text[len("```json"):].strip()
            elif raw_text.startswith("```"):
                 raw_text = raw_text[len("```"):].strip()
            if raw_text.endswith("```"):
                raw_text = raw_text[:-len("```")].strip()

            # Sometimes the LLM might still add a sentence before/after the JSON.
            # Try to find the JSON array specifically.
            json_match = re.search(r'\[\s*\{.*?\}\s*\]', raw_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                logger.warning(f"Could not robustly find JSON array in LLM response for viewpoints. Full response: {raw_text}")
                json_str = raw_text # Fallback to trying to parse the whole cleaned string

            try:
                viewpoints = json.loads(json_str)
                if isinstance(viewpoints, list) and all(isinstance(vp, dict) and "name" in vp and "justification" in vp for vp in viewpoints):
                    logger.info(f"Successfully determined {len(viewpoints)} viewpoints for cluster {cluster_data.get('cluster_id')}.")
                    return viewpoints
                else:
                    logger.error(f"Parsed JSON for viewpoints is not in the expected format (list of dicts with name/justification). Parsed: {viewpoints}")
                    return []
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse LLM response for viewpoints as JSON: {e}. Response text: {json_str[:500]}...")
                return []
        else:
            logger.warning(f"LLM response for viewpoint determination was empty or malformed for cluster {cluster_data.get('cluster_id')}.")
            return []

    except Exception as e:
        logger.error(f"Error during LLM call for viewpoint determination: {e}", exc_info=True)
        return []

if __name__ == "__main__":
    # This is a test block. You'll need a mock `cluster_data` structure.
    async def test_viewpoint_generation():
        logger.info("Testing Viewpoint Generation...")
        try:
            load_viewpoint_prompts()
            await initialize_viewpoint_llm()
        except Exception as e:
            logger.critical(f"Initialization for testing failed: {e}")
            return

        # Mock cluster_data (replace with actual data fetching if testing against DB)
        mock_cluster_data_with_cluster_article = {
            "cluster_id": "test-cluster-123",
            "source_articles": [
                {'id': 1, 'headline': 'Team A Wins Big Game!', 'Content': 'Team A played exceptionally well and secured a victory against Team B. The quarterback threw three touchdowns.', 'created_at': '2023-10-26T10:00:00Z', 'NewsSource': {'Name': 'Sports News Network'}},
                {'id': 2, 'headline': 'Player X Injured in Match', 'Content': 'Star player X from Team A suffered an unfortunate knee injury during the second quarter. The extent is unknown.', 'created_at': '2023-10-26T11:00:00Z', 'NewsSource': {'Name': 'Local Gazette'}},
            ],
            "cluster_article": {
                "headline": "<h1>Team A Victorious but Loses Star Player X to Injury</h1>",
                "summary": "<p>Team A celebrated a significant win today, but the victory was overshadowed by a serious injury to their key player, X, raising concerns about their upcoming games.</p>",
                "content": "<div><p>The game was a rollercoaster of emotions. Team A dominated early on, leading to a decisive win. However, the injury to player X cast a pall over the celebrations. Medical staff attended to X on the field before he was helped off. Coach Z expressed concern in the post-match press conference but remained hopeful for a quick recovery. This incident will undoubtedly affect team strategy moving forward.</p></div>"
            }
        }
        
        mock_cluster_data_sources_only = {
            "cluster_id": "test-cluster-456",
            "source_articles": [
                {'id': 3, 'headline': 'Major Trade Announced: Player Y Joins Team C', 'Content': 'In a surprising move, Player Y has been traded from Team D to Team C. The trade involves multiple draft picks.', 'created_at': '2023-10-27T09:00:00Z', 'NewsSource': {'Name': 'ProSports Weekly'}},
                {'id': 4, 'headline': 'Fans React to Player Y Trade', 'Content': 'Social media is buzzing with reactions from fans of both Team C and Team D regarding the Player Y trade. Opinions are divided.', 'created_at': '2023-10-27T12:00:00Z', 'NewsSource': {'Name': 'FanTalk Radio'}},
            ],
            "cluster_article": None
        }

        logger.info("\n--- Test Case 1: With Synthesized Cluster Article ---")
        viewpoints1 = await determine_viewpoints(mock_cluster_data_with_cluster_article)
        if viewpoints1:
            for i, vp in enumerate(viewpoints1):
                print(f"Viewpoint {i+1}: {vp['name']}")
                print(f"  Justification: {vp['justification']}")
        else:
            print("No viewpoints generated for Test Case 1.")

        logger.info("\n--- Test Case 2: Source Articles Only ---")
        viewpoints2 = await determine_viewpoints(mock_cluster_data_sources_only)
        if viewpoints2:
            for i, vp in enumerate(viewpoints2):
                print(f"Viewpoint {i+1}: {vp['name']}")
                print(f"  Justification: {vp['justification']}")
        else:
            print("No viewpoints generated for Test Case 2.")
            
        logger.info("Viewpoint generation test completed.")

    asyncio.run(test_viewpoint_generation())