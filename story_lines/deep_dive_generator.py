# story_lines/deep_dive_generator.py

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
from story_lines.article_fetcher import (
    format_source_articles_for_analysis,
    extract_cluster_article_content
)
from post_processing import remove_citations_from_text
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
llm_model_info: Dict = {}
prompts: Dict = {}
deep_dive_prompt_template: str = ""

def load_deep_dive_prompts():
    """Load required prompts from the prompts.yml file"""
    global prompts, deep_dive_prompt_template
    try:
        with open(PROMPTS_FILE_PATH, 'r', encoding="utf-8") as f:
            prompts = yaml.safe_load(f)

        if 'generate_deep_dive_article_prompt' not in prompts:
            logger.error("Required 'generate_deep_dive_article_prompt' not found in prompts.yml")
            raise ValueError("Required prompt not found: generate_deep_dive_article_prompt")
        deep_dive_prompt_template = prompts['generate_deep_dive_article_prompt']
        logger.info("Prompts loaded successfully for DeepDiveGenerator.")
    except FileNotFoundError:
        logger.error(f"Prompts file not found at {PROMPTS_FILE_PATH}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML in prompts file: {e}")
        raise
    except ValueError as e:
        logger.error(str(e))
        raise

async def initialize_deep_dive_llm(model_type="default"): # Default model usually better for generation
    """Initialize the LLM model for deep-dive article generation with grounding."""
    global llm_model_info
    try:
        # Ensure grounding_enabled is True for deep dives
        model_info_dict = llm_setup_initialize_model(provider="gemini", model_type=model_type, grounding_enabled=True)
        llm_model_info.update(model_info_dict)
        logger.info(f"LLM ready for DeepDiveGenerator: {llm_model_info.get('model_name')} (Grounding: {llm_model_info.get('grounding_enabled')})")
    except Exception as e:
        logger.error(f"Failed to initialize LLM in DeepDiveGenerator: {e}", exc_info=True)
        raise

def _prepare_story_context_for_deep_dive(cluster_data: Dict) -> str:
    """Prepares the story context string for the deep-dive LLM prompt."""
    source_articles = cluster_data.get('source_articles', [])
    cluster_article_obj = cluster_data.get('cluster_article')

    context_parts = []

    if cluster_article_obj:
        headline, summary, content = extract_cluster_article_content(cluster_article_obj)
        if headline:
            context_parts.append(f"**Synthesized Article Headline:**\n{headline}")
        if summary:
            context_parts.append(f"\n**Synthesized Article Summary:**\n{summary}")
        if content: # Include full synthesized content for deep dive
            context_parts.append(f"\n**Synthesized Article Content:**\n{content}")
    
    if source_articles:
        context_parts.append("\n**Key Source Articles (for reference):**")
        # Format all source articles (or a limited number if too many)
        # For deep dives, providing more source context can be beneficial
        formatted_sources = format_source_articles_for_analysis(source_articles) # Pass all for more context
        context_parts.append(formatted_sources)
    
    if not context_parts:
        return "No specific story content available to analyze."

    full_context = "\n\n".join(context_parts)
    
    # Truncate story_context if it's too long
    # Gemini 1.5 Flash input context window is large (1M tokens), but let's be mindful.
    # For "gemini-pro" or other models, this might be more critical.
    # A character limit is a rough proxy for token limit.
    max_context_chars = 50000 # Increased for deep dive context
    if len(full_context) > max_context_chars:
        logger.warning(f"Deep dive story context for cluster {cluster_data.get('cluster_id')} is very long ({len(full_context)} chars). Truncating to {max_context_chars}.")
        full_context = full_context[:max_context_chars] + "\n...[STORY CONTEXT TRUNCATED]..."
        
    return full_context

def _parse_llm_deep_dive_response(raw_text: str) -> Dict[str, str]:
    """Parses the LLM's response into headline, introduction, bullet points, and article."""
    parsed = {
        "headline": "Could not parse headline.",
        "introduction": "Could not parse introduction.",
        "bullet_points": "Could not parse bullet points.",
        "article": "Could not parse article."
    }

    # Extract headline
    headline_match = re.search(r"\[START Headline\](.*?)\[END Headline\]", raw_text, re.DOTALL)
    if headline_match:
        parsed["headline"] = headline_match.group(1).strip()

    intro_match = re.search(r"\[START INTRODUCTION\](.*?)\[END INTRODUCTION\]", raw_text, re.DOTALL)
    if intro_match:
        parsed["introduction"] = intro_match.group(1).strip()

    bullets_match = re.search(r"\[START BULLET POINTS\](.*?)\[END BULLET POINTS\]", raw_text, re.DOTALL)
    if bullets_match:
        parsed["bullet_points"] = bullets_match.group(1).strip()

    article_match = re.search(r"\[START ARTICLE\](.*?)\[END ARTICLE\]", raw_text, re.DOTALL)
    if article_match:
        parsed["article"] = article_match.group(1).strip()
    elif not intro_match and not bullets_match: # If no markers, assume whole text is article
        logger.warning("No section markers found in LLM response. Assuming entire text is the article.")
        parsed["article"] = raw_text.strip()
        # Try to extract some bullets if possible if they exist without markers
        potential_bullets = []
        for line in raw_text.splitlines():
            if line.strip().startswith(("- ", "* ")):
                potential_bullets.append(line.strip())
        if potential_bullets:
            parsed["bullet_points"] = "\n".join(potential_bullets)

    # Remove citations from all parsed content
    for key in parsed:
        original_content = parsed[key]
        cleaned_content = remove_citations_from_text(original_content)
        if original_content != cleaned_content:
            logger.debug(f"Removed citations from {key} section")
        parsed[key] = cleaned_content

    return parsed


async def generate_deep_dive_for_viewpoint(
    cluster_data: Dict,
    viewpoint: Dict,
    cluster_id: str # For logging
) -> Optional[Dict[str, str]]:
    """
    Generates a deep-dive article for a specific viewpoint.

    Args:
        cluster_data: Contains source_articles and cluster_article.
        viewpoint: A dict with "name" and "justification".
        cluster_id: The ID of the cluster for logging.

    Returns:
        A dictionary with "introduction", "bullet_points", and "article", or None if failed.
    """
    if not llm_model_info.get("model") or not deep_dive_prompt_template:
        logger.error(f"Deep Dive LLM or prompt not initialized for cluster {cluster_id}, viewpoint '{viewpoint.get('name')}'.")
        return None

    story_context = _prepare_story_context_for_deep_dive(cluster_data)
    if "No specific story content available" in story_context:
        logger.warning(f"Skipping deep dive for viewpoint '{viewpoint.get('name')}' in cluster {cluster_id} due to lack of story content.")
        return None

    full_prompt = deep_dive_prompt_template.format(
        story_context=story_context,
        viewpoint_name=viewpoint.get("name", "N/A"),
        viewpoint_justification=viewpoint.get("justification", "N/A")
    )

    logger.info(f"Generating deep dive for cluster {cluster_id}, viewpoint: '{viewpoint.get('name')}'")

    try:
        # Use a model that supports grounding well (e.g., gemini-1.5-flash or gemini-pro)
        # The grounding_enabled=True was set during initialize_deep_dive_llm
        # The tools for grounding are also set in llm_model_info["tools"]
        
        response = await asyncio.to_thread(
            llm_model_info["model"].generate_content,
            model=llm_model_info["model_name"], # e.g., "gemini-1.5-flash-latest" or your default
            contents=full_prompt,
            config=genai_types.GenerateContentConfig(
                temperature=0.5, # Slightly higher for more creative/in-depth writing
                max_output_tokens=4096, # Allow for longer articles
                tools=llm_model_info.get("tools", []) # Pass grounding tools
            )
        )

        if response and hasattr(response, 'text') and response.text:
            raw_text = response.text
            parsed_article = _parse_llm_deep_dive_response(raw_text)
            
            # --- Print to console ---
            print(f"\n\n--- DEEP DIVE ARTICLE ---")
            print(f"Cluster ID: {cluster_id}")
            print(f"Viewpoint: {viewpoint.get('name', 'N/A')}")
            print(f"--------------------------")
            print(f"\n[INTRODUCTION]\n{parsed_article['introduction']}")
            print(f"\n[KEY BULLET POINTS]\n{parsed_article['bullet_points']}")
            print(f"\n[ARTICLE CONTENT]\n{parsed_article['article']}")
            print(f"--- END OF DEEP DIVE ---")
            
            return parsed_article
        else:
            logger.warning(f"LLM response for deep dive (cluster {cluster_id}, viewpoint '{viewpoint.get('name')}') was empty or malformed.")
            if response and hasattr(response, 'prompt_feedback') and response.prompt_feedback:
                logger.warning(f"Prompt Feedback: {response.prompt_feedback}")
            return None

    except Exception as e:
        logger.error(f"Error during LLM call for deep dive (cluster {cluster_id}, viewpoint '{viewpoint.get('name')}'): {e}", exc_info=True)
        return None


if __name__ == "__main__":
    async def test_deep_dive_generation():
        logger.info("Testing Deep Dive Article Generation...")
        try:
            load_deep_dive_prompts()
            # Use a model good for generation and ensure grounding is enabled
            await initialize_deep_dive_llm(model_type="default") 
        except Exception as e:
            logger.critical(f"Initialization for testing failed: {e}")
            return

        mock_cluster_data = {
            "cluster_id": "test-dd-cluster-789",
            "source_articles": [
                {'id': 1, 'headline': 'City Council Approves New Downtown Development Project', 'Content': 'After a lengthy debate, the city council voted 5-2 to approve the controversial "SkyTower" development project. Proponents cite job creation, while opponents worry about traffic and gentrification.', 'created_at': '2023-10-26T10:00:00Z', 'NewsSource': {'Name': 'City News Daily'}},
                {'id': 2, 'headline': 'Community Groups Protest SkyTower Decision', 'Content': 'Several local community organizations have announced plans for a protest rally this weekend, decrying the lack of public consultation before the SkyTower approval.', 'created_at': '2023-10-27T11:00:00Z', 'NewsSource': {'Name': 'The Urban Voice'}},
            ],
            "cluster_article": {
                "headline": "<h1>SkyTower Development Greenlit Amidst Community Backlash</h1>",
                "summary": "<p>The city's ambitious SkyTower project has received official approval, promising economic benefits but also sparking significant opposition from residents concerned about its impact on the urban landscape and community character.</p>",
                "content": "<div><p>The decision to move forward with the SkyTower, a mixed-use high-rise, marks a pivotal moment for the city's development. Supporters, including the Chamber of Commerce, highlight the potential for thousands of new jobs and increased tax revenue. However, critics, represented by groups like 'Save Our Neighborhoods,' argue that the project will exacerbate existing housing shortages for low-income residents and overwhelm local infrastructure. The council meeting was reportedly tense, with passionate testimonies from both sides.</p></div>"
            }
        }
        
        mock_viewpoint = {
            "name": "Economic Impact vs. Community Displacement",
            "justification": "This viewpoint will critically examine the projected economic benefits of the SkyTower project against the potential negative consequences of resident displacement and changes to the community's social fabric."
        }

        generated_article_parts = await generate_deep_dive_for_viewpoint(
            mock_cluster_data, 
            mock_viewpoint,
            cluster_id=mock_cluster_data["cluster_id"]
        )

        if generated_article_parts:
            logger.info("Deep dive article generated successfully (check console output).")
        else:
            logger.error("Failed to generate deep dive article.")
            
        logger.info("Deep dive generation test completed.")

    asyncio.run(test_deep_dive_generation())