import asyncio
import json
import logging
import os
from typing import Dict, List, Any, Union
from uuid import UUID, uuid4

import yaml
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Add parent directory to import path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from database import get_untranslated_story_line_views, save_translated_story_line_view
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
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- LLM and Prompts ---
llm_model_info = {}
prompts = {}


def load_prompts():
    """Load required prompts from the prompts.yml file"""
    global prompts
    try:
        with open(PROMPTS_FILE_PATH, 'r') as f:
            prompts = yaml.safe_load(f)
        
        if 'deep_dive_translation_prompt' not in prompts:
            logger.error("Required 'deep_dive_translation_prompt' not found in prompts.yml")
            raise ValueError("Required prompt not found: deep_dive_translation_prompt")
            
        logger.info("Prompts loaded successfully for deep dive translator.")
    except FileNotFoundError:
        logger.error(f"Prompts file not found at {PROMPTS_FILE_PATH}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML in prompts file: {e}")
        raise
    except ValueError as e:
        logger.error(str(e))
        raise


async def initialize_llm(model_type="flash"):
    """Initialize the LLM model for translation - using flash model by default as it's good for translation"""
    global llm_model_info
    try:
        # Use LLMSetup.py to get the model, but specify a different model than for generation
        model_info = llm_setup_initialize_model(provider="gemini", model_type=model_type, grounding_enabled=False)
        
        llm_model_info["model_name"] = model_info["model_name"]
        llm_model_info["model"] = model_info["model"]  # This is the client.models object
        llm_model_info["tools"] = model_info["tools"]
        llm_model_info["grounding_enabled"] = model_info["grounding_enabled"]
        
        logger.info(f"LLM ready for deep dive translator: {llm_model_info['model_name']} (using LLMSetup pattern)")
        return llm_model_info
    except Exception as e:
        logger.error(f"Failed to initialize LLM for deep dive translation: {e}", exc_info=True)
        raise


async def translate_deep_dive_with_llm(deep_dive_data: Dict, language_code: str = 'de') -> Dict:
    """
    Translate deep dive article data to another language using the LLM.
    
    Args:
        deep_dive_data: The deep dive article data to translate
        language_code: The target language code (default: 'de' for German)
        
    Returns:
        Dict: The translated deep dive article data
    """
    if not llm_model_info or \
       "model_name" not in llm_model_info or \
       "model" not in llm_model_info:
        logger.error("LLM (model_name or model) is not initialized in llm_model_info.")
        raise ConnectionError("LLM model info not available for translation.")
    
    prompt_template = prompts.get('deep_dive_translation_prompt')
    if not prompt_template:
        logger.error("Deep dive translation prompt template is not loaded.")
        raise ValueError("Deep dive translation prompt not found.")
    
    # Prepare the deep dive data for translation
    deep_dive_for_translation = {
        "headline": deep_dive_data.get("headline", ""),
        "content": deep_dive_data.get("content", ""),
        "introduction": deep_dive_data.get("introduction", ""),
        "view": deep_dive_data.get("view", ""),
        "justification": deep_dive_data.get("justification", "")
    }
    
    # Format the prompt with the deep dive data
    formatted_prompt = prompt_template.format(
        language_code=language_code, 
        deep_dive_data=json.dumps(deep_dive_for_translation, indent=2)
    )
    
    try:
        # Call the LLM to translate the deep dive
        response = await asyncio.to_thread(
            llm_model_info["model"].generate_content,
            model=llm_model_info["model_name"],
            contents=formatted_prompt,
            config=types.GenerateContentConfig(
                temperature=0.2,  # Lower temperature for more consistent translations
                max_output_tokens=8192,  # Allow for longer outputs to fit the entire translation
                tools=llm_model_info.get("tools", [])
            )
        )
        
        if response and hasattr(response, 'text') and response.text:
            # Parse the translated deep dive data
            try:
                translated_text = response.text.strip()
                
                # If the response is wrapped in ```json and ```, remove those markers
                if translated_text.startswith("```json"):
                    translated_text = translated_text[7:].strip()
                if translated_text.startswith("```"):
                    translated_text = translated_text[3:].strip()
                if translated_text.endswith("```"):
                    translated_text = translated_text[:-3].strip()
                
                translated_data = json.loads(translated_text)
                
                # Clean citations from translated content
                for key in translated_data:
                    if isinstance(translated_data[key], str):
                        translated_data[key] = remove_citations_from_text(translated_data[key])
                
                logger.info(f"Successfully translated deep dive to {language_code}")
                return translated_data
            except json.JSONDecodeError as json_err:
                logger.error(f"Failed to parse translated deep dive as JSON: {json_err}")
                logger.error(f"Raw translation output: {response.text[:500]}...")
                raise ValueError(f"Failed to parse LLM translation output as JSON: {json_err}")
        else:
            logger.warning("LLM response for deep dive translation was empty or in unexpected format")
            raise ValueError("Empty or invalid translation response from LLM")
    except Exception as e:
        logger.error(f"Error during deep dive translation: {e}", exc_info=True)
        raise


async def translate_deep_dive(story_line_view_id: Union[str, UUID], deep_dive_data: Dict, language_code: str = 'de') -> tuple:
    """
    Translate a specific deep dive article to the target language.
    
    Args:
        story_line_view_id: The ID of the story line view in the database
        deep_dive_data: The deep dive article data to translate
        language_code: Target language code (default: 'de' for German)
        
    Returns:
        tuple: (success, translated_data)
    """
    try:
        translated_deep_dive = await translate_deep_dive_with_llm(deep_dive_data, language_code)
        
        # Save the translated deep dive to the database
        translation_success = await save_translated_story_line_view(
            story_line_view_id=story_line_view_id,
            language_code=language_code,
            translated_data=translated_deep_dive
        )
        
        if not translation_success:
            logger.error(f"Failed to save {language_code} translation of story line view {story_line_view_id} to database")
            return False, translated_deep_dive
            
        logger.info(f"{language_code.upper()} translation of story line view {story_line_view_id} saved to database successfully")
        return True, translated_deep_dive
    
    except Exception as e:
        logger.error(f"Failed to translate story line view {story_line_view_id}: {e}", exc_info=True)
        return False, None


async def main():
    """Main function to process all untranslated story line views"""
    try:
        load_prompts()
        await initialize_llm(model_type="flash") # Use flash model for translations
    except Exception as e:
        logger.critical(f"Initialization failed: {e}")
        return

    # Get story line views that need translation
    untranslated_story_line_views = await get_untranslated_story_line_views(language_code='de')

    if not untranslated_story_line_views:
        logger.info("No story line views found needing translation.")
        return

    logger.info(f"Found {len(untranslated_story_line_views)} story line views to translate to German.")

    # Process story line views sequentially
    results = []
    for story_line_view in untranslated_story_line_views:
        story_line_view_id = story_line_view.get('id')
        
        # Prepare deep dive data from the story line view
        deep_dive_data = {
            "headline": story_line_view.get('headline', ''),
            "content": story_line_view.get('content', ''),
            "introduction": story_line_view.get('introduction', ''),
            "view": story_line_view.get('view', ''),
            "justification": story_line_view.get('justification', '')
        }
        
        if story_line_view_id and any(deep_dive_data.values()):
            logger.info(f"Translating story line view {story_line_view_id}: '{deep_dive_data.get('view', 'Unknown View')}'")
            result = await translate_deep_dive(story_line_view_id, deep_dive_data, language_code='de')
            if result[0]:  # If translation succeeded
                results.append(result)
                logger.info(f"Successfully translated story line view {story_line_view_id}")
            else:
                logger.warning(f"Failed to translate story line view {story_line_view_id}")
                
            # Add a small delay to manage API rate limits
            await asyncio.sleep(1)
        else:
            logger.warning(f"Skipping story line view with missing ID or data: {story_line_view}")
            
    logger.info(f"Translated {len(results)} story line views successfully to German.")
    return results


if __name__ == "__main__":
    asyncio.run(main())
