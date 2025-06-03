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

from database import get_untranslated_timelines, save_translated_timeline
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
    """
    Loads required prompts from the prompts.yml file for translation tasks.

    Side Effects:
        Sets the global 'prompts' variable.
        Logs errors and raises exceptions if the file or required prompt is missing or invalid.
    """
    global prompts
    try:
        with open(PROMPTS_FILE_PATH, 'r') as f:
            prompts = yaml.safe_load(f)
        
        if 'timeline_translation_prompt' not in prompts:
            logger.error("Required 'timeline_translation_prompt' not found in prompts.yml")
            raise ValueError("Required prompt not found: timeline_translation_prompt")
            
        logger.info("Prompts loaded successfully for timeline translator.")
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
    """
    Initializes the LLM model for translation, using the specified model type (default: 'flash').

    Args:
        model_type (str): The type of model to use for translation (default: 'flash').

    Returns:
        dict: The initialized LLM model info.

    Side Effects:
        Sets the global 'llm_model_info' variable.
        Logs errors and raises exceptions if initialization fails.
    """
    global llm_model_info
    try:
        # Use LLMSetup.py to get the model, but specify a different model than for generation
        model_info = llm_setup_initialize_model(provider="gemini", model_type=model_type, grounding_enabled=False)
        
        llm_model_info["model_name"] = model_info["model_name"]
        llm_model_info["model"] = model_info["model"]  # This is the client.models object
        llm_model_info["tools"] = model_info["tools"]
        llm_model_info["grounding_enabled"] = model_info["grounding_enabled"]
        
        logger.info(f"LLM ready for timeline translator: {llm_model_info['model_name']} (using LLMSetup pattern)")
        return llm_model_info
    except Exception as e:
        logger.error(f"Failed to initialize LLM for translation: {e}", exc_info=True)
        raise


async def translate_timeline_with_llm(timeline_data: Dict, language_code: str = 'de') -> Dict:
    """
    Translate timeline data to another language using the LLM.
    
    Args:
        timeline_data: The timeline data to translate
        language_code: The target language code (default: 'de' for German)
        
    Returns:
        Dict: The translated timeline data
    """
    if not llm_model_info or \
       "model_name" not in llm_model_info or \
       "model" not in llm_model_info:
        logger.error("LLM (model_name or model) is not initialized in llm_model_info.")
        raise ConnectionError("LLM model info not available for translation.")
    
    prompt_template = prompts.get('timeline_translation_prompt')
    if not prompt_template:
        logger.error("Timeline translation prompt template is not loaded.")
        raise ValueError("Timeline translation prompt not found.")
    
    # Create a version of the timeline data without full_content for translation
    # to keep the prompt size manageable
    timeline_for_translation = {
        "cluster_id": timeline_data.get("cluster_id", ""),
        "timeline": []
    }
    
    for entry in timeline_data.get("timeline", []):
        # Copy the entry without the full_content field
        translated_entry = {k: v for k, v in entry.items() if k != 'full_content'}
        timeline_for_translation["timeline"].append(translated_entry)
    
    # Format the prompt with the timeline data
    formatted_prompt = prompt_template.format(
        language_code=language_code, 
        timeline_data=json.dumps(timeline_for_translation, indent=2)
    )
    
    try:
        # Call the LLM to translate the timeline
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
            # Parse the translated timeline data
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
                logger.info(f"Successfully translated timeline to {language_code}")
                return translated_data
            except json.JSONDecodeError as json_err:
                logger.error(f"Failed to parse translated timeline as JSON: {json_err}")
                logger.error(f"Raw translation output: {response.text[:500]}...")
                raise ValueError(f"Failed to parse LLM translation output as JSON: {json_err}")
        else:
            logger.warning("LLM response for timeline translation was empty or in unexpected format")
            raise ValueError("Empty or invalid translation response from LLM")
    except Exception as e:
        logger.error(f"Error during timeline translation: {e}", exc_info=True)
        raise


async def translate_timeline(timeline_id: Union[str, UUID], timeline_data: Dict, language_code: str = 'de') -> tuple:
    """
    Translate a specific timeline to the target language.
    
    Args:
        timeline_id: The ID of the timeline in the database
        timeline_data: The timeline data to translate
        language_code: Target language code (default: 'de' for German)
        
    Returns:
        tuple: (success, translated_data)
    """
    try:
        translated_timeline = await translate_timeline_with_llm(timeline_data, language_code)
        
        # Extract cluster_id from timeline_data to pass to save function
        cluster_id = timeline_data.get("cluster_id")
        
        # Save the translated timeline to the database
        translation_success = await save_translated_timeline(
            timeline_id=timeline_id,
            language_code=language_code,
            translated_data=translated_timeline,
            cluster_id=cluster_id  # Pass cluster_id to the save function
        )
        
        if not translation_success:
            logger.error(f"Failed to save {language_code} translation of timeline {timeline_id} to database")
            return False, translated_timeline
            
        logger.info(f"{language_code.upper()} translation of timeline {timeline_id} saved to database successfully")
        return True, translated_timeline
    
    except Exception as e:
        logger.error(f"Failed to translate timeline {timeline_id}: {e}", exc_info=True)
        return False, None


async def main():
    """Main function to process all untranslated timelines"""
    try:
        load_prompts()
        await initialize_llm(model_type="flash") # Use flash model for translations
    except Exception as e:
        logger.critical(f"Initialization failed: {e}")
        return

    # Get timelines that need translation
    untranslated_timelines = await get_untranslated_timelines(language_code='de')

    if not untranslated_timelines:
        logger.info("No timelines found needing translation.")
        return

    logger.info(f"Found {len(untranslated_timelines)} timelines to translate to German.")

    # Process timelines sequentially
    results = []
    for timeline_entry in untranslated_timelines:
        timeline_id = timeline_entry.get('id')
        timeline_data = timeline_entry.get('timeline_data')
        if timeline_id and timeline_data:
            result = await translate_timeline(timeline_id, timeline_data, language_code='de')
            if result[0]:  # If translation succeeded
                results.append(result)
        else:
            logger.warning(f"Skipping timeline with missing ID or data: {timeline_entry}")
            
    logger.info(f"Translated {len(results)} timelines successfully.")
    return results


if __name__ == "__main__":
    asyncio.run(main())
