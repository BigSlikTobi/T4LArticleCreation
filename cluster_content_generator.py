import asyncio
import json
import logging
import os
import textwrap # Use textwrap for prompt formatting
from typing import Dict, List, Optional

import google.generativeai as genai
from google.generativeai import types
import yaml

# Assuming LLMSetup is in the parent directory or accessible via PYTHONPATH
try:
    # from LLMSetup import initialize_model  # <<<< OLD IMPORT
    from LLMSetup_Cluster import initialize_model # <<<< NEW IMPORT
except ImportError as e:
     # Handle case where script might be run directly or structure differs
     logging.error(f"ImportError: {e}. Trying adjusted path...")
     import sys
     sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
     # from LLMSetup import initialize_model # <<<< OLD IMPORT
     from LLMSetup_Cluster import initialize_model # <<<< NEW IMPORT


# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
# Create a logger instance for this module
logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG) # Uncomment for verbose debug logs


# --- Load Prompts ---
# (Prompt loading logic remains the same)
prompts = {}
try:
    # Adjust path if cluster_content_generator.py is not in the root directory
    PROMPTS_FILE_PATH = 'prompts.yml'
    if not os.path.exists(PROMPTS_FILE_PATH):
         script_dir = os.path.dirname(__file__)
         parent_dir = os.path.abspath(os.path.join(script_dir, '..'))
         PROMPTS_FILE_PATH = os.path.join(parent_dir, 'prompts.yml')
         if not os.path.exists(PROMPTS_FILE_PATH):
             PROMPTS_FILE_PATH = 'prompts.yml' # Fallback to root
             if not os.path.exists(PROMPTS_FILE_PATH):
                raise FileNotFoundError("prompts.yml not found in root or parent directory.")

    with open(PROMPTS_FILE_PATH, 'r', encoding='utf-8') as f:
        prompts = yaml.safe_load(f)
    logger.info(f"Prompts loaded successfully from: {PROMPTS_FILE_PATH}")

    # Debug logs (can be removed later)
    logger.debug(f"Loaded prompt keys: {list(prompts.keys())}")
    expected_key = 'cluster_synthesis_english_prompt'
    if expected_key in prompts:
        logger.debug(f"Type of '{expected_key}': {type(prompts[expected_key])}")
    else:
        logger.error(f"Expected key '{expected_key}' NOT FOUND in loaded prompts!")

except FileNotFoundError as e:
    logger.critical(f"CRITICAL: Error loading prompts: {e}. Generation will fail.")
except Exception as e:
    logger.critical(f"CRITICAL: Error loading or parsing prompts.yml: {e}. Generation will fail.", exc_info=True)


# --- Initialize LLMs using LLMSetup_Cluster ---
synthesis_model_info = None
synthesis_model = None
translation_model_info = None
translation_model = None

try:
    # Using 'default' (likely Pro) for synthesis
    synthesis_model_info = initialize_model("gemini", "default") # Uses LLMSetup_Cluster now
    if synthesis_model_info:
        synthesis_model = synthesis_model_info["model"] # Should be GenerativeModel instance
        logger.info(f"Synthesis model initialized via LLMSetup_Cluster: {synthesis_model_info['model_name']}")
        # === ADD LOGGING FOR INITIALIZED TYPE ===
        logger.info(f"Type of initialized synthesis_model: {type(synthesis_model)}")
        # ========================================
    else:
        logger.critical("Failed to initialize synthesis model from LLMSetup_Cluster.")

    # Using 'flash' for translation
    translation_model_info = initialize_model("gemini", "flash") # Uses LLMSetup_Cluster now
    if translation_model_info:
        translation_model = translation_model_info["model"] # Should be GenerativeModel instance
        logger.info(f"Translation model initialized via LLMSetup_Cluster: {translation_model_info['model_name']}")
        # === ADD LOGGING FOR INITIALIZED TYPE ===
        logger.info(f"Type of initialized translation_model: {type(translation_model)}")
        # ========================================
    else:
        logger.critical("Failed to initialize translation model from LLMSetup_Cluster.")

except Exception as e:
    logger.critical(f"CRITICAL: An unexpected error occurred during LLM initialization via LLMSetup_Cluster: {e}. Generation will fail.", exc_info=True)


# --- Helper function _parse_llm_json_response remains the same ---
def _parse_llm_json_response(raw_response: str, expected_keys: List[str]) -> Optional[Dict]:
    """
    Parses JSON response from LLM, handling markdown code blocks and errors.
    (Code is identical to previous version)
    """
    if not raw_response:
        logger.error("Received empty response from LLM.")
        return None
    logger.debug(f"Raw LLM response received (first 500 chars): {raw_response[:500]}...")
    text = raw_response.strip()
    if text.startswith("```"):
        text = text.split('\n', 1)[-1] if '\n' in text else ''
    if text.endswith("```"):
        text = text[:-3].rstrip()
    json_start = text.find("{")
    json_end = text.rfind("}") + 1
    if json_start == -1 or json_end == -1 or json_start > json_end:
        logger.error(f"Could not find valid JSON object boundaries in response: {text[:500]}...")
        return None
    json_text = text[json_start:json_end]
    logger.debug(f"Attempting to parse JSON: {json_text[:500]}...")
    try:
        parsed_data = json.loads(json_text)
        if not all(key in parsed_data for key in expected_keys):
            missing = [key for key in expected_keys if key not in parsed_data]
            logger.error(f"Parsed JSON is missing expected keys: {missing}. Parsed JSON: {parsed_data}")
            return None
        for key in expected_keys:
            if not parsed_data.get(key):
                 logger.warning(f"Parsed JSON has empty value for expected key '{key}'.")
        logger.debug("Successfully parsed JSON response from LLM.")
        return parsed_data
    except json.JSONDecodeError as e:
        logger.error(f"Failed to decode JSON from LLM response: {e}. Response snippet: {json_text[:500]}...")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred during JSON parsing: {e}", exc_info=True)
        return None


async def synthesize_english_story(source_contents: List[str]) -> Optional[Dict]:
    """
    Generates a synthesized English article (headline, summary, content) from multiple source contents.
    """
    if not synthesis_model or not synthesis_model_info: # Check if model was initialized
        logger.error("Synthesis LLM not initialized. Cannot generate English story.")
        return None
    if 'cluster_synthesis_english_prompt' not in prompts or not prompts['cluster_synthesis_english_prompt']:
         logger.error("cluster_synthesis_english_prompt not found or empty in prompts.yml.")
         return None
    if not source_contents:
        logger.warning("No source contents provided for synthesis.")
        return None

    separator = "\n\n--- ARTICLE SEPARATOR ---\n\n"
    combined_content = separator.join(filter(None, source_contents))

    prompt = prompts['cluster_synthesis_english_prompt'].format(source_articles_content=combined_content)
    logger.info(f"Synthesizing English story from {len(source_contents)} sources...")
    logger.debug(f"Synthesis Prompt (first 500 chars): {prompt[:500]}...")

    try:
        # === ADD LOGGING BEFORE CALL ===
        logger.info(f"Calling generate_content on object of type: {type(synthesis_model)}")
        if hasattr(synthesis_model, 'model_name'):
             logger.info(f"Model name associated with object: {synthesis_model.model_name}")
        # ==============================

        response_obj = await asyncio.to_thread(
            synthesis_model.generate_content, # This is now the GenerativeModel instance
            contents=prompt,
            generation_config=types.GenerationConfig(
                temperature=0.3,
                max_output_tokens=8192,
            ),
        )
        raw_response = response_obj.text
        parsed_data = _parse_llm_json_response(raw_response, ["headline", "summary", "content"])

        if parsed_data:
            logger.info("Successfully synthesized English story.")
            if not parsed_data.get("headline") or not parsed_data.get("summary") or not parsed_data.get("content"):
                logger.warning("Synthesized English story has empty essential fields.")
                return None
            if len(parsed_data.get("content", "")) < 50:
                 logger.warning("Synthesized English content seems very short.")
            return parsed_data
        else:
            logger.error("Failed to parse synthesis response.")
            return None
    except Exception as e:
        logger.error(f"Error during English synthesis API call: {e}", exc_info=True)
        return None


async def translate_synthesized_story(english_data: Dict) -> Optional[Dict]:
    """
    Translates a synthesized English story (headline, summary, content) into German.
    """
    if not translation_model or not translation_model_info: # Check if model was initialized
        logger.error("Translation LLM not initialized. Cannot translate story.")
        return None
    # ... (rest of the function is the same, using generation_config) ...
    if 'cluster_translation_german_prompt' not in prompts or not prompts['cluster_translation_german_prompt']:
         logger.error("cluster_translation_german_prompt not found or empty in prompts.yml.")
         return None
    if not all(k in english_data for k in ["headline", "summary", "content"]):
        logger.error("Missing required keys (headline, summary, content) in english_data for translation.")
        return None
    if not english_data['headline'] or not english_data['summary'] or not english_data['content']:
         logger.warning("One or more English fields (headline, summary, content) are empty. Translation might produce poor results or fail.")

    prompt = prompts['cluster_translation_german_prompt'].format(
        synthesized_english_headline=english_data['headline'],
        synthesized_english_summary=english_data['summary'],
        synthesized_english_body=english_data['content']
    )
    logger.info("Translating synthesized story to German...")
    logger.debug(f"Translation Prompt (first 500 chars): {prompt[:500]}...")

    try:
        # === ADD LOGGING BEFORE CALL ===
        logger.info(f"Calling generate_content on object of type: {type(translation_model)}")
        if hasattr(translation_model, 'model_name'):
             logger.info(f"Model name associated with object: {translation_model.model_name}")
        # ==============================

        response_obj = await asyncio.to_thread(
            translation_model.generate_content, # This is now the GenerativeModel instance
            contents=prompt,
            generation_config=types.GenerationConfig( # Correct parameter name
                temperature=0.2,
                max_output_tokens=8192,
            ),
        )
        raw_response = response_obj.text
        parsed_data = _parse_llm_json_response(raw_response, ["headline", "summary", "content"])

        if parsed_data:
            logger.info("Successfully translated synthesized story to German.")
            if not parsed_data.get("headline") or not parsed_data.get("summary") or not parsed_data.get("content"):
                logger.warning("Translated German story has empty essential fields.")
                return None
            if len(parsed_data.get("content", "")) < 50:
                 logger.warning("Translated German content seems very short.")
            return parsed_data
        else:
            logger.error("Failed to parse translation response.")
            return None
    except Exception as e:
        logger.error(f"Error during German translation API call: {e}", exc_info=True)
        return None
