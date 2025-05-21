import asyncio
import json
import os
import yaml
import re
from typing import Dict, Optional, List

from LLMSetup import initialize_model
from post_processing import remove_citations_from_text # If needed for translated content
import database # To interact with the database for translations
from google.genai import types as genai_types # For GenerateContentConfig
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load prompts from YAML file
PROMPTS_FILE_PATH = os.path.join(os.path.dirname(__file__), "prompts.yml")
try:
    with open(PROMPTS_FILE_PATH, "r", encoding="utf-8") as f:
        prompts = yaml.safe_load(f)
    # Remove loading of the old 'multi_source_synthesis_translation' prompt
    # if not prompts or 'multi_source_synthesis_translation' not in prompts:
    #     raise ValueError("CRITICAL: 'multi_source_synthesis_translation' not found in prompts.yml.")
    # translation_prompt_template = prompts['multi_source_synthesis_translation'] # Keep for now, or remove if fully unused
    
    # Load the new prompt for individual component translation
    if 'translate_text_component' not in prompts:
        raise ValueError("CRITICAL: 'translate_text_component' not found in prompts.yml. This is needed for component-wise translation.")
    component_translation_prompt_template = prompts['translate_text_component']

except FileNotFoundError:
    print(f"CRITICAL: {PROMPTS_FILE_PATH} not found. Translation will fail.")
    # translation_prompt_template = "Error: Prompts file not found." # Remove reference
    component_translation_prompt_template = "Error: Prompts file not found." # Ensure this is also set
except ValueError as ve:
    print(ve)
    # translation_prompt_template = "Error: Prompt key missing." # Remove reference
    component_translation_prompt_template = "Error: Prompt key missing." # Ensure this is also set
except Exception as e:
    print(f"CRITICAL: Error loading prompts.yml: {e}")
    # translation_prompt_template = "Error: Could not load prompts." # Remove reference
    component_translation_prompt_template = "Error: Could not load prompts." # Ensure this is also set

# Initialize Gemini model - using "flash" as requested for translation
# Grounding is generally not needed for pure translation tasks.
try:
    # Using "default" model instead of "flash" for translation
    model_info = initialize_model("gemini", "flash", grounding_enabled=False)
    gemini_translator_model = model_info["model"]
    print(f"Cluster Translator: Initialized Gemini model {model_info['model_name']} for translation.")
except Exception as e:
    print(f"CRITICAL: Failed to initialize Gemini model in cluster_translator.py: {e}")
    gemini_translator_model = None
    model_info = {"model_name": "unknown", "tools": []}

# Mapping language codes to full names for the prompt
LANGUAGE_NAME_MAP = {
    "de": "German",
    "es": "Spanish",
    "fr": "French",
    # Add more as needed
}

async def _translate_single_component(
    text_to_translate: str,
    target_language_name: str,
    target_language_code: str,
    component_name: str, # For logging
    max_tokens: int,
    prompt_template: str
) -> Optional[str]:
    """Helper function to translate a single text component."""
    if not text_to_translate:
        logging.info(f"Component '{component_name}' is empty, skipping translation.")
        return ""

    prompt = prompt_template.format(
        language_name=target_language_name,
        language_code=target_language_code,
        text_to_translate=text_to_translate
    )
    logging.info(f"Translating component '{component_name}' to {target_language_name}. Prompt length: {len(prompt)}, Max tokens: {max_tokens}")

    try:
        response_obj = await asyncio.to_thread(
            gemini_translator_model.generate_content,
            model=model_info["model_name"],
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=max_tokens,
            )
        )
        if response_obj and hasattr(response_obj, 'text'):
            translated_text = response_obj.text
            logging.info(f"Successfully translated component '{component_name}'. Response length: {len(translated_text)}")
            # Remove markdown code block fences if present
            if translated_text.startswith("```html"):
                translated_text = translated_text[len("```html"):]
            if translated_text.startswith("```"):
                 translated_text = translated_text[len("```"):]
            if translated_text.endswith("```"):
                translated_text = translated_text[:-len("```")]
            return translated_text.strip()
        else:
            logging.error(f"Error translating component '{component_name}': Gemini API response missing text.")
            return None
    except Exception as e:
        logging.error(f"Error calling Gemini API for component '{component_name}' translation to {target_language_name}: {e}")
        return None

async def translate_article_components(
    english_headline: str,
    english_summary: str,
    english_content: str,
    target_language_code: str,
    target_language_name: Optional[str] = None
) -> Optional[Dict]:
    """
    Translates article components (headline, summary, content) to the target language
    by translating each component separately and concurrently.
    """
    if not gemini_translator_model or "Error:" in component_translation_prompt_template: # Check new prompt
        print("Cluster Translator: Model not initialized or component prompt error. Skipping translation.")
        return None

    if not target_language_name:
        target_language_name = LANGUAGE_NAME_MAP.get(target_language_code.lower())
        if not target_language_name:
            print(f"Error: Language name not found for code '{target_language_code}'. Cannot translate.")
            return None

    # Define max tokens for each component
    # These might need adjustment based on typical content lengths
    headline_max_tokens = 512
    summary_max_tokens = 2048 
    content_max_tokens = 8192 # Keep high for content, but input is now smaller

    try:
        # Translate components concurrently
        results = await asyncio.gather(
            _translate_single_component(english_headline, target_language_name, target_language_code, "headline", headline_max_tokens, component_translation_prompt_template),
            _translate_single_component(english_summary, target_language_name, target_language_code, "summary", summary_max_tokens, component_translation_prompt_template),
            _translate_single_component(english_content, target_language_name, target_language_code, "content", content_max_tokens, component_translation_prompt_template)
        )
    except Exception as e:
        logging.error(f"Error during concurrent component translation: {e}")
        return None

    translated_headline, translated_summary, translated_content = results

    # Check if any component failed
    if translated_headline is None or translated_summary is None or translated_content is None:
        logging.error(f"One or more components failed to translate for {target_language_name}. Aborting.")
        return None

    # Apply post-processing like citation removal
    final_headline = remove_citations_from_text(translated_headline)
    final_summary = remove_citations_from_text(translated_summary)
    final_content = remove_citations_from_text(translated_content)
    
    logging.info(f"Successfully translated all components for {target_language_name}.")

    return {
        "translated_headline": final_headline,
        "translated_summary": final_summary,
        "translated_content": final_content,
    }

async def process_and_store_translation(
    cluster_article_id: str, # UUID as string
    english_data: Dict, # Contains 'headline', 'summary', 'content'
    target_language_code: str
):
    """
    Orchestrates translation for a single article and language, then stores it.
    """
    logger = logging.getLogger(__name__) # Get logger for this specific function call
    logger.info(f"Attempting translation for cluster_article_id {cluster_article_id} to {target_language_code}.")

    existing_translation = await database.get_cluster_article_translation(cluster_article_id, target_language_code)
    if existing_translation:
        logger.info(f"Translation for cluster_article_id {cluster_article_id} to {target_language_code} already exists. Skipping.")
        return

    translated_components = await translate_article_components(
        english_headline=english_data.get("headline", ""),
        english_summary=english_data.get("summary", ""),
        english_content=english_data.get("content", ""),
        target_language_code=target_language_code
    )

    if translated_components:
        success = await database.insert_cluster_article_translation(
            cluster_article_id=cluster_article_id,
            language_code=target_language_code,
            translated_data=translated_components
        )
        if success:
            logger.info(f"Successfully translated and stored for cluster_article_id {cluster_article_id} to {target_language_code}.")
        else:
            logger.error(f"Failed to store translation for cluster_article_id {cluster_article_id} to {target_language_code}.")
    else:
        logger.error(f"Translation failed for cluster_article_id {cluster_article_id} to {target_language_code}.")


async def translate_untranslated_cluster_articles(target_languages: Optional[List[str]] = None):
    """
    Standalone function to find and translate cluster articles that are missing translations
    for the specified languages.
    """
    logger = logging.getLogger(__name__)
    if target_languages is None:
        target_languages = ["de"] # Default to German if no languages specified

    logger.info(f"Starting standalone translation process for languages: {target_languages}")

    for lang_code in target_languages:
        logger.info(f"--- Processing language: {LANGUAGE_NAME_MAP.get(lang_code, lang_code)} ({lang_code}) ---")
        articles_to_translate = await database.get_cluster_articles_needing_translation(lang_code)
        
        if not articles_to_translate:
            logger.info(f"No articles found needing translation to {lang_code}.")
            continue

        logger.info(f"Found {len(articles_to_translate)} articles needing translation to {lang_code}.")
        for i, article_data in enumerate(articles_to_translate):
            cluster_article_db_id = str(article_data.get("id")) # This is cluster_articles.id
            logger.info(f"Translating article {i+1}/{len(articles_to_translate)}: ID {cluster_article_db_id} to {lang_code}")
            
            # The article_data fetched by get_cluster_articles_needing_translation
            # should contain the English headline, summary, content from cluster_articles table.
            await process_and_store_translation(
                cluster_article_id=cluster_article_db_id,
                english_data=article_data, # Pass the whole dict containing eng versions
                target_language_code=lang_code
            )
            if i < len(articles_to_translate) - 1:
                logger.info("Waiting 2 seconds before next article translation...")
                await asyncio.sleep(2) # Small delay

    logger.info("Standalone translation process finished.")


if __name__ == '__main__':
    # Example for standalone execution:
    # This will attempt to translate all cluster articles that don't have German or Spanish translations.
    
    # Basic check for readiness
    db_ready = database.supabase is not None
    llm_ready = (
        gemini_translator_model is not None and
        "Error:" not in component_translation_prompt_template # Check the correct prompt template
    )

    if db_ready and llm_ready:
        # To run for specific languages:
        # asyncio.run(translate_untranslated_cluster_articles(target_languages=["de", "es"]))
        
        # Or run with default (German)
        print("Running standalone translation for default language (German)...")
        asyncio.run(translate_untranslated_cluster_articles())
    else:
        if not db_ready: print("CRITICAL: Database client not initialized in cluster_translator.")
        if not llm_ready: print("CRITICAL: LLM model or prompts not initialized correctly in cluster_translator.")
        print("Halting standalone translation test.")