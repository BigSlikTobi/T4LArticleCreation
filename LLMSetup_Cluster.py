import os
import google.generativeai as genai # Changed import style
# from google.generativeai import types # No longer needed if not using tools
from dotenv import load_dotenv
import logging # Add logging

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')
logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)

google_api_key = os.getenv("GEMINI_API_KEY")
if not google_api_key:
    logger.critical("CRITICAL: GEMINI_API_KEY environment variable not set.")
    raise ValueError("CRITICAL: GEMINI_API_KEY environment variable not set.")
try:
    genai.configure(api_key=google_api_key)
    logger.info("Gemini API key configured globally.")
except Exception as e:
     logger.critical(f"CRITICAL: Failed to configure Gemini API key: {e}")
     raise

def initialize_model(provider: str, model_type: str = "default"):
    """
    Initialize a specific Gemini GenerativeModel instance.
    (Specifically for Cluster Pipeline - NO GROUNDING TOOL)

    Args:
        provider: The provider name. Only 'gemini' is supported.
        model_type: The type of Gemini model to use ("default", "lite", or "flash").

    Returns:
        A dictionary containing the model name and the initialized GenerativeModel object.
        Returns None if initialization fails.
    """
    if provider.lower() != "gemini":
        logger.error("Unsupported provider. Only 'gemini' is supported in this setup.")
        raise ValueError("Unsupported provider. Only 'gemini' is supported in this setup.")

    # Define model names based on type
    if model_type == "lite":
        selected_model_name = "gemini-2.5-flash-preview-04-17"
        logger.info("Using model: gemini-2.5-flash-preview-04-17 (for lite/flash type)")
    elif model_type == "flash":
        selected_model_name = "gemini-2.5-flash-preview-04-17"
        logger.info("Using model: gemini-2.5-flash-preview-04-17 (for flash type)")
    elif model_type == "default":
        selected_model_name = "gemini-2.5-flash-preview-04-17"
        logger.info("Using model: gemini-2.5-flash-preview-04-17 (for default type)")
    else:
        logger.error(f"Unsupported model type: {model_type}. Choose 'default', 'lite', or 'flash'.")
        raise ValueError("Unsupported model type. Choose 'default', 'lite', or 'flash'.")

    logger.info(f"Initializing Gemini model instance: {selected_model_name} (without explicit tools/grounding)")

    # --- REMOVED TOOL CONFIGURATION ---
    # tools_config = [types.Tool(google_search=types.GoogleSearch())]
    # grounding_enabled = True
    # logger.info(f"Configured tools (attempting with types.GoogleSearch): {tools_config}")
    tools_config = None # Explicitly set to None
    grounding_enabled = False # Indicate no explicit grounding tool
    # --- END REMOVAL ---

    try:
        # --- Create the GenerativeModel instance without tools ---
        model_instance = genai.GenerativeModel(
            model_name=selected_model_name
            # No tools argument passed
        )
        logger.info(f"Successfully initialized GenerativeModel instance for {selected_model_name}")

        logger.info(f"LLMSetup_Cluster: Returning model object of type: {type(model_instance)}")

        # --- ADJUSTED RETURN DICT ---
        return {
            "model_name": selected_model_name,
            "model": model_instance, # Return the specific model instance
            "grounding_enabled": grounding_enabled, # Indicate grounding tool status
            "tools": tools_config # Return None for tools
        }
        # --- END ADJUSTMENT ---
    except Exception as e:
        logger.error(f"Failed to initialize GenerativeModel for {selected_model_name}: {e}", exc_info=True)
        return None # Returning None allows calling code to check

if __name__ == "__main__":
    # Keep or remove the test block as needed
    print("Testing LLMSetup_Cluster initialization...")
    # ... (optional test code) ...
