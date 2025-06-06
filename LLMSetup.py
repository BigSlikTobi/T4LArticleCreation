#TODO: Grounding is not working with ligt model and not necessary for translatio

import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

def initialize_model(provider: str, model_type: str = "default", grounding_enabled: bool = True):
    """
    Initialize a Gemini model with optional Google Search grounding using the new client configuration.

    Args:
        provider (str): The provider name. Only 'gemini' is supported.
        model_type (str): The type of Gemini model to use ("default", "lite", or "flash").
        grounding_enabled (bool): Whether to enable Google Search grounding.

    Returns:
        dict: A dictionary containing the model name, the models object (client.models), a flag indicating if grounding is enabled, and the tools configuration.

    Raises:
        ValueError: If the provider or model_type is unsupported, or if the API key is missing.
    """
    if provider.lower() != "gemini":
        raise ValueError("Unsupported provider. Only 'gemini' is supported in this setup.")

    if model_type == "lite":
        selected_model = "gemini-2.0-flash-lite"
        print("Using lite model")
    elif model_type == "flash":
        selected_model = "gemini-2.0-flash"
        print("Using flash model")
    elif model_type == "default":
        selected_model = "gemini-2.5-flash-preview-05-20"
        print("Using default model")
    else:
        raise ValueError("Unsupported model type. Choose 'default', 'lite', or 'flash'.")

    print(f"Initializing Gemini model: {selected_model} with Google Search Grounding")
    google_api_key = os.getenv("GEMINI_API_KEY")
    if not google_api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set.")

    # Create the client with the API key (removing the configure call)
    client = genai.Client(api_key=google_api_key)

    # Configure grounding tool using the provided pattern
    if grounding_enabled:
        tools_config = [types.Tool(google_search=types.GoogleSearch())]
    else:
        tools_config = []

    # Return client.models so that usage like model.generate_content(...) works in your other files
    return {
        "model_name": selected_model,
        "model": client.models,
        "grounding_enabled": grounding_enabled,
        "tools": tools_config
    }

if __name__ == "__main__":
    try:
        model_info = initialize_model("gemini", "flash", grounding_enabled=True)
        print("\nModel Initialized:")
        print(f"  Model Name: {model_info['model_name']}")
        print(f"  Grounding Enabled: {model_info['grounding_enabled']}")
        print(f"  Model Object: {model_info['model']}")
    except Exception as e:
        print(f"Error during initialization test: {e}")
