import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

def initialize_model(provider: str = "gemini", model_type: str = "default"):
    """
    Initialize a Gemini model with specified configuration
    
    Args:
        provider: The model provider (currently only "gemini" supported)
        model_type: The type of model to use ("default", "lite", or "flash")
        
    Returns:
        Dictionary containing model configuration
    """
    if provider.lower() == "gemini":
        # Select model based on type
        if model_type == "lite":
            selected_model = "models/gemini-2.0-flash-lite"
        elif model_type == "flash":
            selected_model = "models/gemini-2.0-flash-thinking-exp-01-21"
        else:
            selected_model = "models/gemini-2.0-flash-thinking-exp-01-21"  # default to flash
        
        print("Initializing Gemini model:", selected_model)
        google_api_key = os.getenv("GEMINI_API_KEY")
        genai.configure(api_key=google_api_key)
        return {
            "provider": "gemini",
            "model_name": selected_model,
            "model": genai.GenerativeModel(selected_model)
        }
    else:
        raise ValueError("Only Gemini provider is supported")

if __name__ == "__main__":
    model = initialize_model("gemini", "flash")
