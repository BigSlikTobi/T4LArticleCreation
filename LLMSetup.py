import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

def initialize_model(provider: str = "gemini"):
    if provider.lower() == "gemini":
        selected_model = "models/gemini-2.0-flash-thinking-exp-01-21"
        print("Initializing Gemini model:", selected_model)
        google_api_key = os.getenv("GEMINI_API_KEY")
        genai.configure(api_key=google_api_key)
        return {"provider": "gemini", "model_name": selected_model, "model": genai.GenerativeModel(selected_model)}
    else:
        raise ValueError("Only Gemini provider is supported")

if __name__ == "__main__":
    model = initialize_model("gemini")
