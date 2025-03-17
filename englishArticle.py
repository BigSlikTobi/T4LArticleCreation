import asyncio
import json  # Added missing import
import google.generativeai as genai
import sys
import os
import yaml
from dotenv import load_dotenv
from LLMSetup import initialize_model

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables
load_dotenv()

# Initialize Gemini model using LLMSetup
model_info = initialize_model("gemini")
gemini_model = model_info["model"]

# Load prompts from YAML file
with open(os.path.join(os.path.dirname(__file__), "prompts.yml"), "r", encoding="utf-8") as f:
    prompts = yaml.safe_load(f)

async def generate_english_article(main_content: str, verbose: bool = False) -> dict:
    """
    Generates an English article with headline and structured content.
    Returns a dict with 'headline' and 'content'.
    If verbose is True, includes the raw Gemini response in the result under 'raw_response'.
    """
    prompt = f"""
{prompts['english_prompt']}

**Source Information:**
Main content (main_content) – the central story:
{main_content}

Please provide your answer strictly in the following JSON format without any additional text:
{{
  "headline": "<h1>Your generated headline</h1>",
  "content": "<div>Your structured article content as HTML, including <p>, <h2>, etc.</div>"
}}
"""
    try:
        # Generate content with increased max_output_tokens.
        response_obj = await asyncio.to_thread(
            gemini_model.generate_content,
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.7,
                max_output_tokens=8192,
            )
        )
        raw_response = response_obj.text
        if verbose:
            print("Raw Gemini response:")
            print(raw_response)
    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        raw_response = ""

    # Remove markdown code block markers if present.
    if (raw_response.strip().startswith("```")):
        lines = raw_response.strip().splitlines()
        # Remove the first line (```json) and the last line if it's a markdown fence.
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        raw_response = "\n".join(lines)
        
    json_start = raw_response.find("{")
    json_end = raw_response.rfind("}") + 1
    if (json_start != -1 and json_end != -1):
        raw_response_clean = raw_response[json_start:json_end]
    else:
        raw_response_clean = raw_response  # Fallback if markers are not found

    try:
        # Removed unnecessary newline replacement and quote stripping
        response_data = json.loads(raw_response_clean)
        
        # Build the result from the parsed data directly
        result = {
            "headline": response_data.get("headline", ""),
            "content": response_data.get("content", "")
        }
        if verbose:
            result["raw_response"] = raw_response_clean
        return result
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON response: {e}")
        # Fallback: Try to extract content using string manipulation
        try:
            headline_start = raw_response_clean.find('"headline": "') + 12
            headline_end = raw_response_clean.find('",', headline_start)
            content_start = raw_response_clean.find('"content": "') + 11
            content_end = raw_response_clean.rfind('"')
            
            headline = raw_response_clean[headline_start:headline_end]
            content = raw_response_clean[content_start:content_end]
            
            return {
                "headline": headline,
                "content": content,
                "raw_response": raw_response_clean if verbose else ""
            }
        except Exception as e:
            print(f"Fallback parsing failed: {e}")
            return {"headline": "", "content": "", "raw_response": raw_response_clean if verbose else ""}
    except Exception as e:
        print(f"Unknown error: {e}")
        return {"headline": "", "content": "", "raw_response": raw_response_clean if verbose else ""}