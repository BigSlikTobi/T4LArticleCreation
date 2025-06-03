import asyncio
import json
import google.generativeai as genai
from google.genai import types
import sys
import os
import re # Added re for more robust fallback parsing
import yaml
from dotenv import load_dotenv
from LLMSetup import initialize_model
from post_processing import remove_citations_from_text # Added import

# Add parent directory to Python path
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # Consider if still needed

# Load environment variables
load_dotenv()

# Initialize Gemini model using LLMSetup
# Ensure LLMSetup.py is in the Python path or same directory
try:
    model_info = initialize_model("gemini", "default", grounding_enabled=True) # Ensure grounding is as intended
    gemini_model = model_info["model"]
except Exception as e:
    print(f"CRITICAL: Failed to initialize Gemini model in germanArticle.py: {e}")
    gemini_model = None
    model_info = {"model_name": "unknown", "tools": []}


# Load prompts from YAML file
prompts_file_path = os.path.join(os.path.dirname(__file__), "prompts.yml")
try:
    with open(prompts_file_path, "r", encoding="utf-8") as f:
        prompts = yaml.safe_load(f)
except FileNotFoundError:
    print(f"CRITICAL: prompts.yml not found at {prompts_file_path}. German article generation will fail.")
    prompts = {"german_prompt": "Error: German prompt not loaded."} # Fallback
except Exception as e:
    print(f"CRITICAL: Error loading prompts.yml: {e}")
    prompts = {"german_prompt": "Error: German prompt not loaded."} # Fallback


async def generate_german_article(main_content: str, verbose: bool = False) -> dict:
    """
    Generates a German article with headline, summary, and structured content using a Gemini LLM.

    Args:
        main_content (str): JSON string with keys 'headline' and 'content', or raw content as fallback.
        verbose (bool): If True, includes the raw Gemini response under 'raw_response'.

    Returns:
        dict: Dictionary with keys 'headline', 'summary', 'content', and optionally 'raw_response'.

    Side Effects:
        Prints warnings and errors to stdout. Logs raw LLM responses if verbose is True.
    
    The main_content parameter is expected to be a JSON string with the format:
    {
      "headline": "headline text", # This is expected to be the *English* generated headline
      "content": "content text"   # This is expected to be the *English* generated content
    }
    """
    if not gemini_model:
        print("German Article Generation: Gemini model not initialized. Skipping generation.")
        return {"headline": "", "summary": "", "content": "", "raw_response": "Model not initialized"}

    # Parse main_content from JSON string
    try:
        content_obj = json.loads(main_content)
        # For German generation, 'headline' and 'content' in main_content are the *English* outputs
        original_english_headline = content_obj.get('headline', '') 
        original_english_content = content_obj.get('content', '')
    except (json.JSONDecodeError, TypeError):
        print("Warning (German Article): main_content was not valid JSON. Treating entire input as content (Source for German generation).")
        original_english_headline = "" # Or handle as an error
        original_english_content = main_content
    
    # Construct the prompt for German generation
    german_prompt_template = prompts.get('german_prompt', "Fallback prompt: Generate a German article.")
    if "Fallback prompt" in german_prompt_template:
        print("Warning (German Article): Using fallback German prompt.")

    # The 'Source Information' for the German prompt should be the English generated article
    # The german_prompt itself should instruct the LLM to translate/create a German version
    # based on the provided English headline and content.
    prompt = f"""
{german_prompt_template}

**Source Information (English Article to adapt/translate to German):**
Original English Headline: {original_english_headline}

Original English Content:
{original_english_content}

Please provide your answer strictly in the following JSON format without any additional text:
{{
  "headline": "<h1>Your generated German headline</h1>",
  "summary": "<p>Your generated German summary</p>",
  "content": "<div>Your structured German article content as HTML, including <p>, <h2>, etc.</div>"
}}
"""
    raw_response = "" # Initialize raw_response
    try:
        # Generate content
        response_obj = await asyncio.to_thread(
            gemini_model.generate_content,
            model=model_info["model_name"],
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.4,
                max_output_tokens=8192,
                tools=model_info.get("tools", [])
            )
        )
        if response_obj and hasattr(response_obj, 'text'):
            raw_response = response_obj.text
        else:
            print("Error (German Article): Gemini API response object or text attribute is missing.")
            raw_response = ""

        if verbose:
            print("Raw Gemini response (German Article):")
            print(raw_response)
            
    except Exception as e:
        print(f"Error calling Gemini API (German Article): {e}")
        raw_response = ""

    # --- Start of Corrected Markdown Cleaning Block ---
    if isinstance(raw_response, str) and raw_response.strip().startswith("```"):
        lines = raw_response.strip().splitlines()
        if lines and lines[0].startswith("```"): 
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        raw_response = "\n".join(lines)
    elif not isinstance(raw_response, str):
        print(f"Warning (German Article): raw_response was not a string. Type: {type(raw_response)}, Value: {raw_response}")
        if raw_response is None:
            raw_response = ""
    # --- End of Corrected Markdown Cleaning Block ---
        
    json_start = raw_response.find("{")
    json_end = raw_response.rfind("}") + 1
    if json_start != -1 and json_end != -1 and json_start < json_end:
        raw_response_clean = raw_response[json_start:json_end]
    else:
        print(f"Warning (German Article): Could not find valid JSON object boundaries in raw_response. Using raw_response as is. Length: {len(raw_response)}")
        raw_response_clean = raw_response

    try:
        response_data = json.loads(raw_response_clean)
        result = {
            "headline": remove_citations_from_text(response_data.get("headline", "")),
            "summary": remove_citations_from_text(response_data.get("summary", "")),
            "content": remove_citations_from_text(response_data.get("content", ""))
        }
        if verbose:
            result["raw_response"] = raw_response_clean
        return result
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON response (German Article): {e}. Response text was: '{raw_response_clean[:500]}...'")
        # Fallback parsing
        try:
            headline_text = ""
            summary_text = ""
            content_text = ""

            hl_match = re.search(r'"headline"\s*:\s*"(.*?)"', raw_response_clean, re.DOTALL)
            if hl_match: headline_text = hl_match.group(1).replace('\\"', '"').replace('\\n', '\n')
            
            sum_match = re.search(r'"summary"\s*:\s*"(.*?)"', raw_response_clean, re.DOTALL)
            if sum_match: summary_text = sum_match.group(1).replace('\\"', '"').replace('\\n', '\n')

            cont_match = re.search(r'"content"\s*:\s*"(.*)"\s*}', raw_response_clean, re.DOTALL)
            if cont_match: content_text = cont_match.group(1).replace('\\"', '"').replace('\\n', '\n')
            else:
                content_start_idx = raw_response_clean.find('"content": "') + 11
                content_end_idx = raw_response_clean.rfind('"') 
                if content_start_idx > 10 and content_end_idx > content_start_idx :
                     content_text = raw_response_clean[content_start_idx:content_end_idx].replace('\\"', '"').replace('\\n', '\n')

            return {
                "headline": remove_citations_from_text(headline_text),
                "summary": remove_citations_from_text(summary_text),
                "content": remove_citations_from_text(content_text),
                "raw_response": raw_response_clean if verbose else ""
            }
        except Exception as fallback_e:
            print(f"Fallback parsing failed (German Article): {fallback_e}")
            return {"headline": "", "summary": "", "content": "", "raw_response": raw_response_clean if verbose else ""}
    except Exception as unknown_e:
        print(f"Unknown error during final processing (German Article): {unknown_e}")
        return {"headline": "", "summary": "", "content": "", "raw_response": raw_response_clean if verbose else ""}

if __name__ == '__main__':
    # Example usage for testing
    async def test_german_generation():
        print("Testing German Article Generation...")
        # Mock main_content (which is the output from English generation)
        mock_english_article_json = json.dumps({
            "headline": "<h1>Test English Headline for German Translation</h1>",
            "summary": "<p>This is the English summary. It will be used as a basis for the German version.</p>",
            "content": "<div><p>This is the main English content paragraph. It contains details about a fictional sports event. The German generation should adapt this. Player Z made a fantastic play.</p><h2>More Details</h2><p>Further English details are provided here.</p></div>"
        })
        
        if 'german_prompt' not in prompts or "Error" in prompts['german_prompt']:
            print("Cannot run test: 'german_prompt' is missing or failed to load from prompts.yml.")
            return

        article_result = await generate_german_article(mock_english_article_json, verbose=True)
        print("\n--- Generated German Article ---")
        print(f"Headline: {article_result.get('headline')}")
        print(f"Summary: {article_result.get('summary')}")
        print(f"Content Snippet: {article_result.get('content', '')[:200]}...")
        if "raw_response" in article_result:
            print(f"Raw LLM JSON Output: {article_result['raw_response']}")
        print("--- End of Test ---")

    asyncio.run(test_german_generation())