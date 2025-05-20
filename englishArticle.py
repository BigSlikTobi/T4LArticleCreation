import asyncio
import json
import google.generativeai as genai
from google.genai import types
import sys
import os
import yaml
from dotenv import load_dotenv
from LLMSetup import initialize_model
from post_processing import remove_citations_from_text # Added import

# Add parent directory to Python path
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # Consider if this is still needed

# Load environment variables
load_dotenv()

# Initialize Gemini model using LLMSetup
# Ensure LLMSetup.py is in the Python path or same directory
try:
    model_info = initialize_model("gemini", "default", grounding_enabled=True) # Ensure grounding is as intended
    gemini_model = model_info["model"]
except Exception as e:
    print(f"CRITICAL: Failed to initialize Gemini model in englishArticle.py: {e}")
    # Depending on your application, you might want to exit or raise the error further
    # For now, we'll let it proceed and fail later if gemini_model is not set.
    gemini_model = None
    model_info = {"model_name": "unknown", "tools": []}


# Load prompts from YAML file
prompts_file_path = os.path.join(os.path.dirname(__file__), "prompts.yml")
try:
    with open(prompts_file_path, "r", encoding="utf-8") as f:
        prompts = yaml.safe_load(f)
except FileNotFoundError:
    print(f"CRITICAL: prompts.yml not found at {prompts_file_path}. English article generation will fail.")
    prompts = {"english_prompt": "Error: English prompt not loaded."} # Fallback
except Exception as e:
    print(f"CRITICAL: Error loading prompts.yml: {e}")
    prompts = {"english_prompt": "Error: English prompt not loaded."} # Fallback


async def generate_english_article(main_content: str, verbose: bool = False) -> dict:
    """
    Generates an English article with headline, summary, and structured content.
    Returns a dict with 'headline', 'summary', and 'content'.
    If verbose is True, includes the raw Gemini response in the result under 'raw_response'.
    
    The main_content parameter is expected to be a JSON string with the format:
    {
      "headline": "headline text",
      "content": "content text"
    }
    """
    if not gemini_model:
        print("English Article Generation: Gemini model not initialized. Skipping generation.")
        return {"headline": "", "summary": "", "content": "", "raw_response": "Model not initialized"}

    # Parse main_content from JSON string
    try:
        content_obj = json.loads(main_content)
        original_headline_from_input = content_obj.get('headline', '')
        content_from_input = content_obj.get('content', '')
    except (json.JSONDecodeError, TypeError):
        # Fallback if main_content is not valid JSON - treat whole input as content
        print("Warning (English Article): main_content was not valid JSON. Treating entire input as content.")
        original_headline_from_input = ""
        content_from_input = main_content
    
    # Construct the prompt
    # Ensure prompts['english_prompt'] is available
    english_prompt_template = prompts.get('english_prompt', "Fallback prompt: Generate an English article.")
    if "Fallback prompt" in english_prompt_template:
        print("Warning (English Article): Using fallback English prompt.")

    prompt = f"""
{english_prompt_template}

**Source Information:**
Original headline: {original_headline_from_input}

Main content – the central story:
{content_from_input}

Please provide your answer strictly in the following JSON format without any additional text:
{{
  "headline": "<h1>Your generated headline</h1>",
  "summary": "<p>Your generated summary</p>",
  "content": "<div>Your structured article content as HTML, including <p>, <h2>, etc.</div>"
}}
"""
    raw_response = "" # Initialize raw_response
    try:
        # Generate content with increased max_output_tokens using the grounding tool.
        response_obj = await asyncio.to_thread(
            gemini_model.generate_content,
            model=model_info["model_name"],
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.4,
                max_output_tokens=8192, # Make sure this is a supported value
                tools=model_info.get("tools", []) # Use .get for safety
            )
        )
        # It's good practice to check if response_obj and response_obj.text exist
        if response_obj and hasattr(response_obj, 'text'):
            raw_response = response_obj.text
        else:
            print("Error (English Article): Gemini API response object or text attribute is missing.")
            raw_response = "" # Ensure raw_response is a string

        if verbose:
            print("Raw Gemini response (English Article):")
            print(raw_response)
            
    except Exception as e:
        print(f"Error calling Gemini API (English Article): {e}")
        raw_response = "" # Ensure raw_response is a string on error

    # --- Start of Corrected Markdown Cleaning Block ---
    # Ensure raw_response is a string before stripping and splitting
    if isinstance(raw_response, str) and raw_response.strip().startswith("```"):
        lines = raw_response.strip().splitlines()
        # Check if the list is not empty and then check the first line
        if lines and lines.startswith("```"): 
            lines = lines[1:]
        # Check if the list is not empty and then check the last line
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        raw_response = "\n".join(lines)
    elif not isinstance(raw_response, str):
        print(f"Warning (English Article): raw_response was not a string. Type: {type(raw_response)}, Value: {raw_response}")
        if raw_response is None:
            raw_response = "" 
    # --- End of Corrected Markdown Cleaning Block ---
        
    json_start = raw_response.find("{")
    json_end = raw_response.rfind("}") + 1
    if json_start != -1 and json_end != -1 and json_start < json_end: # Ensure valid slice
        raw_response_clean = raw_response[json_start:json_end]
    else:
        print(f"Warning (English Article): Could not find valid JSON object boundaries in raw_response. Using raw_response as is. Length: {len(raw_response)}")
        raw_response_clean = raw_response # Fallback if markers are not found or invalid

    try:
        response_data = json.loads(raw_response_clean)
        result = {
            "headline": remove_citations_from_text(response_data.get("headline", "")),
            "summary": remove_citations_from_text(response_data.get("summary", "")),
            "content": remove_citations_from_text(response_data.get("content", ""))
        }
        if verbose:
            result["raw_response"] = raw_response_clean # Store the cleaned JSON string
        return result
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON response (English Article): {e}. Response text was: '{raw_response_clean[:500]}...'")
        # Fallback parsing (less reliable but can be a last resort)
        try:
            # Be careful with find, ensure indices are valid before slicing
            headline_text = ""
            summary_text = ""
            content_text = ""

            hl_match = re.search(r'"headline"\s*:\s*"(.*?)"', raw_response_clean, re.DOTALL)
            if hl_match: headline_text = hl_match.group(1).replace('\\"', '"').replace('\\n', '\n')
            
            sum_match = re.search(r'"summary"\s*:\s*"(.*?)"', raw_response_clean, re.DOTALL)
            if sum_match: summary_text = sum_match.group(1).replace('\\"', '"').replace('\\n', '\n')

            # Content can be tricky due to nested quotes/HTML. This is a simple attempt.
            cont_match = re.search(r'"content"\s*:\s*"(.*)"\s*}', raw_response_clean, re.DOTALL)
            if cont_match: content_text = cont_match.group(1).replace('\\"', '"').replace('\\n', '\n')
            else: # Simpler find if the regex fails due to complexity
                content_start_idx = raw_response_clean.find('"content": "') + 11
                content_end_idx = raw_response_clean.rfind('"') # This might be too greedy
                if content_start_idx > 10 and content_end_idx > content_start_idx :
                     content_text = raw_response_clean[content_start_idx:content_end_idx].replace('\\"', '"').replace('\\n', '\n')


            return {
                "headline": remove_citations_from_text(headline_text),
                "summary": remove_citations_from_text(summary_text),
                "content": remove_citations_from_text(content_text),
                "raw_response": raw_response_clean if verbose else ""
            }
        except Exception as fallback_e:
            print(f"Fallback parsing failed (English Article): {fallback_e}")
            return {"headline": "", "summary": "", "content": "", "raw_response": raw_response_clean if verbose else ""}
    except Exception as unknown_e:
        print(f"Unknown error during final processing (English Article): {unknown_e}")
        return {"headline": "", "summary": "", "content": "", "raw_response": raw_response_clean if verbose else ""}

if __name__ == '__main__':
    # Example usage for testing
    async def test_generation():
        print("Testing English Article Generation...")
        # Mock main_content
        mock_main_content_json = json.dumps({
            "headline": "Sample Test Headline",
            "content": "This is some sample content for testing the English article generation. It should be long enough to produce a decent article. Let's see how the LLM handles this. Adding some more words to meet length requirements if any. The quick brown fox jumps over the lazy dog. Player X scored a goal."
        })
        
        # Ensure your prompts.yml has 'english_prompt' defined
        if 'english_prompt' not in prompts or "Error" in prompts['english_prompt']:
            print("Cannot run test: 'english_prompt' is missing or failed to load from prompts.yml.")
            return

        article_result = await generate_english_article(mock_main_content_json, verbose=True)
        print("\n--- Generated English Article ---")
        print(f"Headline: {article_result.get('headline')}")
        print(f"Summary: {article_result.get('summary')}")
        print(f"Content Snippet: {article_result.get('content', '')[:200]}...")
        if "raw_response" in article_result:
            print(f"Raw LLM JSON Output: {article_result['raw_response']}")
        print("--- End of Test ---")

    asyncio.run(test_generation())