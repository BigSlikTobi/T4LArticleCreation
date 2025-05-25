import asyncio
import json
import os
import yaml
import re
from typing import List, Dict, Optional

from LLMSetup import initialize_model
from post_processing import remove_citations_from_text
from google.genai import types # For GenerateContentConfig

# Load prompts from YAML file
PROMPTS_FILE_PATH = os.path.join(os.path.dirname(__file__), "prompts.yml")
try:
    with open(PROMPTS_FILE_PATH, "r", encoding="utf-8") as f:
        prompts = yaml.safe_load(f)
    if not prompts or 'multi_source_synthesis_prompt' not in prompts:
        raise ValueError("CRITICAL: 'multi_source_synthesis_prompt' not found in prompts.yml.")
    multi_source_synthesis_prompt_template = prompts['multi_source_synthesis_prompt']
except FileNotFoundError:
    print(f"CRITICAL: {PROMPTS_FILE_PATH} not found. Cluster article generation will fail.")
    multi_source_synthesis_prompt_template = "Error: Prompts file not found." # Fallback
except ValueError as ve:
    print(ve)
    multi_source_synthesis_prompt_template = "Error: Prompt key missing." # Fallback
except Exception as e:
    print(f"CRITICAL: Error loading prompts.yml: {e}")
    multi_source_synthesis_prompt_template = "Error: Could not load prompts." # Fallback

# Initialize Gemini model
try:
    model_info = initialize_model("gemini", "default", grounding_enabled=True)
    gemini_model = model_info["model"] # This is client.models
    print(f"Cluster Article Generator: Initialized Gemini model {model_info['model_name']} with grounding {'enabled' if model_info['grounding_enabled'] else 'disabled'}.")
except Exception as e:
    print(f"CRITICAL: Failed to initialize Gemini model in cluster_article_generator.py: {e}")
    gemini_model = None
    model_info = {"model_name": "unknown", "tools": []}


def format_source_articles_for_prompt(
    source_articles_data: List[Dict],
    previous_source_ids: Optional[List[int]] = None
) -> str:
    """
    Formats the list of source articles for inclusion in the LLM prompt.
    Marks articles as (NEW INFORMATION) if they were not in previous_source_ids.
    """
    formatted_articles = []
    previous_source_ids_set = set(previous_source_ids) if previous_source_ids else set()
    
    print(f"DEBUG: Formatting {len(source_articles_data)} source articles for prompt")
    if source_articles_data:
        print(f"DEBUG: First article keys: {list(source_articles_data[0].keys())}")
        print(f"DEBUG: First article ID type: {type(source_articles_data[0].get('id', 'NO_ID'))}")

    for article in source_articles_data:
        content = article.get('Content', article.get('content', '')) 
        headline = article.get('headline', '')
        created_at = article.get('created_at', 'N/A')
        article_id = article.get('id')

        is_new_info_marker = ""
        if previous_source_ids is not None: 
            if article_id not in previous_source_ids_set:
                is_new_info_marker = " (NEW INFORMATION)"
        
        formatted_article = (
            f"--- Source Article (Published: {created_at}){is_new_info_marker} ---\n"
            f"Headline: {headline}\n"
            f"Content: {content}\n"
            f"--- End Source Article ---"
        )
        formatted_articles.append(formatted_article)
    
    return "\n\n".join(formatted_articles)


async def generate_cluster_article(
    source_articles_data: List[Dict],
    previous_combined_article_data: Optional[Dict] = None,
    language: str = "English" 
) -> Dict:
    """
    Generates a synthesized article from a cluster of source articles using an LLM.
    """
    if not gemini_model or "Error:" in multi_source_synthesis_prompt_template:
        print("Cluster Article Generator: Model not initialized or prompt error. Skipping generation.")
        return {}

    previous_source_ids = None
    previous_article_prompt_str = ""
    if previous_combined_article_data:
        previous_source_ids = previous_combined_article_data.get('source_article_ids', [])
        prev_headline = previous_combined_article_data.get('headline', '') or ""
        prev_summary = previous_combined_article_data.get('summary', '') or ""
        prev_content = previous_combined_article_data.get('content', '') or ""

        previous_article_prompt_str = (
            "\n\n**Previous Combined Article (to be updated):**\n"
            f"Headline: {prev_headline}\n"
            f"Summary: {prev_summary}\n"
            f"Content:\n{prev_content}\n"
            "--- End Previous Combined Article ---\n"
        )
    
    formatted_sources_str = format_source_articles_for_prompt(source_articles_data, previous_source_ids)
    current_prompt_template = multi_source_synthesis_prompt_template.replace("[English/German]", language)
    prompt_input_content = f"{formatted_sources_str}{previous_article_prompt_str}"

    try:
        parts = current_prompt_template.split("Provide your response strictly in the following JSON format.", 1)
        if len(parts) == 2:
            main_instruction_part = parts[0]
            json_format_part = "Provide your response strictly in the following JSON format." + parts[1]
            full_prompt = f"{main_instruction_part}\n\n{prompt_input_content}\n\n{json_format_part}"
        else:
            print("Warning: Could not reliably split prompt template. Using a simpler concatenation.")
            full_prompt = f"{current_prompt_template}\n\n**Input Data:**\n{prompt_input_content}"
    except Exception as e_prompt_format:
        print(f"Error formatting prompt: {e_prompt_format}. Using basic prompt structure.")
        full_prompt = f"{current_prompt_template}\n\n{prompt_input_content}"

    raw_response_text = ""
    try:
        print(f"Cluster Article Generator: Calling Gemini for synthesis. Input content length (approx sources + prev): {len(prompt_input_content)}")
        print(f"DEBUG: Full prompt length: {len(full_prompt)} characters")
        
        # Corrected API call: tools go inside GenerateContentConfig
        response_obj = await asyncio.to_thread(
            gemini_model.generate_content,  # gemini_model is client.models
            model=model_info["model_name"], # Specifies which model from client.models to use
            contents=full_prompt,
            config=types.GenerateContentConfig( 
                temperature=0.3, 
                max_output_tokens=8192,
                tools=model_info.get("tools", []) # Moved tools inside GenerateContentConfig
            )
            # No top-level tools argument here
        )

        if response_obj and hasattr(response_obj, 'text'):
            raw_response_text = response_obj.text
            print(f"DEBUG: Raw response length: {len(raw_response_text)} characters")
            print(f"DEBUG: Response ends with: '{raw_response_text[-100:]}'")
        else:
            print("Error (Cluster Article): Gemini API response object or text attribute is missing.")
            raw_response_text = ""
        
    except Exception as e:
        print(f"Error calling Gemini API for cluster synthesis: {e}")
        raw_response_text = f'{{"error": "API call failed: {str(e)}"}}'

    if isinstance(raw_response_text, str) and raw_response_text.strip().startswith("```"):
        lines = raw_response_text.strip().splitlines()
        if lines and lines[0].lower().startswith(("```json", "```")):
            lines = lines[1:]
        if lines and lines[-1] == "```":
            lines = lines[:-1]
        cleaned_response_str = "\n".join(lines).strip()
    else:
        cleaned_response_str = raw_response_text.strip()

    # Enhanced JSON boundary detection
    json_start = cleaned_response_str.find("{")
    json_end = cleaned_response_str.rfind("}") + 1

    if json_start != -1 and json_end > json_start:
        json_str = cleaned_response_str[json_start:json_end]
    else:
        print(f"Warning (Cluster Article): Could not find valid JSON object boundaries in response. Raw: {cleaned_response_str[:500]}")
        # Try multiple strategies to find JSON-like content
        strategies = [
            # Strategy 1: Look for any JSON object
            r'\{[\s\S]*\}',
            # Strategy 2: Look for JSON starting with expected fields
            r'\{\s*"(?:headline|title|summary|content)"[\s\S]*',
            # Strategy 3: Look for partial JSON content
            r'\{[^{}]*"(?:headline|title|summary|content)"[^{}]*',
        ]
        
        json_str = cleaned_response_str
        for strategy in strategies:
            match = re.search(strategy, cleaned_response_str, re.MULTILINE | re.DOTALL)
            if match:
                json_str = match.group(0)
                print(f"Found JSON using strategy: {strategy[:30]}...")
                break 

    try:
        data = json.loads(json_str)
        return {
            "headline": remove_citations_from_text(data.get("headline", "")),
            "summary": remove_citations_from_text(data.get("summary", "")),
            "content": remove_citations_from_text(data.get("content", "")),
            "raw_response_text": raw_response_text 
        }
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON response from cluster synthesis: {e}. Response: {json_str[:500]}...")
        
        # Enhanced fallback parsing with multiple strategies
        def robust_extract_field(text: str, field_name: str) -> str:
            """Extract field using multiple regex patterns for robustness"""
            patterns = [
                # Standard quoted field
                rf'"{field_name}"\s*:\s*"(.*?)"(?=\s*[,\}}])',
                # Field with escaped quotes
                rf'"{field_name}"\s*:\s*"((?:[^"\\]|\\.)*)"\s*[,\}}]',
                # Field at end of truncated JSON
                rf'"{field_name}"\s*:\s*"((?:[^"\\]|\\.)*?)(?:"|\s*$)',
                # Field with unescaped newlines (fallback)
                rf'"{field_name}"\s*:\s*"([^"]*)"',
                # Field without closing quote (truncated)
                rf'"{field_name}"\s*:\s*"([^"]*?)(?:\s*$)',
            ]
            
            for pattern in patterns:
                match = re.search(pattern, text, re.DOTALL | re.MULTILINE)
                if match:
                    result = match.group(1)
                    # Clean up common escape sequences
                    result = result.replace('\\"', '"').replace('\\n', '\n').replace('\\\\', '\\')
                    return result
            return ""

        try:
            # Try progressive JSON parsing for partial content
            progressive_json = None
            if json_str.strip().endswith(','):
                # Attempt to close truncated JSON
                test_json = json_str.rstrip(', \t\n') + '}'
                try:
                    progressive_json = json.loads(test_json)
                except:
                    pass
            
            if progressive_json:
                return {
                    "headline": remove_citations_from_text(progressive_json.get("headline", "")),
                    "summary": remove_citations_from_text(progressive_json.get("summary", "")),
                    "content": remove_citations_from_text(progressive_json.get("content", "")),
                    "raw_response_text": raw_response_text, 
                    "parsing_error": f"JSONDecodeError, used progressive parsing. Original error: {str(e)}"
                }
            
            # Fallback to regex extraction
            headline = robust_extract_field(json_str, "headline")
            summary = robust_extract_field(json_str, "summary") 
            content = robust_extract_field(json_str, "content")
            
            # Validate we got meaningful content
            if not any([headline, summary, content]):
                # Try alternative field names or structures
                headline = robust_extract_field(json_str, "title") or robust_extract_field(json_str, "headline")
                summary = robust_extract_field(json_str, "abstract") or robust_extract_field(json_str, "summary")
                content = robust_extract_field(json_str, "article") or robust_extract_field(json_str, "content") or robust_extract_field(json_str, "body")

            return {
                "headline": remove_citations_from_text(headline),
                "summary": remove_citations_from_text(summary),
                "content": remove_citations_from_text(content),
                "raw_response_text": raw_response_text, 
                "parsing_error": f"JSONDecodeError, used enhanced fallback. Original error: {str(e)}"
            }
        except Exception as fallback_e:
            print(f"Enhanced fallback JSON parsing also failed: {fallback_e}")
            # Last resort: try to extract any meaningful text
            text_content = re.sub(r'[{}",:]+', ' ', json_str)
            text_content = ' '.join(text_content.split())
            
            return {
                "headline": "Content extraction failed", 
                "summary": "Unable to parse response",
                "content": text_content[:500] if text_content else "No content extracted",
                "raw_response_text": raw_response_text, 
                "parsing_error": f"All parsing failed. Original: {str(e)}, Fallback: {str(fallback_e)}"
            }
    except Exception as ex:
        print(f"An unexpected error occurred during cluster article processing: {ex}")
        return {"raw_response_text": raw_response_text, "processing_error": str(ex)}


if __name__ == '__main__':
    async def test_cluster_generation():
        print("Testing Cluster Article Generation...")

        mock_source_articles = [
            {
                "id": 101,
                "headline": "Initial Report: Event X Kicks Off",
                "Content": "<p>Event X started today with much fanfare. Participants gathered early.</p>", 
                "created_at": "2023-01-01T10:00:00Z"
            },
            {
                "id": 102,
                "headline": "Update: Key Speaker Announced for Event X",
                "Content": "<p>A surprising announcement today: Jane Doe will be the keynote speaker at Event X. This changes expectations significantly.</p>", 
                "created_at": "2023-01-01T14:00:00Z"
            },
            {
                "id": 103,
                "headline": "Event X Concludes: Success Declared",
                "Content": "<p>Event X wrapped up this evening. Organizers declared it a major success, with record attendance and positive feedback on Jane Doe's speech.</p>", 
                "created_at": "2023-01-02T18:00:00Z"
            }
        ]

        print("\n--- Test 1: New Article Synthesis ---")
        synthesized_article_new = await generate_cluster_article(mock_source_articles)
        print(f"Headline: {synthesized_article_new.get('headline')}")
        print(f"Summary: {synthesized_article_new.get('summary')}")
        print(f"Content Snippet: {str(synthesized_article_new.get('content', ''))[:200]}...")
        if 'parsing_error' in synthesized_article_new:
            print(f"Parsing Error: {synthesized_article_new['parsing_error']}")

        mock_previous_article = {
            "headline": "<h1>Event X Underway, Key Speaker Jane Doe Confirmed</h1>",
            "summary": "<p>Event X has begun, and Jane Doe's participation as keynote speaker has generated excitement.</p>",
            "content": "<div><p>Event X commenced as scheduled. Early reports confirmed Jane Doe as a speaker, adding to the event's buzz.</p></div>",
            "source_article_ids": [101, 102] 
        }
        
        mock_new_source_for_update = [
             { 
                "id": 100, 
                "headline": "Pre-Event Buzz for Event X", 
                "Content": "<p>Anticipation is building for Event X, scheduled to start tomorrow. Rumors about speakers circulate.</p>",
                "created_at": "2022-12-31T12:00:00Z" 
            },
            *mock_source_articles, 
            { 
                "id": 104,
                "headline": "Post-Event X: Economic Impact Assessed",
                "Content": "<p>Following the conclusion of Event X, analysts are now assessing its economic impact on the local region. Initial figures look promising.</p>",
                "created_at": "2023-01-03T10:00:00Z" 
            }
        ]
        mock_new_source_for_update.sort(key=lambda x: x['created_at'])

        print("\n--- Test 2: Updating Existing Article ---")
        synthesized_article_updated = await generate_cluster_article(
            mock_new_source_for_update, 
            previous_combined_article_data=mock_previous_article
        )
        print(f"Updated Headline: {synthesized_article_updated.get('headline')}")
        print(f"Updated Summary: {synthesized_article_updated.get('summary')}")
        print(f"Updated Content Snippet: {str(synthesized_article_updated.get('content', ''))[:200]}...")
        if 'parsing_error' in synthesized_article_updated:
            print(f"Parsing Error: {synthesized_article_updated['parsing_error']}")

    if gemini_model and "Error:" not in multi_source_synthesis_prompt_template:
        asyncio.run(test_cluster_generation())
    else:
        print("Skipping cluster_article_generator.py test due to model/prompt initialization issues.")