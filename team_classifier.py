import asyncio
import json
import os
from typing import Dict, List, Optional, Tuple
import google.generativeai as genai
from dotenv import load_dotenv
import requests
import re

# Load environment variables
load_dotenv()

# Initialize Gemini model directly
api_key = os.environ.get("GEMINI_API_KEY")
genai.configure(api_key=api_key)
model = genai.GenerativeModel("models/gemini-2.0-flash-lite")

# Initialize Supabase client
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_KEY")

async def fetch_teams() -> List[Dict]:
    """Fetch all teams from the database"""
    try:
        headers = {
            "apikey": supabase_key,
            "Authorization": f"Bearer {supabase_key}"
        }
        response = requests.get(
            f"{supabase_url}/rest/v1/Teams",
            headers=headers,
            params={"select": "id,fullName"}
        )
        if response.status_code == 200:
            return response.json()
        return []
    except Exception as e:
        print(f"Error fetching teams: {e}")
        return []

async def fetch_unassigned_articles() -> List[Dict]:
    """Fetch articles that don't have a team assigned"""
    try:
        headers = {
            "apikey": supabase_key,
            "Authorization": f"Bearer {supabase_key}"
        }
        response = requests.get(
            f"{supabase_url}/rest/v1/NewsArticles",
            headers=headers,
            params={
                "select": "id,headlineEnglish,ContentEnglish",
                "team": "is.null"
            }
        )
        if response.status_code == 200:
            return response.json()
        return []
    except Exception as e:
        print(f"Error fetching articles: {e}")
        return []

async def update_article_team(article_id: int, team_id: int) -> bool:
    """Update an article with its team ID"""
    try:
        headers = {
            "apikey": supabase_key,
            "Authorization": f"Bearer {supabase_key}",
            "Content-Type": "application/json",
            "Prefer": "return=minimal"
        }
        response = requests.patch(
            f"{supabase_url}/rest/v1/NewsArticles",
            headers=headers,
            params={"id": f"eq.{article_id}"},
            json={"team": team_id}
        )
        return response.status_code in [200, 204]
    except Exception as e:
        print(f"Error updating article {article_id}: {e}")
        return False

async def classify_team(headline: str, content: str) -> Tuple[Optional[str], float]:
    """Classify which team an article is about using the LLM"""
    prompt = f"""You are a sports classification expert. Analyze this article and determine which NFL team it's primarily about.

Article Headline: {headline}

Article Content: {content[:4000]}

IMPORTANT: Return ONLY a raw JSON object, with no markdown formatting, no code blocks, and no explanation.
Format: {{"team_name": "Full Team Name", "confidence_score": 0.XX}}

Example response for Cowboys article:
{{"team_name": "Dallas Cowboys", "confidence_score": 0.95}}

Example response for unclear article:
{{"team_name": "Unknown", "confidence_score": 0.1}}"""

    try:
        response = await asyncio.to_thread(
            lambda: model.generate_content(prompt, generation_config={"temperature": 0.1})
        )
        
        # Clean up the response text
        text = response.text.strip()
        
        # First, try to find the JSON object regardless of markdown
        json_pattern = r'\{[^{}]*"team_name"[^{}]*"confidence_score"[^{}]*\}'
        match = re.search(json_pattern, text)
        
        if match:
            json_text = match.group(0)
        else:
            # If no match found, try cleaning markdown and try again
            # Remove code block markers and any surrounding whitespace/newlines
            text = re.sub(r'```\w*\s*', '', text)  # Remove opening ```json or ```
            text = re.sub(r'\s*```', '', text)     # Remove closing ```
            text = text.strip()
            match = re.search(json_pattern, text)
            if match:
                json_text = match.group(0)
            else:
                print(f"No JSON object found in response")
                print(f"Raw response: {text}")
                return None, 0.0
        
        # Try to parse JSON from extracted text
        try:
            result = json.loads(json_text)
            if "team_name" in result and "confidence_score" in result:
                return result["team_name"], float(result["confidence_score"])
            else:
                print(f"Missing required fields in JSON response")
                print(f"Parsed JSON: {result}")
                return None, 0.0
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"Error parsing JSON: {e}")
            print(f"Attempted JSON text: {json_text}")
            return None, 0.0
            
    except Exception as e:
        print(f"Error calling LLM: {e}")
        return None, 0.0

async def process_article(article: Dict, teams_data: List[Dict], verbose: bool = True) -> bool:
    """Process a single article and assign a team if confident"""
    article_id = article['id']
    headline = article.get('headlineEnglish', '')
    content = article.get('ContentEnglish', '')
    
    if verbose:
        print(f"\nProcessing article {article_id}")
        print(f"Headline: {headline[:100]}...")
    
    # Get team classification
    team_name, confidence = await classify_team(headline, content)
    
    if not team_name or confidence < 0.8:
        if verbose:
            print(f"✗ No confident team match: {team_name} ({confidence:.2f})")
        return False
    
    # Find matching team in database
    team_id = None
    for team in teams_data:
        db_team_name = team.get('fullName', '').lower()
        if db_team_name and (
            team_name.lower() in db_team_name or 
            db_team_name in team_name.lower()
        ):
            team_id = team['id']
            if verbose:
                print(f"✓ Matched team: {team.get('fullName')} (ID: {team_id})")
            break
    
    if not team_id:
        if verbose:
            print(f"✗ Could not match '{team_name}' to any team in database")
        return False
    
    # Update the article
    success = await update_article_team(article_id, team_id)
    if success:
        if verbose:
            print(f"✓ Updated article {article_id} with team {team_id}")
        return True
    else:
        if verbose:
            print(f"✗ Failed to update article {article_id}")
        return False

async def process_all_articles(batch_size: int = 5):
    """Process all unassigned articles in batches"""
    # Load teams data
    teams = await fetch_teams()
    if not teams:
        print("Error: No teams found in database")
        return
    
    print(f"Loaded {len(teams)} teams")
    print("\nFirst 5 teams for reference:")
    for team in teams[:5]:
        print(f"- {team.get('id')}: {team.get('fullName', 'Unknown')}")
    
    # Load unassigned articles
    articles = await fetch_unassigned_articles()
    if not articles:
        print("No unassigned articles found")
        return
    
    total = len(articles)
    processed = 0
    assigned = 0
    
    print(f"\nProcessing {total} articles in batches of {batch_size}")
    
    # Process in batches
    for i in range(0, total, batch_size):
        batch = articles[i:i+batch_size]
        print(f"\nBatch {i//batch_size + 1}/{(total+batch_size-1)//batch_size}")
        
        for article in batch:
            processed += 1
            if await process_article(article, teams, verbose=True):
                assigned += 1
            
            print(f"Progress: {processed}/{total} processed, {assigned} assigned")
        
        # Save progress
        with open('team_assignment_progress.json', 'w') as f:
            json.dump({
                'total': total,
                'processed': processed,
                'assigned': assigned,
                'last_batch': i//batch_size + 1
            }, f)
        
        # Delay between batches
        if i + batch_size < total:
            print("\nWaiting 5 seconds before next batch...")
            await asyncio.sleep(5)
    
    print(f"\nProcessing complete!")
    print(f"Total articles processed: {processed}")
    print(f"Articles assigned teams: {assigned}")
    print(f"Articles without teams: {processed - assigned}")

async def main():
    await process_all_articles(batch_size=5)

if __name__ == "__main__":
    asyncio.run(main())