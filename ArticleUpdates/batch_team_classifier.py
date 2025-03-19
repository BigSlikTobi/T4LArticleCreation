import asyncio
import json
from typing import Dict, List
import os
from database import fetch_teams, supabase, supabase_key, supabase_url
import requests
from team_classifier import classify_team

async def fetch_unassigned_articles() -> List[Dict]:
    """Fetch articles that don't have a team assigned"""
    try:
        print("Fetching unassigned articles...")
        headers = {
            "apikey": supabase_key,
            "Authorization": f"Bearer {supabase_key}"
        }
        url = f"{supabase_url}/rest/v1/NewsArticles"
        params = {
            "select": "id,headlineEnglish,ContentEnglish",
            "team": "is.null"
        }
        
        response = requests.get(url, headers=headers, params=params)
        if response.status_code == 200:
            return response.json()
        print(f"Error fetching articles: {response.status_code}")
        return []
    except Exception as e:
        print(f"Error: {e}")
        return []

async def update_article_team(article_id: int, team_id: int) -> bool:
    """Update article with assigned team"""
    try:
        data = {"team": team_id}
        response = supabase.table("NewsArticles").update(data).eq("id", article_id).execute()
        return True
    except Exception as e:
        print(f"Error updating article {article_id}: {e}")
        return False

def find_team_id(team_name: str, teams_data: List[Dict]) -> int:
    """Find team ID from team name using fuzzy matching"""
    # Convert team name to lowercase for comparison
    team_name_lower = team_name.lower()
    
    # First try exact match
    for team in teams_data:
        if team.get("fullName", "").lower() == team_name_lower:
            return team["id"]
    
    # Then try partial match
    for team in teams_data:
        if team_name_lower in team.get("fullName", "").lower() or \
           any(part in team_name_lower for part in team.get("fullName", "").lower().split()):
            return team["id"]
    
    return None

async def process_batch(articles: List[Dict], teams_data: List[Dict], batch_start: int, total: int):
    """Process a batch of articles"""
    assignments = []
    
    for idx, article in enumerate(articles, 1):
        article_id = article["id"]
        overall_idx = batch_start + idx
        
        print(f"\nProcessing article {overall_idx}/{total} (ID: {article_id})")
        print(f"Headline: {article['headlineEnglish'][:100]}...")
        
        # Get team classification
        team_name, confidence = await classify_team(
            article["headlineEnglish"],
            article["ContentEnglish"][:4000]  # Limit content length
        )
        
        print(f"Classification result: {team_name} (confidence: {confidence:.2f})")
        
        if confidence >= 0.8 and team_name.lower() != "unknown":
            # Find matching team ID
            team_id = find_team_id(team_name, teams_data)
            
            if team_id:
                team_fullname = next(
                    (team["fullName"] for team in teams_data if team["id"] == team_id),
                    "Unknown"
                )
                print(f"Matched to team: {team_fullname} (ID: {team_id})")
                
                # Update the article
                if await update_article_team(article_id, team_id):
                    print("✓ Successfully assigned team")
                    assignments.append({
                        "article_id": article_id,
                        "team_id": team_id,
                        "team_name": team_fullname,
                        "confidence": confidence
                    })
                else:
                    print("✗ Failed to update article")
            else:
                print(f"✗ Could not find matching team ID for: {team_name}")
        else:
            print("✗ No team assigned - confidence too low or unknown team")
            
        # Save progress after each article
        with open("team_assignment_progress.json", "w") as f:
            json.dump({
                "processed": overall_idx,
                "total": total,
                "assignments": assignments
            }, f, indent=2)
        
    return assignments

async def main():
    # Get teams data
    teams_data = await fetch_teams()
    if not teams_data:
        print("Error: No teams data available")
        return
    
    print(f"Loaded {len(teams_data)} teams")
    print("\nTeam listing:")
    for team in teams_data:
        print(f"ID: {team['id']}, Name: {team['fullName']}")
    
    # Get unassigned articles
    articles = await fetch_unassigned_articles()
    if not articles:
        print("No unassigned articles found")
        return
    
    total_articles = len(articles)
    print(f"\nFound {total_articles} unassigned articles")
    
    # Process in small batches
    batch_size = 3
    all_assignments = []
    
    for i in range(0, total_articles, batch_size):
        batch = articles[i:i + batch_size]
        print(f"\nProcessing batch {i//batch_size + 1}/{(total_articles + batch_size - 1)//batch_size}")
        
        assignments = await process_batch(batch, teams_data, i, total_articles)
        all_assignments.extend(assignments)
        
        # Add delay between batches
        if i + batch_size < total_articles:
            delay = 10
            print(f"Waiting {delay} seconds before next batch...")
            await asyncio.sleep(delay)
    
    # Final report
    print("\nProcessing complete!")
    print(f"Total articles processed: {total_articles}")
    print(f"Teams assigned: {len(all_assignments)}")
    
    # Save final results
    with open("team_assignments_final.json", "w") as f:
        json.dump({
            "total_processed": total_articles,
            "total_assigned": len(all_assignments),
            "assignments": all_assignments
        }, f, indent=2)

if __name__ == "__main__":
    asyncio.run(main())