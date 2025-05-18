# Deprecated: This script is deprecated and will be removed in future versions.

import sys
import os
import asyncio
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import batch_update_article_status

async def main():
    print("Starting batch update of article status...")
    stats = await batch_update_article_status()
    print("\nSummary:")
    print(f"Total articles processed: {stats['total']}")
    print(f"Articles updated: {stats['updated']}")
    print(f"Errors encountered: {stats['errors']}")

if __name__ == "__main__":
    asyncio.run(main())