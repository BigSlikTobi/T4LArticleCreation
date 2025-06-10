#!/usr/bin/env python
# Automatic archival script for news articles older than 18 days

import sys
import os
import asyncio
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import archive_old_articles

async def main():
    print("Starting archival of old news articles (older than 18 days)...")
    stats = await archive_old_articles()
    print("\nArchival Summary:")
    print(f"Total articles processed: {stats['total']}")
    print(f"Articles archived: {stats['archived']}")
    print(f"Errors encountered: {stats['errors']}")

if __name__ == "__main__":
    asyncio.run(main())
