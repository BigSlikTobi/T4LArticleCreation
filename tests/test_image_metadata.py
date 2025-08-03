import asyncio
import os
import sys
import pytest

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import save_article_image_metadata, save_cluster_image_metadata

@pytest.mark.asyncio
async def test_metadata_functions():
    print('Testing article image metadata:')
    result = await save_article_image_metadata(
        article_id=24, 
        image_url='https://example.com/image.jpg', 
        original_url='https://original.com/image.jpg', 
        author='Test Author', 
        source='Test Source'
    )
    print(f'Article image saved with ID: {result}')
    
    print('\nTesting cluster image metadata:')
    result = await save_cluster_image_metadata(
        cluster_id="e4f2f613-392a-4157-9c8c-427676c35a79", 
        image_url='https://example.com/image.jpg', 
        original_url='https://original.com/image.jpg', 
        author='Test Author', 
        source='Test Source'
    )
    print(f'Cluster image saved with ID: {result}')

if __name__ == "__main__":
    asyncio.run(test_metadata_functions())
