import asyncio
import os
import sys
import pytest

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from articleImage import ImageSearcher

@pytest.mark.asyncio
async def test_domain_extraction():
    print("Starting domain extraction test...")
    
    # Create test environment variables if not present
    if 'SUPABASE_URL' not in os.environ:
        os.environ['SUPABASE_URL'] = 'https://example.supabase.co'
    if 'SUPABASE_KEY' not in os.environ:
        os.environ['SUPABASE_KEY'] = 'dummy_key'
    if 'Custom_Search_API_KEY' not in os.environ:
        os.environ['Custom_Search_API_KEY'] = 'dummy_key'
    if 'GOOGLE_CUSTOM_SEARCH_ID' not in os.environ:
        os.environ['GOOGLE_CUSTOM_SEARCH_ID'] = 'dummy_id'
    
    try:
        # Create a test image with no source
        test_image = {
            'url': 'https://example.com/images/test123.jpg',
            'title': 'Test Image',
            'width': 800,
            'height': 600
        }
        
        print("Initializing ImageSearcher...")
        # Initialize the image searcher with mock initialization
        image_searcher = ImageSearcher(use_llm=False)
        
        # Mock the download_and_upload_image method to avoid actual API calls
        async def mock_download_upload(url, path):
            print(f"Mock download and upload for {url}")
            return f"https://mock-storage.com/{path}"
        
        # Replace the real method with our mock
        image_searcher.download_and_upload_image = mock_download_upload
        
        print("Processing test image...")
        # Process the image
        processed_images = await image_searcher._process_images([test_image], 1)
        
        # Check that source was extracted from the URL
        if processed_images and len(processed_images) > 0:
            source = processed_images[0].get('source', '')
            print(f"Extracted source: {source}")
            
            if source == 'example.com':
                print("✅ Successfully extracted domain from image URL")
            else:
                print(f"❌ Failed to extract correct domain (got '{source}', expected 'example.com')")
                
            # Print all image metadata for debugging
            print("\nAll image metadata:")
            for key, value in processed_images[0].items():
                print(f"  {key}: {value}")
        else:
            print("❌ No processed images returned")
    except Exception as e:
        print(f"Error during test: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_domain_extraction())
