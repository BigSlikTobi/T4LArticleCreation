#!/usr/bin/env python3
"""
Test script to verify the complete story line pipeline with deep dive translation
"""
import asyncio
import logging
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from story_lines.story_line_pipeline import process_cluster_end_to_end
from story_lines.timeline_generator import load_prompts as load_generator_prompts, initialize_llm as initialize_generator_llm
from story_lines.timeline_translator import load_prompts as load_translator_prompts, initialize_llm as initialize_translator_llm
from story_lines.viewpoint_generator import load_viewpoint_prompts, initialize_viewpoint_llm
from story_lines.deep_dive_generator import load_deep_dive_prompts, initialize_deep_dive_llm
from story_lines.deep_dive_translator import load_prompts as load_deep_dive_translator_prompts, initialize_llm as initialize_deep_dive_translator_llm
from database import fetch_all_cluster_ids

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    force=True
)
logger = logging.getLogger(__name__)


async def test_single_cluster_pipeline():
    """Test the complete pipeline on a single cluster"""
    try:
        # Initialize all components
        logger.info("TEST: Initializing all pipeline components...")
        
        logger.info("TEST: Loading timeline generator...")
        load_generator_prompts()
        await initialize_generator_llm(model_type="default")
        
        logger.info("TEST: Loading timeline translator...")
        load_translator_prompts()
        await initialize_translator_llm(model_type="flash")
        
        logger.info("TEST: Loading viewpoint generator...")
        load_viewpoint_prompts()
        await initialize_viewpoint_llm(model_type="flash")
        
        logger.info("TEST: Loading deep dive generator...")
        load_deep_dive_prompts()
        await initialize_deep_dive_llm(model_type="default")
        
        logger.info("TEST: Loading deep dive translator...")
        load_deep_dive_translator_prompts()
        await initialize_deep_dive_translator_llm(model_type="flash")
        
        # Get first cluster for testing
        cluster_ids = await fetch_all_cluster_ids()
        if not cluster_ids:
            logger.error("TEST: No clusters found for testing")
            return
            
        test_cluster_id = str(cluster_ids[0])
        logger.info(f"TEST: Using cluster {test_cluster_id} for testing")
        
        # Run the complete pipeline
        result = await process_cluster_end_to_end(test_cluster_id)
        
        # Display results
        logger.info("\n=== TEST RESULTS ===")
        logger.info(f"Cluster ID: {result['cluster_id']}")
        logger.info(f"Timeline generation: {'✓' if result['timeline_generation_success'] else '✗'}")
        logger.info(f"Timeline translation: {'✓' if result['timeline_translation_success'] else '✗'}")
        logger.info(f"Viewpoint determination: {'✓' if result['viewpoint_determination_success'] else '✗'}")
        logger.info(f"Deep dives generated: {result['deep_dive_generation_count']}")
        logger.info(f"Story line views saved: {result['story_line_views_saved']}")
        logger.info(f"Deep dive translations: {result['deep_dive_translations_count']}")
        
        if result.get('viewpoints'):
            logger.info(f"Viewpoints found:")
            for i, vp in enumerate(result['viewpoints']):
                logger.info(f"  {i+1}. {vp.get('name', 'N/A')}")
                
        logger.info("=== END TEST RESULTS ===")
        return result
        
    except Exception as e:
        logger.error(f"TEST: Pipeline test failed: {e}", exc_info=True)
        return None


async def test_translation_only():
    """Test only the deep dive translation functionality"""
    try:
        logger.info("TEST: Testing deep dive translation only...")
        
        # Initialize translation components
        load_deep_dive_translator_prompts()
        await initialize_deep_dive_translator_llm(model_type="flash")
        
        # Import and test translation
        from database import get_untranslated_story_line_views
        from story_lines.deep_dive_translator import translate_deep_dive
        
        untranslated_views = await get_untranslated_story_line_views(language_code='de')
        
        if not untranslated_views:
            logger.info("TEST: No untranslated views found for testing")
            return
            
        logger.info(f"TEST: Found {len(untranslated_views)} untranslated views")
        
        # Test translation on first view
        test_view = untranslated_views[0]
        story_line_view_id = test_view.get('id')
        
        deep_dive_data = {
            "headline": test_view.get('headline', ''),
            "content": test_view.get('content', ''),
            "introduction": test_view.get('introduction', ''),
            "view": test_view.get('view', ''),
            "justification": test_view.get('justification', '')
        }
        
        logger.info(f"TEST: Translating view '{test_view.get('view')}' (ID: {story_line_view_id})")
        success, translated_data = await translate_deep_dive(story_line_view_id, deep_dive_data, language_code='de')
        
        logger.info(f"TEST: Translation {'succeeded' if success else 'failed'}")
        if translated_data:
            logger.info(f"TEST: Translated headline: {translated_data.get('headline', 'N/A')[:100]}...")
            
        return success
        
    except Exception as e:
        logger.error(f"TEST: Translation test failed: {e}", exc_info=True)
        return False


async def main():
    """Main test function"""
    logger.info("Starting complete pipeline tests...")
    
    # Test 1: Translation only (faster test)
    logger.info("\n=== TEST 1: TRANSLATION ONLY ===")
    translation_result = await test_translation_only()
    
    # Test 2: Complete pipeline (full test)
    if translation_result:
        logger.info("\n=== TEST 2: COMPLETE PIPELINE ===")
        pipeline_result = await test_single_cluster_pipeline()
        
        if pipeline_result:
            logger.info("✓ All tests completed successfully!")
        else:
            logger.error("✗ Complete pipeline test failed")
    else:
        logger.error("✗ Translation test failed - skipping full pipeline test")


if __name__ == "__main__":
    asyncio.run(main())
