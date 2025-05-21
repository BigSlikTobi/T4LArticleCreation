# TODO - Project Optimizations

This document lists potential areas for optimization and improvement in the codebase.

## `pipeline.py`

1.  **Concurrency for Article Processing**
    *   **Issue**: Articles in `run_pipeline` are processed sequentially.
    *   **File/Function**: `pipeline.py`, `run_pipeline` function.
    *   **Suggestion**: Use `asyncio.gather` along with an `asyncio.Semaphore` to process multiple articles concurrently. The semaphore limit should be configurable.
    *   **Benefit**: Significant performance improvement for batch processing by leveraging asynchronous I/O operations.

2.  **Specific Error Handling in `process_single_article`**
    *   **Issue**: Broad `except Exception as e:` in `process_single_article`.
    *   **File/Function**: `pipeline.py`, `process_single_article` function.
    *   **Suggestion**: Implement specific exception handling for network errors, API errors (LLM, Supabase), and data validation. Consider a retry mechanism with backoff for transient errors during API calls within `process_single_article`.
    *   **Benefit**: Increased robustness, better error recovery, and improved diagnostics.

3.  **Robust HTML Cleaning**
    *   **Issue**: Basic string replacement (`.replace()`) for cleaning HTML from LLM responses.
    *   **File/Function**: `pipeline.py`, `process_single_article` (lines where `english_headline`, `english_summary`, etc. are cleaned).
    *   **Suggestion**: For more complex or less predictable HTML, consider using an HTML parsing library like `BeautifulSoup` for more reliable text extraction.
    *   **Benefit**: More robust and flexible HTML content extraction.

4.  **Configurable Pipeline Parameters**
    *   **Issue**: Fixed delay `asyncio.sleep(2)` in `run_pipeline`.
    *   **File/Function**: `pipeline.py`, `run_pipeline` function.
    *   **Suggestion**: Make this delay configurable (e.g., via environment variable or a constant in a config file) if it's for rate limiting.
    *   **Benefit**: Easier tuning of pipeline behavior.

## `database.py`

1.  **Consistent Use of Supabase SDK**
    *   **Issue**: Mixed usage of `requests.get` and the `supabase-py` SDK for database interactions.
    *   **File/Function**: `fetch_primary_sources`, `fetch_unprocessed_articles`, `fetch_teams` use `requests.get`. Other functions use the SDK.
    *   **Suggestion**: Refactor the functions currently using `requests.get` to use the `supabase-py` client methods for consistency, potentially better error handling, and connection management.
    *   **Benefit**: Code consistency, improved maintainability, and leveraging SDK features.

2.  **Caching for Frequently Accessed, Static Data**
    *   **Issue**: `fetch_primary_sources` and `fetch_teams` are called in `fetch_unprocessed_articles` and `run_pipeline` respectively. This data might not change often.
    *   **File/Function**: `fetch_primary_sources`, `fetch_teams`.
    *   **Suggestion**: Implement a simple caching mechanism (e.g., time-based in-memory cache) for data like primary sources and team lists if they are relatively static.
    *   **Benefit**: Reduced API calls to Supabase, potentially faster startup or processing.

3.  **Optimize Batch Updates**
    *   **Issue**: `batch_update_article_status` updates articles individually in a loop.
    *   **File/Function**: `database.py`, `batch_update_article_status` function.
    *   **Suggestion**: Modify the loop to collect all updates and perform a single batch update operation if the Supabase SDK supports it for the `isUpdate` field or structure the updates to minimize individual calls (e.g., `supabase.table("NewsArticles").update({"isUpdate": True}).in_("id", list_of_ids_to_set_true).execute()`).
    *   **Benefit**: Significant performance improvement for batch operations.

4.  **Review Query Efficiency**
    *   **Issue**: Python-side filtering in `fetch_unprocessed_articles` after a database query.
    *   **File/Function**: `database.py`, `fetch_unprocessed_articles` function.
    *   **Suggestion**: Review if the Supabase query can be made more specific to avoid fetching data that is then filtered out in Python. Ensure `SourceArticles` table is appropriately indexed for fields used in `WHERE` clauses (`source`, `contentType`, `isArticleCreated`, `duplication_of`).
    *   **Benefit**: More efficient database queries, reduced data transfer.

5.  **Secure Supabase Client Initialization**
    *   **Issue**: Supabase client is initialized globally. If initialization fails, functions might try to operate on a `None` client.
    *   **File/Function**: `database.py`, global scope.
    *   **Suggestion**: The `_check_supabase_client()` helper is good. Ensure it's used robustly in all functions or consider a pattern where the client is passed or re-validated if module is long-lived. For scripts, current approach is likely fine as they are short-lived.
    *   **Benefit**: Increased robustness against configuration issues.

## LLM Interaction Scripts (`englishArticle.py`, `germanArticle.py`, `team_classifier.py`, `articleImage.py`)

1.  **Refactor Common LLM Interaction Logic**
    *   **Issue**: Code for initializing models, loading prompts, calling LLMs, cleaning responses, and parsing JSON is duplicated across multiple scripts.
    *   **Files**: `englishArticle.py`, `germanArticle.py`, `team_classifier.py`, `articleImage.py` (LLM part).
    *   **Suggestion**: Create a shared utility module/class (e.g., `llm_utils.py`) to encapsulate common LLM interaction tasks. This utility should handle prompt loading, API calls (with retries), response cleaning, and JSON parsing.
    *   **Benefit**: Reduced code duplication, improved maintainability, consistent error handling.

2.  **Improve LLM JSON Output Reliability**
    *   **Issue**: Fallback regex parsing in article generation scripts suggests LLM doesn't always return clean JSON.
    *   **Files**: `englishArticle.py`, `germanArticle.py`.
    *   **Suggestion**:
        *   Refine prompts in `prompts.yml` to be stricter about JSON output.
        *   Investigate and use model features for enforced JSON output mode if available with Gemini.
        *   Consider a retry mechanism that specifically asks the LLM to correct invalid JSON.
    *   **Benefit**: More reliable LLM response parsing, less reliance on complex regex.

3.  **Specific Error Handling for LLM Calls**
    *   **Issue**: Generic error handling for LLM API calls.
    *   **Files**: `englishArticle.py`, `germanArticle.py`, `team_classifier.py`, `articleImage.py`.
    *   **Suggestion**: Implement more specific error handling for Google GenAI API exceptions (e.g., rate limits, content filtering). Include retry logic with exponential backoff for transient errors.
    *   **Benefit**: Increased resilience and better error diagnostics for LLM interactions.

4.  **Centralize LLM Parameter Configuration**
    *   **Issue**: LLM parameters like `temperature`, `max_output_tokens` are hardcoded in API calls.
    *   **Files**: `englishArticle.py`, `germanArticle.py`.
    *   **Suggestion**: Move these parameters to `prompts.yml` (if prompt-specific) or a central config file/module. The shared LLM utility could then use these configurations.
    *   **Benefit**: Easier tuning of LLM behavior, centralized configuration.

5.  **Resolve LLMSetup Grounding TODO**
    *   **Issue**: `TODO` in `LLMSetup.py` regarding grounding for the 'lite' model.
    *   **File**: `LLMSetup.py`.
    *   **Suggestion**: Investigate and fix the grounding issue or update the code to correctly reflect capabilities (e.g., disable grounding by default for incompatible models or warn).
    *   **Benefit**: Correct and intended use of LLM features.

## `articleImage.py`

1.  **SSL Verification in `download_image`**
    *   **Issue**: SSL verification is disabled (`ssl_context.verify_mode = ssl.CERT_NONE`).
    *   **File/Function**: `articleImage.py`, `download_image` function.
    *   **Suggestion**: Avoid disabling SSL verification. If issues arise with specific domains, try to obtain their CA certificates or update `certifi` store. Disabling verification is a security risk. If it must be conditional, make it highly restricted.
    *   **Benefit**: Improved security of image downloads.

2.  **Concurrency for Image Processing**
    *   **Issue**: While individual image download/upload is async, the loop in `_process_images` processes a list of images one by one if `num_images > 1`.
    *   **File/Function**: `articleImage.py`, `_process_images` function.
    *   **Suggestion**: If processing multiple images (i.e., `num_images > 1`), use `asyncio.gather` to download and upload them concurrently.
    *   **Benefit**: Faster processing when multiple images are selected for an article.

3.  **Configuration for Image Search Parameters**
    *   **Issue**: `requests_per_day` and `min_wait_time` for rate limiting are hardcoded. Blacklisted domains are hardcoded.
    *   **File**: `articleImage.py`.
    *   **Suggestion**: Move these to a configuration file or environment variables.
    *   **Benefit**: Easier tuning and management.

## General Codebase Improvements

1.  **Consistent Logging**
    *   **Issue**: Mixed use of `print()` and the `logging` module.
    *   **Suggestion**: Standardize on the `logging` module across all scripts. Configure a root logger in the main entry point (`pipeline.py`).
    *   **Benefit**: Improved diagnostics, centralized log management and control.

2.  **Centralized Configuration**
    *   **Issue**: Some parameters (batch sizes, delays) are hardcoded.
    *   **Suggestion**: Create a `config.py` or use environment variables more extensively for such parameters.
    *   **Benefit**: Easier application tuning and management across different environments.

3.  **Comprehensive Type Hinting**
    *   **Issue**: Type hinting is present but could be more consistent and complete.
    *   **Suggestion**: Add/improve type hints for all function signatures (arguments and return types) across the project.
    *   **Benefit**: Improved code readability, better static analysis, and easier refactoring.

4.  **Docstrings and Comments**
    *   **Issue**: Docstring and comment coverage can be improved.
    *   **Suggestion**: Ensure all public functions/methods have clear docstrings. Add inline comments for complex or non-obvious logic.
    *   **Benefit**: Better code understanding and maintainability.

5.  **Remove Unused Code/Comments**
    *   **Issue**: Commented-out `sys.path.append` lines in article generation scripts.
    *   **Files**: `englishArticle.py`, `germanArticle.py`.
    *   **Suggestion**: Remove these if they are not necessary. If needed, address the underlying path/module resolution issue.
    *   **Benefit**: Cleaner codebase.
