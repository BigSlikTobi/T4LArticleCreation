## Overview
##### Tackle4LossArticleCreation is part 3 of the Tackle4Loss Projext that gathers extracts, **enriches** and publicates American Football News an Tackle4Loss.com.

**Tackle4LossArticleCreation** is an automated pipeline designed to gather, enrich, and prepare American Football news articles for publication on Tackle4Loss.com. The core goal is to transform raw article data into comprehensive, multilingual content ready for readers.

The automated process involves:
*   Fetching raw news articles from various sources.
*   Utilizing AI (specifically Google GenAI) for several enrichment tasks:
    *   Generating well-structured English language articles.
    *   Translating these articles into German.
    *   Selecting relevant images to accompany the articles.
    *   Classifying articles by the American Football teams they pertain to.
*   Storing these enriched articles in a Supabase database, ready for further use.

Key technologies leveraged in this project include Python for scripting and orchestration, Supabase (PostgreSQL) for data storage, and Google GenAI for advanced content processing and generation.

## Features

*   **Automated Article Fetching:** The system automatically fetches new articles from configured sources.
*   **Article Processing Pipeline:**
    *   Fetches unprocessed articles from Supabase.
    *   Generates English articles using Google GenAI (leveraging `englishArticle.py` and `prompts.yml`).
    *   Translates articles into German using Google GenAI (leveraging `germanArticle.py` and `prompts.yml`).
    *   Searches for and selects relevant article images (using `articleImage.py`).
    *   Classifies articles by sports team (using `team_classifier.py`).
    *   Stores the enriched articles back into Supabase.
*   **Data Storage:** Uses Supabase to store both raw and enriched article data.
*   **Scalable Architecture:** Designed to handle a growing volume of articles and data processing tasks.

## Module Structure and Key Components

The project is organized into several key modules and components, each responsible for specific aspects of the article processing and generation workflow:

### Core Pipeline Orchestration
*   **`pipeline.py`**: The main orchestrator for the primary article creation pipeline. It coordinates the sequence of operations from fetching raw articles from Supabase, invoking AI enrichment modules, to storing the processed articles back in the database.
*   **`cluster_pipeline.py`**: Orchestrates the workflow for generating articles based on clusters of information or topics.
*   **`.github/workflows/`**: Contains GitHub Actions workflow files (e.g., `article-creation.yml`) that automate the execution of pipelines on a schedule or via manual triggers.

### AI & Enrichment Modules
*   **`LLMSetup.py`**: Crucial for configuring and initializing the Google GenAI (Gemini models). It handles API key setup and model selection (e.g., default, lite, flash models) and configures grounding capabilities.
*   **`englishArticle.py`**: Responsible for generating the English version of articles using GenAI models. It takes raw content and transforms it into structured article format (headline, summary, body) based on prompts from `prompts.yml`.
*   **`germanArticle.py`**: Translates the generated English articles into German, again utilizing GenAI models and prompts from `prompts.yml`.
*   **`articleImage.py`**: Searches for and selects relevant images for articles. It uses Google Custom Search (configured with `Custom_Search_API_KEY` and `GOOGLE_CUSTOM_SEARCH_ID`) and may employ AI for ranking or relevance checking.
*   **`team_classifier.py`**: Classifies articles by American Football teams based on their content (headline and body). This helps in categorizing and tagging articles appropriately.

### Data Management
*   **`database.py`**: A centralized module for all database interactions with Supabase (PostgreSQL). It abstracts functions for fetching unprocessed articles, inserting processed articles, marking articles as processed, fetching team data, etc.

### Specialized Content Pipelines (`story_lines/` directory)
This directory contains modules for generating more in-depth and structured content based on existing articles or topics.
*   **`story_lines/story_line_pipeline.py`**: The main orchestrator for the story line generation process.
*   **`story_lines/article_fetcher.py`**: Fetches articles relevant to specific story lines or topics.
*   **`story_lines/deep_dive_generator.py`**: Generates detailed "deep dive" articles on specific subjects.
*   **`story_lines/deep_dive_analyzer.py`**: Analyzes content to support deep dive generation.
*   **`story_lines/deep_dive_translator.py`**: Translates deep dive articles.
*   **`story_lines/timeline_generator.py`**: Creates chronological timelines related to events or topics.
*   **`story_lines/timeline_translator.py`**: Translates timeline content.
*   **`story_lines/viewpoint_generator.py`**: Generates articles presenting different viewpoints on a subject.
*   **`story_lines/story_line_writer.py`**: Assists in writing or structuring story lines.

### Configuration Files
*   **`prompts.yml`**: A YAML configuration file that stores all the prompts used to interact with the GenAI models for various tasks like article generation, translation, summarization, image search queries, and team classification. This allows for easy modification and management of AI instructions without code changes.
*   **`.env` (file - user created)**: Used to store sensitive credentials and environment-specific configurations like API keys and database URLs. Details on variables are in the "Setup and Installation" section.

### Utilities (`ArticleUpdates/` directory)
*   This directory contains various utility scripts for batch updates, data maintenance, or specific administrative tasks on the article data in Supabase (e.g., `batch_team_classifier.py`, `update_article_status.py`). These are typically run manually as needed.

## High-Level Workflow

This section outlines the typical data flow and processing steps within the project's main pipelines.

### Main Article Creation Workflow (`pipeline.py`)

The primary goal of this workflow is to take raw article information and transform it into enriched, multilingual content ready for use. The process is as follows:

1.  **Fetch Unprocessed Articles:** The pipeline queries the Supabase database (via `database.py`) for articles that have not yet been processed.
2.  **English Article Generation:** For each unprocessed article, the raw content (often just a headline and basic text) is sent to a Google GenAI model (via `englishArticle.py` and `LLMSetup.py`) using prompts from `prompts.yml`. The AI generates a more complete English article, including an engaging headline, a concise summary, and the main body content.
3.  **German Article Translation:** The newly generated English article (headline and content) is then sent back to a GenAI model (via `germanArticle.py`) for translation into German. This also produces a German headline, summary, and body.
4.  **Image Searching:** Relevant images are sought for the article. The English content and headline are used as search queries for Google Custom Search (via `articleImage.py`). The system aims to find and select up to three relevant images.
5.  **Team Classification:** The English headline and content are analyzed (via `team_classifier.py`) to determine which American Football team the article is primarily about. If a team is identified with sufficient confidence, it's associated with the article.
6.  **Content Validation:** Before saving, the pipeline checks if all essential components (English/German headlines, summaries, content, and at least one image) have been successfully generated.
7.  **Store Enriched Article:** If validation passes, the complete enriched article (English and German versions, image URLs, team classification, etc.) is inserted into a separate table in the Supabase database (via `database.py`).
8.  **Mark as Processed:** Finally, the original source article is marked as processed in Supabase to prevent redundant processing in future pipeline runs.

This entire sequence is orchestrated by `pipeline.py` and can be run manually or automatically via GitHub Actions (see "Running the Pipelines" section).

### Story Lines Content Workflow (`story_lines/`)

The modules within the `story_lines/` directory facilitate the creation of more specialized and in-depth content, such as:

*   **Deep Dives:** Generating comprehensive articles on specific topics.
*   **Timelines:** Creating chronological summaries of events.
*   **Viewpoints:** Presenting different perspectives on a subject.

The general workflow for these often involves:
1.  **Fetching Relevant Articles:** Gathering base articles related to a chosen topic or story line (using `story_lines/article_fetcher.py`).
2.  **AI-Powered Analysis and Generation:** Utilizing GenAI models to analyze the fetched content, extract key information, and generate new structured content according to the desired format (deep dive, timeline, etc., using modules like `story_lines/deep_dive_generator.py`, `story_lines/timeline_generator.py`).
3.  **Translation:** Translating the generated specialized content if required (e.g., `story_lines/deep_dive_translator.py`).
4.  **Storage:** Saving the output to the database.

These workflows are typically orchestrated by scripts like `story_lines/story_line_pipeline.py` or more specific pipeline scripts within the directory.

## Setup and Installation

1.  **Clone the repository:**
    Replace `<repository_url>` with the actual URL of the repository.
    ```bash
    git clone <repository_url>
    cd <repository_name> 
    ```
    (Replace `<repository_name>` with the actual directory name of the cloned repository).
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: Ensure you have Python 3.x installed.)*
3.  **Configure environment variables:**
    *   Create a `.env` file in the root directory of the project by copying an example if provided (e.g., `.env.example`) or creating a new one.
    *   Add the following environment variables to the `.env` file, replacing the placeholder values with your actual credentials and information:

        ```env
        # Supabase Configuration
        SUPABASE_URL="your_supabase_project_url"         # URL for your Supabase project
        SUPABASE_KEY="your_supabase_service_role_key"   # Service role key for your Supabase project

        # Google Cloud & GenAI Configuration
        GEMINI_API_KEY="your_google_gemini_api_key"     # API Key for Google GenAI (Gemini models)
        Custom_Search_API_KEY="your_google_custom_search_api_key"  # API Key for Google Custom Search Engine (for image search)
        GOOGLE_CUSTOM_SEARCH_ID="your_google_custom_search_engine_id" # CX ID for Google Custom Search Engine (for image search)

        # Pipeline Configuration
        ARTICLE_SOURCES_FILE="config/article_sources.json" # Path to the JSON file defining article sources
        LOG_LEVEL="INFO"                                   # Logging level (e.g., INFO, DEBUG, ERROR)

        # Optional / Specific Features
        # OPENAI_API_KEY="your_openai_api_key"            # OpenAI API Key, may be used for specific functionalities or older versions.
        ```
    *   **Note on API Keys:**
        *   `SUPABASE_KEY` should ideally be a service role key with appropriate permissions.
        *   Keep all API keys confidential and do not commit them directly to your repository. The `.gitignore` file should already be configured to ignore `.env` files.
4.  **Set up Supabase:**
    *   Create a new project on Supabase.
    *   Ensure your Supabase database schema is set up. If schema migration scripts are provided in the repository, run them as per their instructions.

## Running the Pipelines

This project contains several pipelines for different data processing tasks. Ensure you have completed the setup steps, including environment variable configuration, before running any pipeline.

### Main Article Creation Pipeline

This is the primary workflow for fetching, enriching, and storing articles.

*   **Manual Execution:**
    To run the pipeline manually from your local environment:
    ```bash
    python pipeline.py
    ```
    This will process any unprocessed articles found in the database according to the defined workflow.

*   **Automated Execution (GitHub Actions):**
    The main pipeline is also configured to run automatically via a GitHub Actions workflow located in `.github/workflows/article-creation.yml`.
    *   **Schedule:** It runs every 10 minutes.
    *   **Manual Trigger:** You can also manually trigger this workflow from the "Actions" tab of the GitHub repository.
    This automated setup ensures continuous processing of new articles.

### Cluster Article Generation Pipeline

This pipeline appears to handle the generation of articles based on clusters.

*   **Manual Execution:**
    To run the cluster article generation pipeline:
    ```bash
    python cluster_pipeline.py
    ```

### Story Line Pipeline

This pipeline is responsible for tasks related to story line generation, including fetching articles, generating deep dives, timelines, and viewpoints.

*   **Manual Execution:**
    To run the story line pipeline:
    ```bash
    python story_lines/story_line_pipeline.py
    ```
    There are also more specific scripts within the `story_lines` directory (e.g., `deep_dive_pipeline.py`) that might be run for more granular tasks.

### Utility Scripts

*   Scripts within the `ArticleUpdates/` directory are generally intended for specific batch updates or maintenance tasks. Refer to comments within these scripts for usage instructions if needed.

## Error Handling and Logging

*   The system implements robust error handling to manage issues during article fetching or AI processing.
*   Logs are generated and can be configured via the `LOG_LEVEL` environment variable.
*   Check the default `logs/` directory (or as configured) for detailed log files.

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

This project is licensed under the MIT License.
