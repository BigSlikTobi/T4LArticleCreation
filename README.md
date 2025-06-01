# Project Overview

Tackle4LossArticleCreation is part 3 of the Tackle4Loss Projext that gathers extracts, **enriches** and publicates American Football News an Tackle4Loss.com.

This project is an automated pipeline that fetches raw article data, processes it using AI, and stores enriched articles. The key technologies used are Python, Supabase for the database, and Google GenAI for content generation and processing.

# Features

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

# Architecture

The system follows a modular architecture:

1.  **Data Ingestion:** Scripts responsible for fetching raw article data from external sources.
2.  **Article Processing Pipeline:** A core set of modules that enrich articles. This includes:
    *   English article generation.
    *   German translation.
    *   Image selection.
    *   Team classification.
3.  **AI Model Interaction:** Modules dedicated to configuring and interacting with Google GenAI models for various NLP tasks.
4.  **Database Interaction:** A dedicated module (`database.py`) manages all communication with the Supabase PostgreSQL database, handling data C.R.U.D. operations.
5.  **Orchestration:** The overall pipeline is orchestrated by `pipeline.py` and scheduled/triggered via GitHub Actions workflows (see `.github/workflows/`).

# Key Scripts and Modules

*   **`pipeline.py`**: Main orchestrator of the article processing workflow. It coordinates the execution of different modules, from fetching raw articles to storing the final enriched versions.
*   **`database.py`**: Handles all interactions with the Supabase database. This includes fetching articles needing processing, updating them with generated content, and storing new data.
*   **`LLMSetup.py`**: Configures and initializes Google GenAI models used for content generation, translation, and other AI-driven tasks.
*   **`englishArticle.py`**: Script responsible for generating the English version of articles. It utilizes LLMs with prompts defined in `prompts.yml`.
*   **`germanArticle.py`**: Script responsible for translating English articles into German. It also uses LLMs and prompts from `prompts.yml`.
*   **`articleImage.py`**: Script used to find and select relevant images for articles, likely by querying image services or using AI to determine relevance.
*   **`team_classifier.py`**: Script for classifying articles by sports team based on their content. This helps in categorizing and tagging articles.
*   **`prompts.yml`**: A YAML configuration file storing various prompts used for interacting with the LLMs. This allows for easy management and modification of prompts for tasks like article generation, translation, image search queries, and team classification.
*   **`ArticleUpdates/` (directory)**: Contains utility scripts for performing batch updates, data maintenance tasks, or specific changes to article statuses within the database.

# GitHub Actions Workflows

The project utilizes GitHub Actions for automation. Key workflows found in `.github/workflows/` include:

*   **`article-creation.yml`**: This workflow is responsible for automating the main article processing pipeline. It likely runs on a schedule (e.g., daily) or is triggered by specific events (e.g., a push to the main branch). Its tasks include fetching new raw articles, processing them through the various AI-powered enrichment steps (English generation, German translation, image selection, team classification), and storing the final, enriched articles in the Supabase database.

# Setup and Installation

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
    *   Create a `.env` file in the root directory.
    *   Add the following variables:
        ```
        SUPABASE_URL="your_supabase_url"
        SUPABASE_KEY="your_supabase_service_key"
        GOOGLE_API_KEY="your_google_ai_api_key"
        # Add any other necessary configuration variables
        ```
4.  **Set up Supabase:**
    *   Create a new project on Supabase.
    *   Ensure your Supabase database schema is set up. If schema migration scripts are provided in the repository, run them as per their instructions.

# Usage

*   **Running the main pipeline:**
    To execute the full article processing pipeline, run:
    ```bash
    python pipeline.py
    ```
*   **Manual Script Execution:**
    *   Certain utility scripts, particularly those in the `ArticleUpdates/` directory, might be designed for manual execution for specific data maintenance tasks. Refer to the script's documentation or comments for usage instructions.
    *   *(Note: The primary method for processing articles is through the main `pipeline.py` script.)*

# Configuration

The project is configured through environment variables. Create a `.env` file in the root directory (you can copy `.env.example` if it exists and rename it to `.env`). 
Refer to `.env.example` (if provided) for a comprehensive list of all possible configuration options.

Key variables to set in your `.env` file:

*   `SUPABASE_URL`: Your Supabase project URL.
*   `SUPABASE_KEY`: Your Supabase service key (use with caution, should be kept secret).
*   `GOOGLE_API_KEY`: Your API key for Google GenAI services.
*   `ARTICLE_SOURCES_FILE`: Path to the JSON file defining article sources (defaults to `config/article_sources.json`).
*   `LOG_LEVEL`: Logging level (e.g., INFO, DEBUG, ERROR).

# Error Handling and Logging

*   The system implements robust error handling to manage issues during article fetching or AI processing.
*   Logs are generated and can be configured via the `LOG_LEVEL` environment variable.
*   Check the default `logs/` directory (or as configured) for detailed log files.

# Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

# License

This project is licensed under the MIT License.
