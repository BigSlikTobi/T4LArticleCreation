import logging
from typing import Dict, List, Set, Tuple # Use Set type hint

# --- Logging Setup ---
# Basic logging config, could be enhanced or centralized
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)

def calculate_jaccard(set1: Set[int], set2: Set[int]) -> float:
    """
    Calculates the Jaccard similarity coefficient between two sets of article IDs.

    Args:
        set1: The first set of article IDs.
        set2: The second set of article IDs.

    Returns:
        The Jaccard similarity score (float between 0.0 and 1.0).
        Returns 0.0 if the union is empty.
    """
    # Ensure inputs are sets
    if not isinstance(set1, set): set1 = set(set1)
    if not isinstance(set2, set): set2 = set(set2)

    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))

    if union == 0:
        # If both sets are empty, they are identical in content (lack thereof).
        # If union is 0, intersection must also be 0. Division by zero error.
        # Jaccard index is typically defined as 1 for two empty sets, but 0 if one is empty and other not.
        # If both are empty, similarity is perfect (1.0).
        # If union is 0, it means both sets were empty.
        return 1.0 if intersection == 0 else 0.0 # Should simplify to 1.0 if union is 0
    else:
        return intersection / union

def match_and_categorize_clusters(
    recent_clusters: Dict[str, Set[int]],
    existing_stories: List[Dict],
    similarity_threshold: float = 0.5
) -> Tuple[List[Dict], List[Dict]]:
    """
    Matches recent clusters against existing stories and categorizes them
    for creation or update based on article ID similarity.

    Args:
        recent_clusters: Dictionary mapping cluster_id (str) to a set of recent SourceArticle IDs.
                         (Output of database.fetch_recent_clusters)
        existing_stories: List of recent ClusterStories, each a dict with at least
                          'id' (story UUID str), 'cluster_id' (str), and
                          'source_article_ids' (list or set of ints).
                          (Output of database.fetch_recent_cluster_stories)
        similarity_threshold: The Jaccard similarity score above which a cluster is
                              considered an update candidate if article sets differ.

    Returns:
        A tuple containing two lists:
        1. clusters_to_create: List of dicts [{'cluster_id': str, 'article_ids': Set[int]}]
        2. clusters_to_update: List of dicts [{'cluster_id': str, 'article_ids': Set[int], 'existing_story_id': str}]
    """
    clusters_to_create = []
    clusters_to_update = []
    processed_cluster_ids = set() # Keep track of existing cluster IDs we've handled

    # --- Prepare existing stories for efficient lookup ---
    # Map cluster_id to the most relevant existing story details
    existing_stories_map: Dict[str, Dict] = {}
    for story in existing_stories:
        # Ensure required keys exist and cluster_id is string
        cluster_id_val = story.get('cluster_id')
        story_id_val = story.get('id')
        if cluster_id_val is None or story_id_val is None:
            logging.warning(f"Skipping existing story due to missing 'id' or 'cluster_id': {story}")
            continue
        cluster_id_str = str(cluster_id_val)
        story_id_str = str(story_id_val)

        # Ensure source_article_ids is a set of integers
        article_ids_val = story.get('source_article_ids', [])
        if article_ids_val is None: article_ids_val = []
        try:
            existing_article_set = set(int(aid) for aid in article_ids_val)
        except (ValueError, TypeError):
             logging.warning(f"Could not parse source_article_ids for existing story {story_id_str} (cluster {cluster_id_str}). Skipping story. Value: {article_ids_val}")
             continue # Skip if IDs aren't parsable

        # If multiple stories exist for the same cluster_id, prefer the one with more articles
        # or potentially the most recent update timestamp (if available and reliable).
        # Simple approach: keep the first one encountered or overwrite if desired.
        # Let's keep the one with the largest article set as potentially more complete.
        if cluster_id_str not in existing_stories_map or len(existing_article_set) > len(existing_stories_map[cluster_id_str]['source_article_ids']):
             existing_stories_map[cluster_id_str] = {
                 'id': story_id_str,
                 'source_article_ids': existing_article_set
             }
        # else: # Optional logging if overwriting or skipping duplicates
            # logging.debug(f"Duplicate recent ClusterStory found for cluster_id {cluster_id_str}. Keeping the one with more articles.")

    logging.info(f"Prepared {len(existing_stories_map)} unique existing cluster stories for matching.")

    # --- Iterate through recently identified clusters ---
    for cluster_id, new_article_ids in recent_clusters.items():
        # Ensure IDs are integers
        try:
            current_article_set = set(int(aid) for aid in new_article_ids)
        except (ValueError, TypeError):
             logging.warning(f"Could not parse article IDs for recent cluster {cluster_id}. Skipping cluster. Value: {new_article_ids}")
             continue

        logging.debug(f"Processing recent cluster: {cluster_id} ({len(current_article_set)} articles)")

        if cluster_id in existing_stories_map:
            # Cluster ID exists in recent stories - potential UPDATE or NO_CHANGE
            processed_cluster_ids.add(cluster_id)
            existing_story_details = existing_stories_map[cluster_id]
            existing_article_ids = existing_story_details['source_article_ids']
            existing_story_id = existing_story_details['id']

            # --- Check for NO_CHANGE ---
            if current_article_set == existing_article_ids:
                logging.info(f"Cluster {cluster_id}: NO_CHANGE detected. Article sets ({len(current_article_set)}) are identical. Skipping.")
                continue

            # --- Sets are different, calculate similarity ---
            similarity = calculate_jaccard(current_article_set, existing_article_ids)
            logging.info(f"Cluster {cluster_id}: Existing story found ({existing_story_id}). Article sets differ ({len(current_article_set)} vs {len(existing_article_ids)}). Jaccard Similarity: {similarity:.4f}")

            # --- Check for UPDATE ---
            if similarity >= similarity_threshold:
                logging.info(f"Cluster {cluster_id}: Categorized as UPDATE for story {existing_story_id}.")
                clusters_to_update.append({
                    'cluster_id': cluster_id,
                    'article_ids': current_article_set, # Use the new set of IDs
                    'existing_story_id': existing_story_id
                })
            # --- Check for significant difference (Treat as NEW) ---
            else:
                # If similarity is low, it might be a new distinct event reusing a cluster ID,
                # or the cluster definition drifted significantly. Creating a new story might be safer.
                logging.warning(f"Cluster {cluster_id}: Existing story found ({existing_story_id}), but similarity ({similarity:.4f}) is below threshold ({similarity_threshold}). Treating as NEW cluster creation.")
                clusters_to_create.append({
                    'cluster_id': cluster_id,
                    'article_ids': current_article_set
                })
        else:
            # Cluster ID not found in recent existing stories - definitely NEW
            logging.info(f"Cluster {cluster_id}: No matching recent story found. Categorized as NEW.")
            clusters_to_create.append({
                'cluster_id': cluster_id,
                'article_ids': current_article_set
            })

    # --- Optional: Check for existing stories whose cluster_id didn't appear recently ---
    inactive_cluster_ids = set(existing_stories_map.keys()) - processed_cluster_ids
    if inactive_cluster_ids:
       logging.info(f"Found {len(inactive_cluster_ids)} existing stories whose cluster IDs were not in the recent fetch. These might be inactive.")


    logging.info(f"Categorization complete: {len(clusters_to_create)} clusters to create, {len(clusters_to_update)} clusters to update.")
    return clusters_to_create, clusters_to_update