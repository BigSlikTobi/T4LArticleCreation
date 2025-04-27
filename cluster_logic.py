import logging
from typing import Dict, List, Set, Tuple, FrozenSet # Use FrozenSet for dictionary keys

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)


def calculate_jaccard(set1: Set[int], set2: Set[int]) -> float:
    """Calculates the Jaccard similarity coefficient."""
    # Ensure inputs are sets
    if not isinstance(set1, set): set1 = set(set1)
    if not isinstance(set2, set): set2 = set(set2)

    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))

    if union == 0:
        # If both sets are empty, Jaccard is often defined as 1.0
        return 1.0
    else:
        return intersection / union

def match_and_categorize_clusters(
    # Takes the direct output of fetch_recent_clusters
    recent_article_groups: Dict[str, Set[int]],
    existing_stories: List[Dict],
    similarity_threshold: float = 0.5
) -> Tuple[List[Dict], List[Dict]]:
    """
    Matches recent groups of articles (identified by their article ID sets)
    against existing stories based on article ID similarity, ignoring the
    ephemeral cluster_id from SourceArticles for matching.

    Args:
        recent_article_groups: Dictionary mapping an ephemeral cluster_id (str) from the
                               current run to a set of recent SourceArticle IDs.
                               (Output of database.fetch_recent_clusters)
        existing_stories: List of recent ClusterStories, each a dict with at least
                          'id' (story UUID str), 'cluster_id' (str, ignored for matching),
                          and 'source_article_ids' (list or set of ints).
                          (Output of database.fetch_recent_cluster_stories)
        similarity_threshold: The Jaccard similarity score above which a group is
                              considered an update candidate if article sets differ.

    Returns:
        A tuple containing two lists:
        1. clusters_to_create: List of dicts [{'latest_cluster_id': str, 'article_ids': Set[int]}]
        2. clusters_to_update: List of dicts [{'latest_cluster_id': str, 'article_ids': Set[int], 'existing_story_id': str}]
    """
    clusters_to_create = []
    clusters_to_update = []
    # Keep track of existing story IDs that have been matched to avoid duplicates if multiple recent groups match the same story
    matched_existing_story_ids = set()
    # Keep track of recent article sets that have been processed to avoid creating duplicates if multiple ephemeral IDs point to the same set
    processed_recent_article_sets: Set[FrozenSet[int]] = set()


    # --- Prepare existing stories data (including converting article IDs to sets) ---
    existing_story_data = []
    logger.info(f"Processing {len(existing_stories)} fetched existing stories for comparison...")
    for story in existing_stories:
        story_id_val = story.get('id')
        if story_id_val is None:
            logger.warning(f"Skipping existing story due to missing 'id': {story}")
            continue
        story_id_str = str(story_id_val)

        article_ids_val = story.get('source_article_ids', [])
        if article_ids_val is None: article_ids_val = []
        try:
            existing_article_set = set(int(aid) for aid in article_ids_val)
        except (ValueError, TypeError):
             logger.warning(f"Could not parse source_article_ids for existing story {story_id_str}. Skipping story. Value: {article_ids_val}")
             continue

        existing_story_data.append({
            'id': story_id_str,
            'source_article_ids': existing_article_set
        })
    logger.info(f"Prepared {len(existing_story_data)} existing stories with article sets.")


    # --- Iterate through recently identified article groups ---
    logger.info(f"Starting iteration through {len(recent_article_groups)} recent article groups...")
    for latest_cluster_id, current_article_ids_set in recent_article_groups.items():
        logger.info(f"--- Comparing Recent Group (Ephemeral Cluster ID: {latest_cluster_id}) ---")
        logger.debug(f"Article Set ({len(current_article_ids_set)}): {current_article_ids_set}")

        # Ensure current set is valid integers
        try:
            current_article_set = set(int(aid) for aid in current_article_ids_set)
        except (ValueError, TypeError):
             logger.warning(f"Could not parse article IDs for recent group {latest_cluster_id}. Skipping group. Value: {current_article_ids_set}")
             continue

        # Use frozenset for checking if we've already processed this exact set under a different ephemeral ID
        current_article_frozenset = frozenset(current_article_set)
        if current_article_frozenset in processed_recent_article_sets:
            logger.info(f"Article set {current_article_set} already processed under a different ephemeral cluster ID. Skipping.")
            continue

        best_match_story_id = None
        best_similarity_score = -1.0 # Initialize below 0

        # Compare current set against ALL existing stories
        for existing_story in existing_story_data:
            existing_article_set = existing_story['source_article_ids']
            similarity = calculate_jaccard(current_article_set, existing_article_set)

            logger.debug(f"  Comparing with existing story {existing_story['id']} (Set size: {len(existing_article_set)}): Jaccard = {similarity:.4f}")

            # Update best match if this one is better
            if similarity > best_similarity_score:
                best_similarity_score = similarity
                best_match_story_id = existing_story['id']
                logger.debug(f"    -> New best match found: Story {best_match_story_id}, Score {best_similarity_score:.4f}")

        # --- Categorize based on the best match found ---
        logger.info(f"Finished comparing. Best match: Story ID {best_match_story_id}, Similarity {best_similarity_score:.4f} (Threshold: {similarity_threshold})")

        processed_recent_article_sets.add(current_article_frozenset) # Mark this set as processed

        if best_similarity_score == 1.0:
            logger.info(f"Categorizing as NO_CHANGE (Identical to existing story {best_match_story_id}). Skipping.")
            matched_existing_story_ids.add(best_match_story_id) # Mark story as matched
        elif best_similarity_score >= similarity_threshold:
             # Check if this existing story has already been matched by a potentially better recent cluster
             if best_match_story_id in matched_existing_story_ids:
                  logger.warning(f"Existing story {best_match_story_id} was already matched by another recent cluster. Treating current group {latest_cluster_id} as NEW to avoid duplicate updates.")
                  clusters_to_create.append({
                      'latest_cluster_id': latest_cluster_id, # Store ephemeral ID for reference
                      'article_ids': current_article_set
                  })
             else:
                  logger.info(f"Categorizing as UPDATE for existing story {best_match_story_id}.")
                  clusters_to_update.append({
                      'latest_cluster_id': latest_cluster_id, # Store ephemeral ID for reference
                      'article_ids': current_article_set,
                      'existing_story_id': best_match_story_id
                  })
                  matched_existing_story_ids.add(best_match_story_id) # Mark story as matched
        else: # best_similarity_score < threshold
            logger.info(f"Categorizing as NEW (Similarity below threshold or no existing stories).")
            clusters_to_create.append({
                'latest_cluster_id': latest_cluster_id, # Store ephemeral ID for reference
                'article_ids': current_article_set
            })
        logger.info(f"--- Finished categorizing group {latest_cluster_id} ---")


    logger.info(f"Categorization complete: {len(clusters_to_create)} clusters to create, {len(clusters_to_update)} clusters to update.")
    return clusters_to_create, clusters_to_update