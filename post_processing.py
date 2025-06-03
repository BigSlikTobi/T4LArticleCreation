import re

def remove_citations_from_text(text_content: str) -> str:
    """
    Removes citation markers like [1], [1,2], [1, 5, 12] from a string.

    Args:
        text_content (str): The input text possibly containing citation markers.

    Returns:
        str: The text with citation markers removed. If input is not a string, returns input unchanged.
    """
    if not isinstance(text_content, str): # Handle potential non-string inputs
        return text_content
    pattern = r'\s*\[\d+(?:,\s*\d+)*\]\s*'
    return re.sub(pattern, '', text_content).strip() # .strip() to remove any trailing/leading spaces left after removal

