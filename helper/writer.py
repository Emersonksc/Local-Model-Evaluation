import ast
import json
import re
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

def read_json(filepath: str) -> List[Dict]:
    
    """Read JSON file and return data"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"File not found: {filepath}")
        return []
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON in file: {filepath}")
        return []
    except Exception as e:
        logger.error(f"Error reading JSON file: {e}")
        return []


# Additional formatting options
def write_json_formatted(data: List[Dict], filepath: str, 
                        indent: int = 4, 
                        sort_keys: bool = False) -> None:
    """Write JSON with custom formatting options"""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,  # Allows non-ASCII characters
            indent=indent,       # Pretty print with indentation
            sort_keys=sort_keys, # Optional: sort dictionary keys
            separators=(',', ': ')  # Custom separators
        )

def chat_object(chat_list: List[Dict], temperature: float, top_p: float, presence_penalty: float, frequency_penalty: float, max_completion_tokens: int) -> Dict:
    """Convert chat list to object"""
    return {
        "chats": chat_list,
        "temperature": temperature,
        "top_p": top_p,
        "presence_penalty": presence_penalty,
        "frequency_penalty": frequency_penalty,
        "max_completion_tokens": max_completion_tokens
    }

def anwser_object(anwser: List[Dict], temperature: float, top_p: float, presence_penalty: float, frequency_penalty: float, max_completion_tokens: int) -> Dict:
    """Convert chat list to object"""
    return {
        "anwser": anwser,
        "temperature": temperature,
        "top_p": top_p,
        "presence_penalty": presence_penalty,
        "frequency_penalty": frequency_penalty,
        "max_completion_tokens": max_completion_tokens
    }

def convert_markdown_to_dict(md_string: str) -> dict:
    """
    Convert a markdown-formatted string containing a JSON-like Python literal into a dictionary.
    
    The function:
    1. Removes the markdown code block markers (```json and ```)
    2. Uses ast.literal_eval() to safely convert the string to a Python object
    
    Returns:
        A dictionary if conversion is successful; otherwise, an empty dict.
    """
    # Remove the Markdown code block markers.
    pattern = r'```json\s*\n([\s\S]+?)\n```'
    match = re.search(pattern, md_string)
    json_content = match.group(1) if match else md_string

    try:
        # Convert the Python literal string into a dictionary.
        data = ast.literal_eval(json_content)
        return data
    except Exception as e:
        logger.error("Error parsing JSON content: %s", e)
        return {}
