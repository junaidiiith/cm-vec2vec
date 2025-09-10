import json


def sanitize_text(text):
    """Remove newlines and extra spaces from a text value."""
    if isinstance(text, str):
        return text.replace("\n", " ").strip()
    return text

def camel_or_snake_to_title(name):
    """Convert camelCase or snake_case to Title Case, preserving all-uppercase abbreviations and avoiding extra spaces."""
    if not name:
        return ""
    
    # Handle snake_case
    if '_' in name:
        parts = name.split('_')
    else:
        # Handle camelCase and abbreviations
        parts = []
        current_part = ''
        for _, char in enumerate(name):
            if char.isupper():
                # If current_part is not empty and the previous char is lowercase, start a new part
                if current_part and not current_part[-1].isupper():
                    parts.append(current_part)
                    current_part = char
                else:
                    current_part += char
            else:
                # If previous was all uppercase and next is lowercase, split (for cases like 'RDBMSCopy')
                if len(current_part) > 1 and current_part.isupper():
                    parts.append(current_part)
                    current_part = char
                else:
                    current_part += char
        if current_part:
            parts.append(current_part)
    
    # Remove any accidental empty strings
    parts = [p for p in parts if p]
    
    # Capitalize only non-all-uppercase parts, preserve abbreviations
    def smart_cap(part):
        return part if part.isupper() else part.capitalize()
    
    return sanitize_text(' '.join(smart_cap(part) for part in parts))



def read_json_file(json_path):
    """Attempt to load a JSON file using UTF-8 and fallback to Latin-1 if needed."""
    # print(f"Reading JSON file: {json_path}")
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except UnicodeDecodeError as e:
        # print(f"UTF-8 decode error: {e}, trying latin-1...")
        with open(json_path, 'r', encoding='latin-1') as f:
            data = json.load(f)
        # print(f"Successfully read JSON file with latin-1 encoding: {json_path}")
        return data

def add_element_ids(data, id_map):
    """Recursively add element ids to each term and property in the JSON structure."""
    if isinstance(data, dict):
        if 'id' in data and len(data) > 2: 
            id_map[data['id']] = data
        for _, value in data.items():
            if isinstance(value, (dict, list)):
                add_element_ids(value, id_map)
    elif isinstance(data, list):
        for item in data:
            add_element_ids(item, id_map)