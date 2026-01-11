import re

def clean_text(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    # Remove boilerplate
    text = re.sub(r"i am writing to file a complaint|to whom it may concern", "", text)
    # Remove CFPB redacted dates/info [XX/XX/XXXX]
    text = re.sub(r"\[.*?\]", "", text)
    # Remove special characters
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    return text.strip()