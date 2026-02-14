import os
import requests
import re

def ensure_downloaded(url, filepath):
    """
    Checks if a file exists at filepath. If not, downloads it from url.
    """
    if os.path.exists(filepath):
        print(f"File already exists: {filepath}")
        return

    print(f"Downloading from {url} to {filepath}...")
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Download complete.")
    except Exception as e:
        print(f"Error downloading file: {e}")
        # Clean up partial file if needed
        if os.path.exists(filepath):
            os.remove(filepath)
        raise

def load_and_chunk_text(filepath, min_length=200):
    """
    Loads text from filepath, splits into paragraphs, and filters them.
    """
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()

    # Split by double newlines (common for Project Gutenberg texts)
    # Also handle common Gutenberg headers/footers roughly if possible, 
    # but for now we just filter by length.
    paragraphs = text.split('\n\n')
    
    clean_paragraphs = []
    for p in paragraphs:
        # Replace single newlines within a paragraph with spaces
        clean_p = p.replace('\n', ' ').strip()
        
        if len(clean_p) >= min_length:
            clean_paragraphs.append(clean_p)
            
    print(f"Loaded {len(clean_paragraphs)} paragraphs from {filepath}.")
    return clean_paragraphs

if __name__ == "__main__":
    # Test with Gibbon
    url = "https://www.gutenberg.org/cache/epub/25717/pg25717.txt" # Volume 1
    path = "experiments/data/gibbon_vol1.txt"
    ensure_downloaded(url, path)
    paragraphs = load_and_chunk_text(path)
    if paragraphs:
        print("First paragraph sample:")
        print(paragraphs[0][:200] + "...")
