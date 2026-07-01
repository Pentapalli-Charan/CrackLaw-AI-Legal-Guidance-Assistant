import hashlib
import os
import re
from typing import List, Set

def calculate_checksum(file_path: str, algorithm: str = "sha256") -> str:
    """Calculates the hash checksum of a file to detect duplicates and verify integrity."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found for checksum calculation: {file_path}")
    
    hash_obj = hashlib.new(algorithm)
    # Read in 64kb chunks
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hash_obj.update(chunk)
            
    return hash_obj.hexdigest()

def estimate_word_count(text: str) -> int:
    """Estimates the number of words in a text block."""
    if not text:
        return 0
    # Clean whitespace and split
    words = re.findall(r"\b\w+\b", text)
    return len(words)

def estimate_token_count(text: str) -> int:
    """Estimates tokens based on a standard multiplier (1 word ~ 1.3 tokens)."""
    return int(estimate_word_count(text) * 1.33)

def scan_files_by_extension(directory: str, extensions: Set[str]) -> List[str]:
    """Scans directory and subdirectories for files with given extensions (case-insensitive)."""
    matched_files = []
    normalized_exts = {ext.lower().lstrip(".") for ext in extensions}
    
    for root, _, files in os.walk(directory):
        for file in files:
            ext = os.path.splitext(file)[1].lower().lstrip(".")
            if ext in normalized_exts:
                matched_files.append(os.path.normpath(os.path.join(root, file)))
                
    return matched_files

def get_relative_path(path: str, start: str) -> str:
    """Gets a clean relative path from start, replacing backslashes on Windows for uniformity."""
    try:
        rel = os.path.relpath(path, start)
        return rel.replace("\\", "/")
    except ValueError:
        return path
