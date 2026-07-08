import hashlib

class Deduplicator:
    """Advanced Deduplication to detect exact and near duplicates while preserving history."""
    
    def __init__(self):
        self.exact_hashes = set()
        # In a production system, this would be a LSH (Locality Sensitive Hashing) index like MinHash
        self.history = {}

    def _get_exact_hash(self, text: str) -> str:
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def _get_near_hash(self, text: str) -> str:
        # Simplified near-duplicate hash using word-level set
        words = set(text.lower().split())
        return hash(frozenset(words))

    def is_duplicate(self, text: str, doc_id: str) -> dict:
        exact = self._get_exact_hash(text)
        near = self._get_near_hash(text)
        
        result = {
            "is_duplicate": False,
            "duplicate_type": None
        }

        if exact in self.exact_hashes:
            result["is_duplicate"] = True
            result["duplicate_type"] = "exact"
            return result
            
        # Check near duplicate for amendments
        # (Mock logic for near duplicate matching)
        if near in self.history:
            # We preserve the historical version
            result["is_duplicate"] = True
            result["duplicate_type"] = "amended_version"
            return result

        self.exact_hashes.add(exact)
        self.history[near] = doc_id
        return result
