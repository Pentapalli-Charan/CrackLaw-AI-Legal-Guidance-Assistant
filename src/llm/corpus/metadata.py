import json
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List

@dataclass
class CorpusDocument:
    """Represents a single unified legal training document in the corpus."""
    
    document_id: str
    text: str
    doc_type: str  # laws, judgments, etc.
    
    # Hierarchical Legal Metadata
    act: Optional[str] = None
    chapter: Optional[str] = None
    section: Optional[str] = None
    subsection: Optional[str] = None
    judgment_title: Optional[str] = None
    
    # Source Metadata
    source: Optional[str] = None
    language: str = "en"
    
    # Pipeline tracking
    is_valid: bool = True
    word_count: int = 0
    validation_errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for JSON serialization."""
        return asdict(self)
        
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), ensure_ascii=False)
