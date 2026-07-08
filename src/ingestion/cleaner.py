import re
from typing import Dict, Any

class DocumentCleaner:
    @staticmethod
    def clean_text(text: str) -> str:
        # Remove Headers and Footers (e.g., Page 1 of 50, Copyright..., etc)
        text = re.sub(r'(?im)^Page\s+\d+(?:\s+of\s+\d+)?$', '', text)
        text = re.sub(r'(?im)^\s*\d+\s*$', '', text) # Standalone page numbers
        
        # Remove repeated titles (basic heuristic)
        text = re.sub(r'(?im)^([A-Z][A-Z\s]+)$\n(?=.*^\1$)', '', text)
        
        # Clean broken unicode and OCR artifacts
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff\uFFFD]', '', text)
        
        # Normalize whitespace (replace multiple spaces/newlines but preserve intentional paragraph breaks)
        text = re.sub(r'[ \t]+', ' ', text)
        
        # Merge split paragraphs: if a line ends with a comma, or a lowercase letter, and next line starts with lowercase, merge
        text = re.sub(r'([a-z,])\n([a-z])', r'\1 \2', text)
        
        # Preserve legal structures (Sections, Articles, Chapters, Subsections)
        # Ensure they start on a new line
        text = re.sub(r'(?<!\n)(Section \d+[A-Z]?(?:\(\d+\))?|Article \d+[A-Z]?|Chapter [IVXLCDM\d]+)', r'\n\1', text)
        
        # Ensure citations remain intact (e.g., [2021] 10 SCC 1)
        text = re.sub(r'(\[\d{4}\]\s+\d+\s+[A-Z]+\s+\d+)', r' \1 ', text)
        
        # Clean up excessive newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text.strip()

class MetadataExtractor:
    @staticmethod
    def extract(text: str, filename: str) -> Dict[str, Any]:
        metadata = {
            "title": filename,
            "category": "Uncategorized",
            "act": None,
            "chapter": None,
            "section": None,
            "article": None,
            "jurisdiction": "India",
            "year": None,
            "language": "en",
            "bench": None,
            "judge": None,
            "petitioner": None,
            "respondent": None,
            "citation": None,
            "legal_domain": "General"
        }
        
        # Advanced Extractions
        bench_match = re.search(r'(Division|Full|Constitution)\s+Bench', text, re.IGNORECASE)
        if bench_match: metadata["bench"] = bench_match.group(0).strip()
            
        judge_match = re.search(r'(?:Hon\'ble\s+)?(?:Mr\.|Mrs\.|Ms\.|Justice)\s+([A-Z][a-z]+(?:\s+[A-Z]\.?)*\s+[A-Z][a-z]+)', text)
        if judge_match: metadata["judge"] = judge_match.group(1).strip()
            
        petitioner_match = re.search(r'(?i)petitioner[s]?\s*:\s*([^\n]+)', text)
        if petitioner_match: metadata["petitioner"] = petitioner_match.group(1).strip()
            
        respondent_match = re.search(r'(?i)respondent[s]?\s*:\s*([^\n]+)', text)
        if respondent_match: metadata["respondent"] = respondent_match.group(1).strip()
            
        citation_match = re.search(r'\[\d{4}\]\s+\d+\s+[A-Z]+\s+\d+', text)
        if citation_match: metadata["citation"] = citation_match.group(0).strip()
        
        # Extract Act
        act_match = re.search(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+Act)', text)
        if act_match:
            metadata["act"] = act_match.group(1)
            
        # Extract Year
        year_match = re.search(r'\b(18\d{2}|19\d{2}|20\d{2})\b', text)
        if year_match:
            metadata["year"] = int(year_match.group(1))
            
        # Determine category based on heuristics
        lower_text = text.lower()
        if "constitution of india" in lower_text:
            metadata["category"] = "Constitution"
        elif "bharatiya nyaya sanhita" in lower_text or "bns" in lower_text:
            metadata["category"] = "BNS"
        elif "bharatiya nagarik suraksha sanhita" in lower_text or "bnss" in lower_text:
            metadata["category"] = "BNSS"
        elif "bharatiya sakshya adhiniyam" in lower_text or "bsa" in lower_text:
            metadata["category"] = "BSA"
        elif "supreme court of india" in lower_text:
            metadata["category"] = "Supreme Court Judgments"
        elif "high court" in lower_text:
            metadata["category"] = "High Court Judgments"
        elif "contract" in lower_text and "agreement" in lower_text:
            metadata["category"] = "Contracts"
        elif "gazette" in lower_text:
            metadata["category"] = "Gazette documents"
        elif "notification" in lower_text:
            metadata["category"] = "Notifications"
        elif "circular" in lower_text:
            metadata["category"] = "Circulars"
            
        return metadata
