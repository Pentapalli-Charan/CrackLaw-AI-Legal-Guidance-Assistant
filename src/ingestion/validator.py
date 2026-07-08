import re

class DataValidator:
    """Validates raw text before ingestion to prevent bad data from poisoning the corpus."""

    @staticmethod
    def validate(text: str, filename: str) -> dict:
        report = {
            "is_valid": True,
            "reasons": []
        }

        if not text or len(text.strip()) == 0:
            report["is_valid"] = False
            report["reasons"].append("Empty document")
            return report

        # Encoding / Corruption Check (checking for NULL bytes or excessive weird unicode)
        if '\x00' in text:
            report["is_valid"] = False
            report["reasons"].append("Corrupted Encoding: NULL bytes detected")

        # OCR Quality Heuristic: Check alphanumeric to symbol ratio
        alphanumeric_count = sum(c.isalnum() for c in text)
        total_chars = len(text)
        if total_chars > 100 and (alphanumeric_count / total_chars) < 0.4:
            report["is_valid"] = False
            report["reasons"].append(f"Bad OCR: Alphanumeric ratio too low ({alphanumeric_count/total_chars:.2f})")

        # Check unsupported language (Naive check: mostly english/hindi characters)
        # Assumes valid legal docs in our corpus will be largely Latin or Devanagari.
        latin_or_dev = len(re.findall(r'[a-zA-Z0-9\u0900-\u097F]', text))
        if total_chars > 100 and (latin_or_dev / total_chars) < 0.2:
            report["is_valid"] = False
            report["reasons"].append("Unsupported Language or heavy corruption.")

        return report
