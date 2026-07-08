import os
import re
import json
import csv
from typing import Dict, Any, List

class DocumentParser:
    """Parses various document formats into raw text."""
    
    @staticmethod
    def parse_txt(file_path: str) -> str:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    @staticmethod
    def parse_json(file_path: str) -> str:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return json.dumps(data, indent=2)

    @staticmethod
    def parse_csv(file_path: str) -> str:
        content = []
        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                content.append(" | ".join(row))
        return "\n".join(content)

    @staticmethod
    def parse_html(file_path: str) -> str:
        try:
            from bs4 import BeautifulSoup
            with open(file_path, "r", encoding="utf-8") as f:
                soup = BeautifulSoup(f.read(), "html.parser")
                return soup.get_text(separator="\n")
        except ImportError:
            return DocumentParser.parse_txt(file_path)

    @staticmethod
    def parse_pdf(file_path: str) -> str:
        try:
            import fitz # PyMuPDF
            text = ""
            with fitz.open(file_path) as doc:
                for page in doc:
                    text += page.get_text() + "\n"
            return text
        except ImportError:
            return f"[PDF Parsing requires PyMuPDF] Simulated output for {file_path}"

    @staticmethod
    def parse_docx(file_path: str) -> str:
        try:
            import docx
            doc = docx.Document(file_path)
            return "\n".join([p.text for p in doc.paragraphs])
        except ImportError:
            return f"[DOCX Parsing requires python-docx] Simulated output for {file_path}"

    @classmethod
    def parse(cls, file_path: str) -> str:
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".txt":
            return cls.parse_txt(file_path)
        elif ext == ".json":
            return cls.parse_json(file_path)
        elif ext == ".csv":
            return cls.parse_csv(file_path)
        elif ext in [".htm", ".html"]:
            return cls.parse_html(file_path)
        elif ext == ".pdf":
            return cls.parse_pdf(file_path)
        elif ext == ".docx":
            return cls.parse_docx(file_path)
        else:
            return cls.parse_txt(file_path)
