import re
import random

class SyntheticTaskGenerator:
    """Generates 14 distinct synthetic fine-tuning tasks from legal chunks using deterministic rule-based NLP."""

    def generate_tasks(self, chunk: str, metadata: dict = None) -> list:
        tasks = []
        if not chunk.strip(): return tasks
        
        # Determine category context
        category = metadata.get("category", "") if metadata else ""
        
        # Base tasks
        tasks.extend(self._gen_qa(chunk))
        tasks.extend(self._gen_summarization(chunk))
        tasks.extend(self._gen_definitions(chunk))
        
        # Category specific
        if "Judgment" in category:
            tasks.extend(self._gen_case_summary(chunk))
            tasks.extend(self._gen_judgment_analysis(chunk))
            tasks.extend(self._gen_issue_identification(chunk))
        
        if "Act" in chunk or category in ["BNS", "BNSS", "BSA"]:
            tasks.extend(self._gen_act_summary(chunk))
            tasks.extend(self._gen_section_explanation(chunk))
            tasks.extend(self._gen_legal_reasoning(chunk))
            
        if "Contract" in category:
            tasks.extend(self._gen_contract_clause(chunk))
            
        tasks.extend(self._gen_difference(chunk))
        tasks.extend(self._gen_compliance(chunk))
        tasks.extend(self._gen_mcq(chunk))
        tasks.extend(self._gen_scenario(chunk))
        
        return tasks

    def _gen_qa(self, chunk: str) -> list:
        tasks = []
        if "punished with" in chunk.lower() or "liable to" in chunk.lower():
            tasks.append({
                "instruction": "What is the penalty prescribed in this provision?",
                "context": chunk,
                "response": "According to the text, the penalty is specified as follows: " + chunk
            })
        return tasks

    def _gen_summarization(self, chunk: str) -> list:
        sentences = re.split(r'(?<=[.!?]) +', chunk)
        if len(sentences) > 2:
            return [{
                "instruction": "Summarize the following legal text.",
                "context": chunk,
                "response": " ".join(sentences[:2])
            }]
        return []

    def _gen_definitions(self, chunk: str) -> list:
        tasks = []
        match = re.search(r'([A-Za-z\s]+)\s+means\s+([^.]+)', chunk)
        if match:
            tasks.append({
                "instruction": f"Define '{match.group(1).strip()}'.",
                "context": chunk,
                "response": match.group(2).strip() + "."
            })
        return tasks

    def _gen_case_summary(self, chunk: str) -> list:
        if "vs." in chunk or "v." in chunk:
            return [{
                "instruction": "Provide a brief case summary based on the excerpt.",
                "context": chunk,
                "response": "This excerpt relates to the case mentioned. " + chunk
            }]
        return []
        
    def _gen_judgment_analysis(self, chunk: str) -> list:
        if "held that" in chunk.lower() or "ordered" in chunk.lower():
            return [{
                "instruction": "Analyze the court's holding in this judgment excerpt.",
                "context": chunk,
                "response": "The court analyzed the facts and held accordingly based on the provided text."
            }]
        return []

    def _gen_issue_identification(self, chunk: str) -> list:
        if "whether" in chunk.lower() and "?" in chunk:
            return [{
                "instruction": "Identify the primary legal issue.",
                "context": chunk,
                "response": "The primary issue is framed around the question presented in the text."
            }]
        return []

    def _gen_act_summary(self, chunk: str) -> list:
        if "Act" in chunk:
            return [{
                "instruction": "What is the primary objective of this act excerpt?",
                "context": chunk,
                "response": "The text outlines specific provisions governing the act."
            }]
        return []

    def _gen_section_explanation(self, chunk: str) -> list:
        if "Section" in chunk:
            return [{
                "instruction": "Explain the section mentioned in plain English.",
                "context": chunk,
                "response": "This section dictates the legal requirements and consequences as detailed."
            }]
        return []

    def _gen_legal_reasoning(self, chunk: str) -> list:
        if "provided that" in chunk.lower() or "if" in chunk.lower():
            return [{
                "instruction": "Extract the legal reasoning and conditions from this text.",
                "context": chunk,
                "response": "The application of this rule is conditional based on the proviso mentioned."
            }]
        return []

    def _gen_contract_clause(self, chunk: str) -> list:
        if "shall" in chunk and ("agree" in chunk or "party" in chunk):
            return [{
                "instruction": "Explain this contract clause.",
                "context": chunk,
                "response": "This clause imposes a mandatory obligation on the parties involved."
            }]
        return []

    def _gen_difference(self, chunk: str) -> list:
        if "notwithstanding" in chunk.lower() or "however" in chunk.lower():
            return [{
                "instruction": "Identify the distinction or exception made in this law.",
                "context": chunk,
                "response": "An exception is carved out using notwithstanding/however clauses."
            }]
        return []

    def _gen_compliance(self, chunk: str) -> list:
        if "shall be required" in chunk.lower() or "must" in chunk.lower():
            return [{
                "instruction": "What are the compliance requirements here?",
                "context": chunk,
                "response": "The text mandates strict adherence to the stated requirements."
            }]
        return []

    def _gen_mcq(self, chunk: str) -> list:
        return [{
            "instruction": "Based on the text, which of the following is true? A) It applies broadly. B) It is restricted as per text. C) It is repealed.",
            "context": chunk,
            "response": "The correct answer is based on the specific restrictions in the text."
        }]

    def _gen_scenario(self, chunk: str) -> list:
        return [{
            "instruction": "Apply this text to a hypothetical scenario.",
            "context": chunk,
            "response": "In a scenario where these conditions are met, this text would strictly apply."
        }]
