import os
import sys
import importlib
import json
import pkg_resources

# Ensure we can import from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import PROJECT_ROOT

def check_dependencies():
    print("Checking dependencies...")
    req_file = os.path.join(PROJECT_ROOT, "requirements.txt")
    if not os.path.exists(req_file):
        return False, ["requirements.txt not found."]
        
    missing = []
    with open(req_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"): continue
            
            pkg = line.split(">=")[0].split("==")[0]
            try:
                pkg_resources.get_distribution(pkg)
            except pkg_resources.DistributionNotFound:
                missing.append(pkg)
                
    if missing:
        return False, [f"Missing dependencies: {', '.join(missing)}"]
    return True, []

def check_imports():
    print("Checking imports...")
    modules_to_test = [
        "src.config",
        "src.llm.transformer.model",
        "src.llm.training.trainer",
        "src.llm.evaluation.evaluation_engine",
        "src.ai.ai_service",
        "src.retrieval.retrieval_service"
    ]
    
    errors = []
    for mod in modules_to_test:
        try:
            importlib.import_module(mod)
        except Exception as e:
            errors.append(f"Failed to import {mod}: {e}")
            
    return len(errors) == 0, errors

def check_data_pipeline():
    print("Checking data pipeline...")
    corpus_file = os.path.join(PROJECT_ROOT, "datasets", "corpus", "cracklaw_corpus.jsonl")
    if not os.path.exists(corpus_file):
        return False, ["Corpus file not found: datasets/corpus/cracklaw_corpus.jsonl"]
        
    try:
        with open(corpus_file, "r", encoding="utf-8") as f:
            first_line = f.readline()
            if first_line:
                json.loads(first_line)
    except Exception as e:
        return False, [f"Corpus file is invalid: {e}"]
        
    return True, []

def check_checkpoints():
    print("Checking models...")
    ckpt_dir = os.path.join(PROJECT_ROOT, "models", "checkpoints")
    if not os.path.exists(ckpt_dir):
        return False, ["Checkpoint directory not found. Have you trained yet?"]
        
    files = os.listdir(ckpt_dir)
    ckpts = [f for f in files if f.endswith(".pt")]
    
    if not ckpts:
        return False, ["No model checkpoints found. Have you trained yet?"]
        
    return True, []

def main():
    print("="*50)
    print("CrackLaw Integration Audit")
    print("="*50)
    
    results = {
        "Dependencies": check_dependencies(),
        "Imports": check_imports(),
        "Data Pipeline": check_data_pipeline(),
        "Model Checkpoints": check_checkpoints(),
    }
    
    all_passed = True
    report_lines = ["# CrackLaw Integration Audit Report\n"]
    
    for section, (passed, errors) in results.items():
        status = "PASS" if passed else "FAIL"
        if not passed: all_passed = False
        
        print(f"\n{section}: {status}")
        report_lines.append(f"## {section}: {status}")
        
        for err in errors:
            print(f"  - {err}")
            report_lines.append(f"- {err}")
            
    print("\n" + "="*50)
    if all_passed:
        print("ALL CHECKS PASSED! The system is fully integrated.")
    else:
        print("SOME CHECKS FAILED. Please review the errors above.")
        
    # Write report
    log_dir = os.path.join(PROJECT_ROOT, "logs")
    os.makedirs(log_dir, exist_ok=True)
    report_path = os.path.join(log_dir, "integration_report.md")
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
        
    print(f"\nReport written to: {report_path}")

if __name__ == "__main__":
    main()
