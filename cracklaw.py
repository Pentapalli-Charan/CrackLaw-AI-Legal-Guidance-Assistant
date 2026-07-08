import os
import sys
import argparse
import subprocess
import signal
import time
import urllib.request
import webbrowser
import platform

# ---------------------------------------------------------
# UI & STYLING
# ---------------------------------------------------------
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

def print_status(msg, status="INFO"):
    if status == "PASS":
        print(f"{Colors.GREEN}[OK] {msg}{Colors.RESET}")
    elif status == "WARNING":
        print(f"{Colors.YELLOW}[WARN] {msg}{Colors.RESET}")
    elif status == "FAIL":
        print(f"{Colors.RED}[FAIL] {msg}{Colors.RESET}")
    elif status == "HEADER":
        print(f"\n{Colors.BOLD}{Colors.BLUE}=== {msg} ==={Colors.RESET}")
    else:
        print(msg)

# ---------------------------------------------------------
# SYSTEM DIAGNOSTICS & PROCESS MANAGEMENT
# ---------------------------------------------------------
def run_command_silent(cmd):
    try:
        return subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except Exception:
        return None

def kill_process_on_port(port):
    if platform.system() == "Windows":
        result = run_command_silent(f"netstat -ano | findstr :{port}")
        if result and result.stdout:
            lines = result.stdout.strip().split('\n')
            for line in lines:
                if f":{port}" in line and "LISTENING" in line:
                    pid = line.strip().split()[-1]
                    print_status(f"Port {port} is occupied by PID {pid}. Terminating...", "WARNING")
                    run_command_silent(f"taskkill /PID {pid} /F")
                    time.sleep(1)
    else:
        result = run_command_silent(f"lsof -t -i:{port}")
        if result and result.stdout:
            pids = result.stdout.strip().split('\n')
            for pid in pids:
                print_status(f"Port {port} is occupied by PID {pid}. Terminating...", "WARNING")
                run_command_silent(f"kill -9 {pid}")
                time.sleep(1)

def wait_for_url(url, timeout=30):
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response:
                if response.getcode() == 200:
                    return True
        except Exception:
            pass
        time.sleep(1)
    return False

def check_dependencies():
    print_status("CrackLaw v1.0", "HEADER")
    print_status("Running pre-flight checks...\n")
    
    all_passed = True
    
    # 1. Python
    if sys.version_info >= (3, 8):
        print_status(f"Python {sys.version_info.major}.{sys.version_info.minor}", "PASS")
    else:
        print_status("Python 3.8+ required", "FAIL")
        all_passed = False
        
    # 2. Node & NPM
    node_res = run_command_silent("node -v")
    if node_res and node_res.returncode == 0:
        print_status(f"Node.js {node_res.stdout.strip()}", "PASS")
    else:
        print_status("Node.js not found", "FAIL")
        all_passed = False
        
    npm_res = run_command_silent("npm -v")
    if npm_res and npm_res.returncode == 0:
        print_status(f"npm {npm_res.stdout.strip()}", "PASS")
    else:
        print_status("npm not found", "FAIL")
        all_passed = False

    # 3. Dependencies & Directories
    if os.path.exists("node_modules"):
        print_status("Frontend Dependencies", "PASS")
    else:
        print_status("Frontend Dependencies (run 'npm install')", "FAIL")
        all_passed = False

    if os.path.exists("config/config.json"):
        print_status("Configuration (config.json)", "PASS")
    else:
        print_status("Configuration (config.json missing)", "FAIL")
        all_passed = False

    if os.path.exists("datasets"):
        print_status("Dataset Directories", "PASS")
    else:
        print_status("Dataset Directories missing", "WARNING")

    # We assume tokenizer or checkpoints might not exist on a fresh clone, but we warn.
    if os.path.exists("models/checkpoints"):
        print_status("Model Checkpoint Directory", "PASS")
    else:
        print_status("Model Checkpoints (Not found, requires training)", "WARNING")

    return all_passed

# ---------------------------------------------------------
# COMMAND ROUTING
# ---------------------------------------------------------
processes = []

def cleanup(signum=None, frame=None):
    print_status("\nShutting down CrackLaw...", "HEADER")
    for p in processes:
        try:
            if platform.system() == "Windows":
                subprocess.run(f"taskkill /F /T /PID {p.pid}", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            else:
                p.terminate()
        except Exception:
            pass
    print_status("All processes terminated.", "PASS")
    sys.exit(0)

signal.signal(signal.SIGINT, cleanup)
signal.signal(signal.SIGTERM, cleanup)

def cmd_start():
    if not check_dependencies():
        print_status("Pre-flight checks failed. Cannot start.", "FAIL")
        sys.exit(1)
        
    print_status("\nStarting Backend...", "HEADER")
    kill_process_on_port(8000)
    
    # We pipe stdout to avoid clutter, but allow errors to pass
    backend_p = subprocess.Popen(["python", "-m", "uvicorn", "src.api.main:app", "--reload", "--port", "8000"],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    processes.append(backend_p)
    
    if wait_for_url("http://127.0.0.1:8000/docs", timeout=30):
        print_status("Backend Ready", "PASS")
    else:
        print_status("Backend failed to start or timed out.", "FAIL")
        cleanup()

    print_status("\nStarting Frontend...", "HEADER")
    kill_process_on_port(5173)
    
    frontend_p = subprocess.Popen("npm run dev", shell=True,
                                  stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    processes.append(frontend_p)
    
    if wait_for_url("http://localhost:5173", timeout=30):
        print_status("Frontend Ready", "PASS")
    else:
        print_status("Frontend failed to start or timed out.", "FAIL")
        cleanup()

    print_status("\nOpening Browser...", "HEADER")
    webbrowser.open("http://localhost:5173")
    
    print_status("\nCrackLaw is Ready", "PASS")
    print(f"  Frontend:  http://localhost:5173")
    print(f"  Backend:   http://127.0.0.1:8000")
    print(f"  API Docs:  http://127.0.0.1:8000/docs")
    print("\nPress Ctrl+C to shutdown.")
    
    # Monitoring loop
    while True:
        if backend_p.poll() is not None:
            print_status("Backend crashed unexpectedly!", "FAIL")
            cleanup()
        if frontend_p.poll() is not None:
            print_status("Frontend crashed unexpectedly!", "FAIL")
            cleanup()
        time.sleep(2)

def run_script(script_path):
    if not os.path.exists(script_path):
        print_status(f"Script {script_path} not found.", "FAIL")
        sys.exit(1)
    subprocess.run([sys.executable, script_path])

def cmd_train():
    print_status("Executing Training Pipeline...", "HEADER")
    run_script("scripts/ingest.py")
    run_script("scripts/chunk.py")
    run_script("scripts/train_tokenizer.py")
    run_script("scripts/build_dataset.py")
    run_script("train.py")

def cmd_evaluate():
    print_status("Executing Evaluation Engine...", "HEADER")
    run_script("scripts/evaluate_model.py")

def cmd_inference():
    print_status("Starting CLI Inference...", "HEADER")
    # Using existing generation logic if present, else fallback warning
    run_script("scripts/test_generalization.py")

def cmd_dataset():
    print_status("Executing Dataset Assembly...", "HEADER")
    run_script("scripts/ingest.py")
    run_script("scripts/chunk.py")
    run_script("scripts/build_dataset.py")

def cmd_tokenizer():
    print_status("Retraining Tokenizer...", "HEADER")
    run_script("scripts/train_tokenizer.py")

def main():
    parser = argparse.ArgumentParser(description="CrackLaw Production Launcher")
    parser.add_argument("command", choices=["start", "train", "evaluate", "inference", "dataset", "tokenizer", "doctor"], 
                        help="Command to execute")
    args = parser.parse_args()

    if args.command == "doctor":
        check_dependencies()
        print_status("Diagnostics complete.", "PASS")
    elif args.command == "start":
        cmd_start()
    elif args.command == "train":
        cmd_train()
    elif args.command == "evaluate":
        cmd_evaluate()
    elif args.command == "inference":
        cmd_inference()
    elif args.command == "dataset":
        cmd_dataset()
    elif args.command == "tokenizer":
        cmd_tokenizer()

if __name__ == "__main__":
    # Ensure ANSI colors work on Windows terminal
    if platform.system() == "Windows":
        os.system('color')
    main()
