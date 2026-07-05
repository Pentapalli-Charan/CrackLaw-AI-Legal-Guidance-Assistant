import subprocess
import sys
import os
import time

# Load .env file if it exists (before any subprocess inherits the environment)
def _load_dotenv():
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.exists(env_path):
        return
    try:
        from dotenv import load_dotenv
        load_dotenv(env_path, override=True)
        print(f"[Launcher] Loaded environment from .env (via python-dotenv)")
    except ImportError:
        # Fallback: manually parse KEY=VALUE lines
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, _, value = line.partition("=")
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    if key and value:
                        os.environ[key] = value
        print(f"[Launcher] Loaded environment from .env (manual parse)")

_load_dotenv()

def main():
    print("==================================================")
    print("           CRACKLAW SYSTEM LAUNCHER               ")
    print("==================================================")
    print("Starting backend and frontend services concurrently...")
    
    # Run uvicorn as a module: python -m uvicorn src.api.main:app
    backend_cmd = [sys.executable, "-m", "uvicorn", "src.api.main:app", "--reload", "--port", "8000"]
    frontend_cmd = ["npm", "run", "dev"]
    
    is_windows = os.name == 'nt'
    if is_windows:
        # On Windows, npm needs to be run using npm.cmd or via shell=True
        frontend_cmd = ["npm.cmd", "run", "dev"]
    
    processes = []
    try:
        # Start Backend
        print("\n[Launcher] Launching FastAPI Backend on http://localhost:8000 ...")
        backend_proc = subprocess.Popen(
            backend_cmd,
            stdout=None,
            stderr=None
        )
        processes.append(backend_proc)
        
        # Give the backend a brief moment to initialize
        time.sleep(1.5)
        
        # Start Frontend
        print("[Launcher] Launching React/Vite Frontend on http://localhost:5173 ...")
        frontend_proc = subprocess.Popen(
            frontend_cmd,
            stdout=None,
            stderr=None,
            shell=is_windows  # Shell=True is recommended for executing npm on Windows
        )
        processes.append(frontend_proc)
        
        print("\n==================================================")
        print(" CrackLaw is running!")
        print(" - Backend API: http://localhost:8000")
        print(" - API Docs:    http://localhost:8000/docs")
        print(" - Frontend:    http://localhost:5173")
        print(" Press Ctrl+C to terminate both servers.")
        print("==================================================")
        
        # Keep launcher running and monitor subprocesses
        while True:
            for proc in processes:
                exit_code = proc.poll()
                if exit_code is not None:
                    raise RuntimeError(f"Subprocess terminated unexpectedly with code {exit_code}")
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n[Launcher] KeyboardInterrupt received. Stopping both servers...")
    except Exception as e:
        print(f"\n[Launcher] Error encountered: {e}")
        print("[Launcher] Stopping both servers...")
    finally:
        # Ensure clean termination of all subprocesses
        for proc in processes:
            if proc.poll() is None:
                try:
                    proc.terminate()
                    # Give process a moment to exit gracefully
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    proc.kill()
                except Exception:
                    pass
        print("[Launcher] Both servers stopped successfully.")

if __name__ == "__main__":
    main()
