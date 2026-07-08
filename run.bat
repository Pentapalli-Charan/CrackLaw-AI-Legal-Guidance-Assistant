@echo off
SETLOCAL

IF "%1"=="start" GOTO start_app
IF "%1"=="train" GOTO train_model
GOTO help

:help
echo =======================================================
echo CRACKLAW AUTOMATION SCRIPT
echo =======================================================
echo Usage:
echo   run.bat start   - Starts the Backend API and Frontend UI
echo   run.bat train   - Runs the entire data pipeline and training
echo =======================================================
GOTO end

:start_app
echo Starting Backend API...
start cmd /k "uvicorn src.api.main:app --reload --port 8000"

echo Starting Frontend UI...
start cmd /k "cd frontend && npm run dev"

echo CrackLaw is running! Check the frontend terminal for the localhost URL.
GOTO end

:train_model
echo Step 1/5: Ingesting Raw Documents...
python scripts/ingest.py

echo Step 2/5: Chunking Documents...
python scripts/chunk.py

echo Step 3/5: Training Tokenizer...
python scripts/train_tokenizer.py

echo Step 4/5: Building Dataset...
python scripts/build_dataset.py

echo Step 5/5: Starting Model Training...
python train.py
GOTO end

:end
ENDLOCAL
