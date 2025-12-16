# GenArt Studio - Start Script
# Starts both backend and frontend servers

Write-Host "Starting GenArt Studio..." -ForegroundColor Green

# Activate virtual environment
& .\.venv\Scripts\Activate.ps1

# Start backend in background
Write-Host "Starting backend on http://127.0.0.1:8000..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-NoExit", "-Command", "& .\.venv\Scripts\Activate.ps1; C:/Users/JACKI/GenArt-Studio/.venv/Scripts/python.exe -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000"

# Wait a moment for backend to start
Start-Sleep -Seconds 2

# Start frontend
Write-Host "Starting frontend on http://localhost:8501..." -ForegroundColor Cyan
C:/Users/JACKI/GenArt-Studio/.venv/Scripts/python.exe -m streamlit run frontend/app.py
