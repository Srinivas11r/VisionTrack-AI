@echo off
echo Starting Object Detection Backend...
cd backend
python -m uvicorn api.main:app --reload --port 8000
