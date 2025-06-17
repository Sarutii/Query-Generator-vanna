@echo off
echo ================================
echo Starting Flask App...
echo ================================
start cmd /k python "Latest app V5.py"

timeout /t 5 >nul

echo ================================
echo Starting ngrok on port 8080...
echo ================================
start cmd /k ngrok http 8080
