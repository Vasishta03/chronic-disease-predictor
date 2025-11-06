@echo off
REM Simple batch script to run the Streamlit application on Windows

echo Starting Chronic Disease Prediction System...
echo ============================================

REM Check if Python is available
python --version >nul 2>&1 || (
    echo Python is not installed or not in PATH
    pause
    exit /b 1
)

REM Install requirements if needed
echo Installing/checking required packages...
pip install -r requirements.txt

REM Run the Streamlit app
echo Launching Streamlit dashboard...
streamlit run app.py --server.port 8501 --server.address localhost

echo Application should open in your browser at http://localhost:8501
pause