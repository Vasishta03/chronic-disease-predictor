#!/bin/bash
# Simple script to run the Streamlit application

echo "Starting Chronic Disease Prediction System..."

# Check if requirements are installed
python -c "import streamlit, pandas, numpy, sklearn, plotly" 2>/dev/null || {
    echo "Installing required packages..."
    pip install -r requirements.txt
}

echo "Launching Streamlit dashboard..."
streamlit run app.py --server.port 8501 --server.address localhost

echo "Application should open in your browser at http://localhost:8501"
