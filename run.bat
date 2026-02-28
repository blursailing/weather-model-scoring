@echo off
echo Starting Weather Model Scoring...
python -m streamlit run blur_weather/app.py --server.port 8501
pause
