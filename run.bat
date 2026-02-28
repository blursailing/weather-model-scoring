@echo off
echo Starting BLUR Weather Intelligence...
python -m streamlit run blur_weather/app.py --server.port 8501
pause
