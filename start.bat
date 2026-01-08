@echo off
REM =============================================================
REM Script de démarrage Windows - Bank Churn MLOps
REM =============================================================

echo 🏦 Bank Churn MLOps - Démarrage des services
echo ==============================================

set MODE=%1
if "%MODE%"=="" set MODE=local

if "%MODE%"=="local" goto local
if "%MODE%"=="docker" goto docker
if "%MODE%"=="api-only" goto api_only
if "%MODE%"=="streamlit-only" goto streamlit_only
if "%MODE%"=="test" goto test
if "%MODE%"=="train" goto train
if "%MODE%"=="drift" goto drift
goto usage

:local
echo 🚀 Démarrage en mode LOCAL
echo.

REM Vérifier si le modèle existe
if not exist "model\churn_model.pkl" (
    echo ⚠️ Modèle non trouvé. Entraînement en cours...
    python train_model.py
)

echo Démarrage de l'API FastAPI en arrière-plan...
start /B cmd /c "uvicorn app.main:app --reload --port 8000"

echo Attente du démarrage de l'API (5s)...
timeout /t 5 /nobreak >nul

echo Démarrage du Dashboard Streamlit...
start cmd /c "streamlit run streamlit_app.py --server.port 8501"

echo.
echo ==============================================
echo ✅ Services démarrés avec succès!
echo.
echo 📡 API FastAPI:  http://localhost:8000
echo 📚 API Docs:     http://localhost:8000/docs
echo 📊 Dashboard:    http://localhost:8501
echo ==============================================
goto end

:docker
echo 🐳 Démarrage en mode DOCKER
docker-compose up --build -d
echo.
echo ✅ Conteneurs démarrés!
echo 📡 API:       http://localhost:8000
echo 📊 Dashboard: http://localhost:8501
goto end

:api_only
echo 🚀 Démarrage de l'API uniquement
uvicorn app.main:app --reload --port 8000
goto end

:streamlit_only
echo 📊 Démarrage du Dashboard uniquement
streamlit run streamlit_app.py --server.port 8501
goto end

:test
echo 🧪 Exécution des tests
pytest tests/ -v --cov=app --cov-report=term
goto end

:train
echo 🎓 Entraînement du modèle
python train_model.py
goto end

:drift
echo 🔍 Vérification du drift
python -c "from app.drift_detect import DriftDetector; import pandas as pd; d=DriftDetector(); ref=pd.read_csv('data/bank_churn.csv'); prod=pd.read_csv('data/production_data.csv'); r=d.detect_all(ref,prod); rpt=d.generate_report(r); print(f\"Risk: {rpt['summary']['risk_level']}\")"
goto end

:usage
echo.
echo Usage: start.bat [mode]
echo.
echo Modes disponibles:
echo   local          - Démarre API + Dashboard sans Docker
echo   docker         - Démarre avec Docker Compose
echo   api-only       - Démarre uniquement l'API
echo   streamlit-only - Démarre uniquement le Dashboard
echo   test           - Exécute les tests
echo   train          - Entraîne le modèle
echo   drift          - Vérifie le data drift
echo.

:end
