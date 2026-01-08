#!/bin/bash
# =============================================================
# Script de démarrage - Bank Churn MLOps
# =============================================================

set -e

echo "🏦 Bank Churn MLOps - Démarrage des services"
echo "=============================================="

# Vérifier si Docker est installé
if ! command -v docker &> /dev/null; then
    echo "❌ Docker n'est pas installé. Veuillez l'installer d'abord."
    exit 1
fi

# Vérifier si le modèle existe
if [ ! -f "model/churn_model.pkl" ]; then
    echo "⚠️ Modèle non trouvé. Entraînement en cours..."
    python train_model.py
fi

# Mode de démarrage
MODE=${1:-local}

case $MODE in
    local)
        echo "🚀 Démarrage en mode LOCAL (sans Docker)"
        echo ""
        echo "Démarrage de l'API FastAPI..."
        uvicorn app.main:app --reload --port 8000 &
        API_PID=$!
        
        echo "Attente du démarrage de l'API (5s)..."
        sleep 5
        
        echo "Démarrage du Dashboard Streamlit..."
        streamlit run streamlit_app.py --server.port 8501 &
        STREAMLIT_PID=$!
        
        echo ""
        echo "=============================================="
        echo "✅ Services démarrés avec succès!"
        echo ""
        echo "📡 API FastAPI:  http://localhost:8000"
        echo "📚 API Docs:     http://localhost:8000/docs"
        echo "📊 Dashboard:    http://localhost:8501"
        echo ""
        echo "Pour arrêter: Ctrl+C ou kill $API_PID $STREAMLIT_PID"
        echo "=============================================="
        
        # Attendre les processus
        wait
        ;;
    
    docker)
        echo "🐳 Démarrage en mode DOCKER"
        echo ""
        
        # Build et démarrage avec docker-compose
        docker-compose up --build -d
        
        echo ""
        echo "=============================================="
        echo "✅ Conteneurs démarrés avec succès!"
        echo ""
        echo "📡 API FastAPI:  http://localhost:8000"
        echo "📚 API Docs:     http://localhost:8000/docs"
        echo "📊 Dashboard:    http://localhost:8501"
        echo ""
        echo "Commandes utiles:"
        echo "  - Logs API:       docker logs -f bank-churn-api"
        echo "  - Logs Dashboard: docker logs -f bank-churn-dashboard"
        echo "  - Arrêter:        docker-compose down"
        echo "=============================================="
        ;;
    
    docker-monitoring)
        echo "🐳 Démarrage en mode DOCKER avec Monitoring"
        echo ""
        
        docker-compose --profile monitoring up --build -d
        
        echo ""
        echo "=============================================="
        echo "✅ Conteneurs démarrés avec succès!"
        echo ""
        echo "📡 API FastAPI:  http://localhost:8000"
        echo "📚 API Docs:     http://localhost:8000/docs"
        echo "📊 Dashboard:    http://localhost:8501"
        echo "📈 MLflow UI:    http://localhost:5000"
        echo ""
        echo "Arrêter: docker-compose --profile monitoring down"
        echo "=============================================="
        ;;
    
    api-only)
        echo "🚀 Démarrage de l'API uniquement"
        uvicorn app.main:app --reload --port 8000
        ;;
    
    streamlit-only)
        echo "📊 Démarrage du Dashboard uniquement"
        streamlit run streamlit_app.py --server.port 8501
        ;;
    
    test)
        echo "🧪 Exécution des tests"
        pytest tests/ -v --cov=app --cov-report=term
        ;;
    
    train)
        echo "🎓 Entraînement du modèle"
        python train_model.py
        ;;
    
    drift)
        echo "🔍 Vérification du drift"
        python -c "
from app.drift_detect import DriftDetector
import pandas as pd

detector = DriftDetector(threshold=0.05)
ref = pd.read_csv('data/bank_churn.csv')

try:
    prod = pd.read_csv('data/production_data.csv')
    results = detector.detect_all(ref, prod)
    report = detector.generate_report(results)
    print(f\"Risk Level: {report['summary']['risk_level']}\")
    print(f\"Drifted Features: {report['summary']['drifted_features']}\")
except FileNotFoundError:
    print('No production data found')
"
        ;;
    
    *)
        echo "Usage: $0 {local|docker|docker-monitoring|api-only|streamlit-only|test|train|drift}"
        echo ""
        echo "Modes disponibles:"
        echo "  local             - Démarre API + Dashboard sans Docker"
        echo "  docker            - Démarre avec Docker Compose"
        echo "  docker-monitoring - Docker avec MLflow UI"
        echo "  api-only          - Démarre uniquement l'API"
        echo "  streamlit-only    - Démarre uniquement le Dashboard"
        echo "  test              - Exécute les tests"
        echo "  train             - Entraîne le modèle"
        echo "  drift             - Vérifie le data drift"
        exit 1
        ;;
esac
