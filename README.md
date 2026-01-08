# 🏦 Bank Churn Prediction - MLOps Workshop

[![CI/CD Pipeline](https://github.com/Medash69/workshop-MLOPS-bank-churn/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/Medash69/workshop-MLOPS-bank-churn/actions/workflows/ci-cd.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.29.0-FF4B4B.svg)](https://streamlit.io/)

Application MLOps complète pour la prédiction de churn client bancaire, déployée sur Azure avec CI/CD automatisé.

## 🎯 Fonctionnalités

- **🔮 Prédiction ML** : Modèle Random Forest pour prédire le risque de départ client
- **📊 Dashboard Streamlit** : Interface interactive pour les prédictions et visualisations
- **🔌 API REST** : Endpoints FastAPI pour l'intégration
- **⚠️ Détection de Drift** : Monitoring des changements de distribution des données
- **🚀 CI/CD** : Pipeline GitHub Actions avec déploiement Azure automatisé
- **📈 MLflow** : Tracking des expériences et versioning des modèles
- **🐳 Docker** : Conteneurisation complète

## 📁 Structure du Projet

```
bank-churn-mlops/
├── app/                          # Code de l'API
│   ├── main.py                   # Endpoints FastAPI
│   ├── models.py                 # Schémas Pydantic
│   ├── drift_detect.py           # Détection de drift
│   └── utils.py                  # Fonctions utilitaires
├── data/                         # Données
│   ├── bank_churn.csv            # Dataset d'entraînement
│   └── production_data.csv       # Données de production
├── model/                        # Modèles sauvegardés
│   └── churn_model.pkl           # Modèle entraîné
├── tests/                        # Tests unitaires
├── mlruns/                       # Expériences MLflow
├── .github/workflows/            # CI/CD
│   └── ci-cd.yml                 # Pipeline GitHub Actions
├── streamlit_app.py              # Dashboard Streamlit
├── train_model.py                # Script d'entraînement
├── Dockerfile                    # Image Docker API
├── Dockerfile.streamlit          # Image Docker Dashboard
├── docker-compose.yml            # Orchestration Docker
├── requirements.txt              # Dépendances Python
├── start.bat                     # Script démarrage Windows
└── start.sh                      # Script démarrage Linux/Mac
```

## 🚀 Démarrage Rapide

### Prérequis

- Python 3.9+
- Docker Desktop (optionnel)
- Git

### Installation

```bash
# Cloner le repo
git clone https://github.com/Medash69/workshop-MLOPS-bank-churn.git
cd workshop-MLOPS-bank-churn

# Créer un environnement virtuel
python -m venv venv

# Activer l'environnement
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Installer les dépendances
pip install -r requirements.txt
```

### Entraînement du Modèle

```bash
python train_model.py
```

### Démarrage des Services

#### Option 1 : Mode Local (Windows)
```batch
start.bat local
```

#### Option 2 : Mode Local (Linux/Mac)
```bash
chmod +x start.sh
./start.sh local
```

#### Option 3 : Avec Docker
```bash
docker-compose up --build
```

### Accès aux Services

| Service | URL | Description |
|---------|-----|-------------|
| API FastAPI | http://localhost:8000 | API REST |
| API Docs | http://localhost:8000/docs | Documentation Swagger |
| Dashboard | http://localhost:8501 | Interface Streamlit |
| MLflow UI | http://localhost:5000 | Tracking des expériences |

## 📡 API Endpoints

### Prédiction Simple
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "CreditScore": 650,
    "Age": 35,
    "Tenure": 5,
    "Balance": 50000,
    "NumOfProducts": 2,
    "HasCrCard": 1,
    "IsActiveMember": 1,
    "EstimatedSalary": 75000,
    "Geography_Germany": 0,
    "Geography_Spain": 1
  }'
```

### Réponse
```json
{
  "churn_probability": 0.2345,
  "prediction": 0,
  "risk_level": "Low"
}
```

### Health Check
```bash
curl http://localhost:8000/health
```

### Vérification du Drift
```bash
curl -X POST "http://localhost:8000/drift/check?threshold=0.05"
```

## 📊 Dashboard Streamlit

Le dashboard offre 5 sections :

1. **🎯 Prédiction** : Formulaire interactif pour prédire le churn
2. **📊 Exploration** : Visualisations des données (distributions, corrélations)
3. **⚠️ Détection de Drift** : Analyse des changements de distribution
4. **📈 Métriques** : Performance du modèle (accuracy, precision, recall, F1, AUC)
5. **🔧 Configuration** : État des services et actions de maintenance

## ⚠️ Détection de Data Drift

Le système utilise le test de Kolmogorov-Smirnov pour détecter les changements de distribution :

```python
from app.drift_detect import DriftDetector
import pandas as pd

# Initialiser le détecteur
detector = DriftDetector(threshold=0.05)

# Charger les données
ref_data = pd.read_csv("data/bank_churn.csv")
prod_data = pd.read_csv("data/production_data.csv")

# Détecter le drift
results = detector.detect_all(ref_data, prod_data)
report = detector.generate_report(results)

print(f"Risk Level: {report['summary']['risk_level']}")
print(f"Drifted Features: {report['summary']['drifted_features']}")
```

## 🚀 Déploiement Azure

### Prérequis Azure

1. Compte Azure avec abonnement actif
2. Azure CLI installé et connecté
3. Docker Desktop en cours d'exécution

### Secrets GitHub à Configurer

| Secret | Description |
|--------|-------------|
| `AZURE_CREDENTIALS` | Credentials JSON du Service Principal |
| `ACR_USERNAME` | Nom d'utilisateur du Container Registry |
| `ACR_PASSWORD` | Mot de passe du Container Registry |

### Déploiement Manuel

```bash
# Exécuter le script de déploiement
chmod +x deploy.sh
./deploy.sh
```

### CI/CD Automatique

Le pipeline GitHub Actions se déclenche automatiquement sur push vers `main` :

1. ✅ Exécution des tests
2. 🔨 Build des images Docker (API + Streamlit)
3. 📤 Push vers Azure Container Registry
4. 🚀 Déploiement sur Azure Container Apps
5. 🩺 Vérification du déploiement

## 🧪 Tests

```bash
# Exécuter tous les tests
pytest tests/ -v

# Avec couverture
pytest tests/ -v --cov=app --cov-report=term

# Rapport HTML
pytest tests/ -v --cov=app --cov-report=html
```

## 📈 MLflow Tracking

```bash
# Lancer l'UI MLflow
mlflow ui --port 5000

# Accéder à http://localhost:5000
```

## 🛠️ Commandes Utiles

```bash
# Entraîner le modèle
python train_model.py

# Générer des données de test
python generate_data.py

# Vérifier le drift
python -c "from app.drift_detect import detect_drift; detect_drift('data/bank_churn.csv', 'data/production_data.csv')"

# Docker - voir les logs
docker logs -f bank-churn-api
docker logs -f bank-churn-dashboard

# Docker - arrêter les services
docker-compose down

# Docker - nettoyer
docker-compose down --rmi all --volumes
```

## 📚 Technologies Utilisées

- **ML/Data** : scikit-learn, pandas, numpy, scipy
- **API** : FastAPI, uvicorn, pydantic
- **Dashboard** : Streamlit, Plotly
- **MLOps** : MLflow, pytest
- **Cloud** : Azure Container Apps, Azure Container Registry
- **CI/CD** : GitHub Actions
- **Conteneurisation** : Docker, Docker Compose

## 🔧 Configuration

### Variables d'Environnement

| Variable | Description | Défaut |
|----------|-------------|--------|
| `MODEL_PATH` | Chemin vers le modèle | `model/churn_model.pkl` |
| `API_URL` | URL de l'API FastAPI | `http://localhost:8000` |
| `APPLICATIONINSIGHTS_CONNECTION_STRING` | Azure Application Insights | - |

## 📝 Auteur

**Workshop MLOps avec Azure**

## 📄 Licence

Ce projet est sous licence MIT.
