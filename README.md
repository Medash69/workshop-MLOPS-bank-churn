# 🏦 Bank Churn Prediction - MLOps Workshop Complet

[![CI/CD Pipeline](https://github.com/Medash69/workshop-MLOPS-bank-churn/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/Medash69/workshop-MLOPS-bank-churn/actions/workflows/ci-cd.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.29.0-FF4B4B.svg)](https://streamlit.io/)
[![Azure](https://img.shields.io/badge/Azure-Container%20Apps-0078D4.svg)](https://azure.microsoft.com/)

---

## 📋 Table des Matières

1. [Présentation du Projet](#-présentation-du-projet)
2. [Architecture](#-architecture)
3. [URLs de Production](#-urls-de-production)
4. [Prérequis](#-prérequis)
5. [Installation Complète](#-installation-complète)
6. [Entraînement du Modèle](#-entraînement-du-modèle)
7. [Lancement de l'Application](#-lancement-de-lapplication)
8. [API FastAPI](#-api-fastapi)
9. [Dashboard Streamlit](#-dashboard-streamlit)
10. [Détection de Data Drift](#-détection-de-data-drift)
11. [Conteneurisation Docker](#-conteneurisation-docker)
12. [Déploiement Azure](#-déploiement-azure)
13. [Pipeline CI/CD GitHub Actions](#-pipeline-cicd-github-actions)
14. [MLflow Tracking](#-mlflow-tracking)
15. [Tests](#-tests)
16. [Structure du Projet](#-structure-du-projet)
17. [Commandes Utiles](#-commandes-utiles)
18. [Dépannage](#-dépannage)

---

## 🎯 Présentation du Projet

Ce projet MLOps complet prédit le **churn client bancaire** (risque de départ) en utilisant un modèle de Machine Learning. Il inclut :

- **Modèle ML** : Random Forest Classifier avec tracking MLflow
- **API REST** : FastAPI avec endpoints de prédiction
- **Dashboard** : Interface Streamlit interactive
- **Monitoring** : Détection de data drift
- **CI/CD** : Pipeline automatisé GitHub Actions
- **Cloud** : Déploiement sur Azure Container Apps

### Contexte Business
Une banque souhaite prédire quels clients risquent de partir pour proposer des actions de rétention proactives.

### Dataset
- **10 features** : CreditScore, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary, Geography_Germany, Geography_Spain
- **Target** : Exited (0 = reste, 1 = part)
- **Taille** : 10,000 clients

---

## 🏗 Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   GitHub Repo   │───▶│ GitHub Actions  │───▶│  Azure ACR      │
│                 │    │   (CI/CD)       │    │  (Container     │
│  - Code         │    │  - Tests        │    │   Registry)     │
│  - Dockerfile   │    │  - Build        │    │                 │
│  - Workflows    │    │  - Deploy       │    │                 │
└─────────────────┘    └─────────────────┘    └────────┬────────┘
                                                       │
                                                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Streamlit     │◀───│ Azure Container │◀───│   FastAPI       │
│   Dashboard     │    │     Apps        │    │     API         │
│   :8501         │    │                 │    │    :8000        │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

---

## 🌐 URLs de Production

| Service | URL |
|---------|-----|
| **📊 Dashboard Streamlit** | https://bank-churn-dashboard.ashywater-496e8508.swedencentral.azurecontainerapps.io |
| **🔌 API FastAPI** | https://bank-churn.ashywater-496e8508.swedencentral.azurecontainerapps.io |
| **📚 API Documentation** | https://bank-churn.ashywater-496e8508.swedencentral.azurecontainerapps.io/docs |
| **📦 GitHub Repository** | https://github.com/Medash69/workshop-MLOPS-bank-churn |

---

## 📦 Prérequis

### Logiciels Requis

| Logiciel | Version | Téléchargement |
|----------|---------|----------------|
| Python | 3.9+ | https://www.python.org/downloads/ |
| Git | Latest | https://git-scm.com/downloads |
| Docker Desktop | Latest | https://www.docker.com/products/docker-desktop |
| Azure CLI | Latest | https://docs.microsoft.com/cli/azure/install-azure-cli |
| VS Code | Latest | https://code.visualstudio.com/ |

### Vérification des Installations

```bash
# Python
python --version
# Doit afficher: Python 3.9.x ou supérieur

# Git
git --version

# Docker
docker --version
docker ps

# Azure CLI
az --version
```

### Comptes Nécessaires

- **GitHub** : https://github.com/signup
- **Azure for Students (100$)** : https://azure.microsoft.com/students

---

## 🚀 Installation Complète

### Étape 1 : Cloner le Repository

```bash
# Cloner le projet
git clone https://github.com/Medash69/workshop-MLOPS-bank-churn.git

# Aller dans le dossier
cd workshop-MLOPS-bank-churn
```

### Étape 2 : Créer l'Environnement Virtuel

```bash
# Créer l'environnement virtuel
python -m venv venv

# Activer l'environnement (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# Activer l'environnement (Windows CMD)
venv\Scripts\activate.bat

# Activer l'environnement (Linux/Mac)
source venv/bin/activate
```

### Étape 3 : Installer les Dépendances

```bash
# Mettre à jour pip
pip install --upgrade pip

# Installer toutes les dépendances
pip install -r requirements.txt
```

### Étape 4 : Générer les Données (si nécessaire)

```bash
# Générer le dataset synthétique
python generate_data.py
```

---

## 🎓 Entraînement du Modèle

### Lancer l'Entraînement

```bash
# Entraîner le modèle Random Forest
python train_model.py
```

### Résultat Attendu

```
Chargement des donnees...
Dataset : 10000 lignes, 11 colonnes
Taux de churn : 23.45%

Train : 8000 lignes
Test : 2000 lignes

Entrainement du modele...

==================================================
RESULTATS DE L'ENTRAINEMENT
==================================================
Accuracy  : 0.8650
Precision : 0.7823
Recall    : 0.6542
F1 Score  : 0.7125
ROC AUC   : 0.8934
==================================================

Modele sauvegarde dans : model/churn_model.pkl
MLflow UI : mlflow ui --port 5000
```

### Visualiser les Expériences MLflow

```bash
# Lancer l'interface MLflow
mlflow ui --port 5000

# Ouvrir dans le navigateur
# http://localhost:5000
```

---

## ▶️ Lancement de l'Application

### Option 1 : Mode Local Simple (Recommandé pour le développement)

```bash
# Terminal 1 : Lancer l'API FastAPI
uvicorn app.main:app --reload --port 8000

# Terminal 2 : Lancer le Dashboard Streamlit
streamlit run streamlit_app.py --server.port 8501
```

### Option 2 : Utiliser les Scripts de Lancement

**Windows (PowerShell ou CMD) :**
```batch
# Lancer API + Dashboard
start.bat local

# Lancer uniquement l'API
start.bat api-only

# Lancer uniquement le Dashboard
start.bat streamlit-only
```

**Linux/Mac :**
```bash
# Rendre le script exécutable
chmod +x start.sh

# Lancer API + Dashboard
./start.sh local

# Lancer uniquement l'API
./start.sh api-only

# Lancer uniquement le Dashboard
./start.sh streamlit-only
```

### Option 3 : Avec Docker Compose

```bash
# Construire et lancer tous les services
docker-compose up --build

# Lancer en arrière-plan
docker-compose up --build -d

# Arrêter les services
docker-compose down
```

### URLs Locales

| Service | URL |
|---------|-----|
| API FastAPI | http://localhost:8000 |
| API Documentation (Swagger) | http://localhost:8000/docs |
| API Documentation (ReDoc) | http://localhost:8000/redoc |
| Dashboard Streamlit | http://localhost:8501 |
| MLflow UI | http://localhost:5000 |

---

## 🔌 API FastAPI

### Endpoints Disponibles

| Méthode | Endpoint | Description |
|---------|----------|-------------|
| GET | `/` | Informations sur l'API |
| GET | `/health` | Health check |
| POST | `/predict` | Prédiction simple |
| POST | `/predict/batch` | Prédictions en lot |
| POST | `/drift/check` | Vérification du drift |

### Exemple : Health Check

```bash
curl http://localhost:8000/health
```

**Réponse :**
```json
{
  "status": "healthy",
  "is_model_active": true
}
```

### Exemple : Prédiction Simple

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

**Réponse :**
```json
{
  "churn_probability": 0.2345,
  "prediction": 0,
  "risk_level": "Low"
}
```

### Exemple avec Python

```python
import requests

url = "http://localhost:8000/predict"
data = {
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
}

response = requests.post(url, json=data)
print(response.json())
```

### Vérification du Drift

```bash
curl -X POST "http://localhost:8000/drift/check?threshold=0.05"
```

---

## 📊 Dashboard Streamlit

Le dashboard offre 5 pages interactives :

### 1. 🎯 Prédiction
- Formulaire interactif pour entrer les caractéristiques client
- Affichage de la probabilité de churn
- Jauge visuelle du risque
- Recommandations personnalisées

### 2. 📊 Exploration des Données
- Statistiques descriptives
- Distribution du churn
- Histogrammes par feature
- Matrice de corrélation

### 3. ⚠️ Détection de Drift
- Analyse des changements de distribution
- Test de Kolmogorov-Smirnov
- Visualisation comparative
- Alertes et recommandations

### 4. 📈 Métriques du Modèle
- Accuracy, Precision, Recall, F1, ROC AUC
- Historique des entraînements MLflow
- Matrice de confusion
- Feature importance

### 5. 🔧 Configuration
- État des services (API, modèle)
- Actions de maintenance
- Génération de données de test

---

## ⚠️ Détection de Data Drift

### Utilisation via Python

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

# Générer le rapport
report = detector.generate_report(results)

print(f"Risk Level: {report['summary']['risk_level']}")
print(f"Drifted Features: {report['summary']['drifted_features']}")
```

### Générer des Données de Production Simulées

```python
from app.drift_detect import generate_drift_data

generate_drift_data(
    reference_file="data/bank_churn.csv",
    output_file="data/production_data.csv",
    drift_features={
        "Age": {"shift": 5, "scale": 1.0},
        "Balance": {"shift": 0, "scale": 1.2},
        "CreditScore": {"shift": -30, "scale": 1.0}
    }
)
```

---

## 🐳 Conteneurisation Docker

### Construire l'Image de l'API

```bash
# Construire l'image
docker build -t bank-churn-api:v1 .

# Vérifier l'image
docker images bank-churn-api

# Lancer le conteneur
docker run -d -p 8000:8000 --name churn-api bank-churn-api:v1

# Voir les logs
docker logs churn-api

# Arrêter et supprimer
docker stop churn-api
docker rm churn-api
```

### Construire l'Image Streamlit

```bash
# Construire l'image Streamlit
docker build -f Dockerfile.streamlit -t bank-churn-streamlit:v1 .

# Lancer le conteneur
docker run -d -p 8501:8501 --name churn-dashboard bank-churn-streamlit:v1
```

### Docker Compose (Tous les Services)

```bash
# Construire et lancer
docker-compose up --build

# Lancer en arrière-plan
docker-compose up -d

# Voir les logs
docker-compose logs -f

# Arrêter
docker-compose down

# Arrêter et supprimer les volumes
docker-compose down -v
```

---

## ☁️ Déploiement Azure

### Configuration Azure

| Ressource | Valeur |
|-----------|--------|
| Resource Group | `rg-nlp-deployment` |
| Container Registry (ACR) | `mlopsashash` |
| Container Apps Environment | `env-nlp` |
| API Container App | `bank-churn` |
| Dashboard Container App | `bank-churn-dashboard` |
| Région | `swedencentral` |

### Commandes Azure CLI

#### Connexion à Azure

```bash
# Se connecter
az login

# Vérifier l'abonnement
az account show

# Lister les ressources
az group list -o table
```

#### Gérer le Container Registry

```bash
# Se connecter à l'ACR
az acr login --name mlopsashash

# Lister les images
az acr repository list --name mlopsashash -o table

# Voir les tags d'une image
az acr repository show-tags --name mlopsashash --repository bank-churn-api
```

#### Gérer les Container Apps

```bash
# Lister les Container Apps
az containerapp list --resource-group rg-nlp-deployment -o table

# Voir les logs de l'API
az containerapp logs show \
  --name bank-churn \
  --resource-group rg-nlp-deployment \
  --tail 100

# Voir les logs du Dashboard
az containerapp logs show \
  --name bank-churn-dashboard \
  --resource-group rg-nlp-deployment \
  --tail 100

# Redémarrer une app
az containerapp revision restart \
  --name bank-churn \
  --resource-group rg-nlp-deployment

# Obtenir l'URL
az containerapp show \
  --name bank-churn \
  --resource-group rg-nlp-deployment \
  --query properties.configuration.ingress.fqdn -o tsv
```

#### Déploiement Manuel

```bash
# Build et push de l'image
docker build -t mlopsashash.azurecr.io/bank-churn-api:latest .
az acr login --name mlopsashash
docker push mlopsashash.azurecr.io/bank-churn-api:latest

# Mettre à jour la Container App
az containerapp update \
  --name bank-churn \
  --resource-group rg-nlp-deployment \
  --image mlopsashash.azurecr.io/bank-churn-api:latest
```

---

## 🔄 Pipeline CI/CD GitHub Actions

### Secrets GitHub à Configurer

Allez sur : **https://github.com/Medash69/workshop-MLOPS-bank-churn/settings/secrets/actions**

| Secret | Description |
|--------|-------------|
| `AZURE_CREDENTIALS` | JSON du Service Principal Azure |
| `ACR_USERNAME` | Username du Container Registry (`mlopsashash`) |
| `ACR_PASSWORD` | Password du Container Registry |

### Créer le Service Principal Azure

```bash
az ad sp create-for-rbac \
  --name "github-mlops-sp" \
  --role contributor \
  --scopes /subscriptions/924feefb-f89f-423a-a62f-3d81583d01da \
  --json-auth
```

### Récupérer les Credentials ACR

```bash
# Username
az acr credential show --name mlopsashash --query username -o tsv

# Password
az acr credential show --name mlopsashash --query "passwords[0].value" -o tsv
```

### Structure du Pipeline

```yaml
# .github/workflows/ci-cd.yml

Jobs:
1. test                        # Exécute pytest avec couverture
2. build-and-deploy-api        # Build et déploie l'API
3. build-and-deploy-streamlit  # Build et déploie Streamlit
```

### Déclencheurs

- **Push sur main** : Déploiement automatique
- **Pull Request** : Tests uniquement
- **Manual** : Via workflow_dispatch

### Relancer le Pipeline Manuellement

1. Aller sur **Actions** dans GitHub
2. Cliquer sur **CI/CD Pipeline**
3. Cliquer sur **Run workflow**

---

## 📈 MLflow Tracking

### Lancer l'Interface MLflow

```bash
mlflow ui --port 5000
```

### Voir les Expériences

```python
import mlflow

# Configurer le tracking
mlflow.set_tracking_uri("./mlruns")

# Lister les expériences
experiments = mlflow.search_experiments()
for exp in experiments:
    print(f"{exp.name}: {exp.experiment_id}")

# Lister les runs
runs = mlflow.search_runs(experiment_ids=["159076234787646138"])
print(runs[['run_id', 'metrics.accuracy', 'metrics.f1_score']])
```

### Charger un Modèle depuis MLflow

```python
import mlflow.sklearn

# Charger le modèle enregistré
model = mlflow.sklearn.load_model("models:/bank-churn-classifier/latest")

# Faire une prédiction
prediction = model.predict([[650, 35, 5, 50000, 2, 1, 1, 75000, 0, 1]])
```

---

## 🧪 Tests

### Exécuter Tous les Tests

```bash
# Tests simples
pytest tests/ -v

# Tests avec couverture
pytest tests/ -v --cov=app --cov-report=term

# Rapport HTML de couverture
pytest tests/ -v --cov=app --cov-report=html

# Ouvrir le rapport (Windows)
start htmlcov/index.html

# Ouvrir le rapport (Mac)
open htmlcov/index.html
```

### Structure des Tests

```
tests/
├── test_api.py           # Tests des endpoints API
├── test_model.py         # Tests du modèle ML (à créer)
└── test_drift.py         # Tests de détection de drift (à créer)
```

---

## 📁 Structure du Projet

```
bank-churn-mlops/
│
├── 📂 .github/
│   └── 📂 workflows/
│       └── ci-cd.yml              # Pipeline CI/CD
│
├── 📂 app/                        # Code de l'API
│   ├── __init__.py
│   ├── main.py                    # Endpoints FastAPI
│   ├── models.py                  # Schémas Pydantic
│   ├── drift_detect.py            # Détection de drift
│   ├── drift_data_gen.py          # Génération de données
│   └── utils.py                   # Fonctions utilitaires
│
├── 📂 data/                       # Données
│   ├── bank_churn.csv             # Dataset d'entraînement
│   └── production_data.csv        # Données de production
│
├── 📂 model/                      # Modèles sauvegardés
│   └── churn_model.pkl            # Modèle Random Forest
│
├── 📂 mlruns/                     # Expériences MLflow
│   └── ...
│
├── 📂 tests/                      # Tests unitaires
│   └── test_api.py
│
├── 📂 drift_reports/              # Rapports de drift
│   └── *.json
│
├── 📄 streamlit_app.py            # Dashboard Streamlit
├── 📄 train_model.py              # Script d'entraînement
├── 📄 generate_data.py            # Génération du dataset
│
├── 🐳 Dockerfile                  # Image Docker API
├── 🐳 Dockerfile.streamlit        # Image Docker Streamlit
├── 🐳 docker-compose.yml          # Orchestration Docker
├── 🐳 .dockerignore               # Fichiers ignorés par Docker
│
├── 📄 requirements.txt            # Dépendances Python
├── 📄 start.bat                   # Script Windows
├── 📄 start.sh                    # Script Linux/Mac
├── 📄 .gitignore                  # Fichiers ignorés par Git
└── 📄 README.md                   # Ce fichier
```

---

## 🛠 Commandes Utiles

### Commandes de Démarrage Rapide

```bash
# 1. Cloner le projet
git clone https://github.com/Medash69/workshop-MLOPS-bank-churn.git
cd workshop-MLOPS-bank-churn

# 2. Créer et activer l'environnement virtuel
python -m venv venv
.\venv\Scripts\Activate.ps1  # Windows PowerShell

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Entraîner le modèle
python train_model.py

# 5. Lancer l'API (Terminal 1)
uvicorn app.main:app --reload --port 8000

# 6. Lancer le Dashboard (Terminal 2)
streamlit run streamlit_app.py --server.port 8501
```

### Git

```bash
# Voir le statut
git status

# Ajouter tous les fichiers
git add -A

# Committer
git commit -m "votre message"

# Pousser sur GitHub
git push origin main

# Récupérer les dernières modifications
git pull origin main
```

### Python/Pip

```bash
# Installer les dépendances
pip install -r requirements.txt

# Mettre à jour une dépendance
pip install --upgrade <package>

# Sauvegarder les dépendances
pip freeze > requirements.txt
```

### Docker

```bash
# Lister les conteneurs
docker ps -a

# Lister les images
docker images

# Supprimer un conteneur
docker rm <container_id>

# Supprimer une image
docker rmi <image_id>

# Nettoyer les ressources inutilisées
docker system prune -a
```

### Azure

```bash
# Se connecter
az login

# Voir les logs
az containerapp logs show --name bank-churn --resource-group rg-nlp-deployment --tail 50

# Redémarrer l'app
az containerapp revision restart --name bank-churn --resource-group rg-nlp-deployment
```

---

## 🔧 Dépannage

### Problème : Le modèle n'est pas trouvé

```bash
# Solution : Entraîner le modèle
python train_model.py
```

### Problème : L'API ne répond pas

```bash
# Vérifier si le port est utilisé
netstat -ano | findstr :8000

# Tuer le processus (Windows)
taskkill /PID <PID> /F

# Relancer l'API
uvicorn app.main:app --reload --port 8000
```

### Problème : Erreur Docker "port already in use"

```bash
# Arrêter tous les conteneurs
docker stop $(docker ps -aq)

# Relancer
docker-compose up --build
```

### Problème : Tests échouent

```bash
# Vérifier que le modèle existe
python train_model.py

# Relancer les tests
pytest tests/ -v
```

### Problème : Déploiement Azure échoue

```bash
# Vérifier les secrets GitHub
# https://github.com/Medash69/workshop-MLOPS-bank-churn/settings/secrets/actions

# Vérifier la connexion Azure
az login
az account show

# Vérifier l'ACR
az acr login --name mlopsashash
```

---

## 📝 Variables d'Environnement

| Variable | Description | Défaut |
|----------|-------------|--------|
| `MODEL_PATH` | Chemin vers le modèle | `model/churn_model.pkl` |
| `API_URL` | URL de l'API FastAPI | `http://localhost:8000` |
| `APPLICATIONINSIGHTS_CONNECTION_STRING` | Azure App Insights | - |

---

## 📊 Métriques du Modèle

| Métrique | Valeur |
|----------|--------|
| Accuracy | ~0.86 |
| Precision | ~0.78 |
| Recall | ~0.65 |
| F1 Score | ~0.71 |
| ROC AUC | ~0.89 |

---

## 👨‍💻 Auteur

**Workshop MLOps avec Azure**
- GitHub : https://github.com/Medash69

---

## 📄 Licence

Ce projet est sous licence MIT.

---

## 🙏 Remerciements

- FastAPI pour le framework API
- Streamlit pour le dashboard
- MLflow pour le tracking
- Azure pour l'hébergement cloud
- Scikit-learn pour le modèle ML

---

**Dernière mise à jour :** Janvier 2026
