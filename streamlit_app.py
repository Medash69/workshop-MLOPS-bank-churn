"""
Bank Churn Prediction - Streamlit Dashboard
Application complète avec prédictions, visualisations et monitoring de drift
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import json
import joblib
from pathlib import Path
from scipy.stats import ks_2samp
import os

# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Bank Churn Prediction",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuration de l'API
API_URL = os.getenv("API_URL", "http://localhost:8000")
MODEL_PATH = "model/churn_model.pkl"
DATA_PATH = "data/bank_churn.csv"
PROD_DATA_PATH = "data/production_data.csv"

# ============================================================
# STYLES CSS
# ============================================================

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }
    .risk-high { color: #ff4b4b; font-weight: bold; }
    .risk-medium { color: #ffa500; font-weight: bold; }
    .risk-low { color: #00cc00; font-weight: bold; }
    .stMetric { background-color: #f8f9fa; padding: 10px; border-radius: 5px; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# FONCTIONS UTILITAIRES
# ============================================================

@st.cache_resource
def load_model():
    """Charge le modèle ML"""
    try:
        model = joblib.load(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Erreur chargement modèle: {e}")
        return None

@st.cache_data
def load_data():
    """Charge les données de référence"""
    try:
        df = pd.read_csv(DATA_PATH)
        return df
    except Exception as e:
        st.error(f"Erreur chargement données: {e}")
        return None

def predict_local(features: dict, model):
    """Prédiction locale avec le modèle"""
    try:
        X = np.array([[
            features['CreditScore'],
            features['Age'],
            features['Tenure'],
            features['Balance'],
            features['NumOfProducts'],
            features['HasCrCard'],
            features['IsActiveMember'],
            features['EstimatedSalary'],
            features['Geography_Germany'],
            features['Geography_Spain']
        ]])
        proba = float(model.predict_proba(X)[0][1])
        prediction = int(proba > 0.5)
        risk = "Low" if proba < 0.3 else "Medium" if proba < 0.7 else "High"
        return {
            "churn_probability": round(proba, 4),
            "prediction": prediction,
            "risk_level": risk
        }
    except Exception as e:
        return {"error": str(e)}

def predict_api(features: dict):
    """Prédiction via l'API FastAPI"""
    try:
        response = requests.post(f"{API_URL}/predict", json=features, timeout=10)
        return response.json()
    except requests.exceptions.ConnectionError:
        return None
    except Exception as e:
        return {"error": str(e)}

def calculate_drift(ref_data: pd.DataFrame, prod_data: pd.DataFrame, threshold: float = 0.05):
    """Calcule le drift pour chaque feature"""
    results = {}
    for col in ref_data.columns:
        if col != 'Exited' and col in prod_data.columns:
            stat, p_value = ks_2samp(ref_data[col].dropna(), prod_data[col].dropna())
            results[col] = {
                'p_value': float(p_value),
                'statistic': float(stat),
                'drift_detected': bool(p_value < threshold)
            }
    return results

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.image("https://img.icons8.com/fluency/96/bank-building.png", width=80)
st.sidebar.title("🏦 Bank Churn ML")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["🎯 Prédiction", "📊 Exploration des Données", "⚠️ Détection de Drift", "📈 Métriques du Modèle", "🔧 Configuration"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ À propos")
st.sidebar.info("""
Cette application permet de:
- Prédire le churn client
- Visualiser les données
- Détecter le data drift
- Monitorer le modèle
""")

# ============================================================
# PAGE: PRÉDICTION
# ============================================================

if page == "🎯 Prédiction":
    st.markdown('<h1 class="main-header">🎯 Prédiction de Churn Client</h1>', unsafe_allow_html=True)
    
    model = load_model()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 Informations Client")
        
        credit_score = st.slider("Score Crédit", 300, 850, 650, help="Score de crédit du client (300-850)")
        age = st.slider("Âge", 18, 80, 35)
        tenure = st.slider("Ancienneté (années)", 0, 10, 5, help="Nombre d'années comme client")
        balance = st.number_input("Solde Compte (€)", 0.0, 250000.0, 50000.0, step=1000.0)
        
    with col2:
        st.subheader("📊 Détails du Compte")
        
        num_products = st.selectbox("Nombre de Produits", [1, 2, 3, 4], index=1)
        has_credit_card = st.selectbox("Carte de Crédit", ["Non", "Oui"], index=1)
        is_active = st.selectbox("Membre Actif", ["Non", "Oui"], index=1)
        salary = st.number_input("Salaire Estimé (€)", 20000.0, 200000.0, 75000.0, step=5000.0)
        
        geography = st.selectbox("Pays", ["France", "Allemagne", "Espagne"])
    
    # Préparation des features
    features = {
        "CreditScore": credit_score,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": num_products,
        "HasCrCard": 1 if has_credit_card == "Oui" else 0,
        "IsActiveMember": 1 if is_active == "Oui" else 0,
        "EstimatedSalary": salary,
        "Geography_Germany": 1 if geography == "Allemagne" else 0,
        "Geography_Spain": 1 if geography == "Espagne" else 0
    }
    
    st.markdown("---")
    
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    
    with col_btn2:
        predict_button = st.button("🔮 Prédire le Churn", use_container_width=True, type="primary")
    
    if predict_button:
        with st.spinner("Analyse en cours..."):
            # Essayer l'API d'abord, sinon utiliser le modèle local
            result = predict_api(features)
            source = "API"
            
            if result is None and model is not None:
                result = predict_local(features, model)
                source = "Local"
            
            if result and "error" not in result:
                st.markdown("---")
                
                # Affichage des résultats
                res_col1, res_col2, res_col3 = st.columns(3)
                
                with res_col1:
                    st.metric(
                        "Probabilité de Churn",
                        f"{result['churn_probability']*100:.1f}%"
                    )
                
                with res_col2:
                    pred_text = "🚨 Va partir" if result['prediction'] == 1 else "✅ Va rester"
                    st.metric("Prédiction", pred_text)
                
                with res_col3:
                    risk = result['risk_level']
                    risk_emoji = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}
                    st.metric("Niveau de Risque", f"{risk_emoji.get(risk, '')} {risk}")
                
                # Jauge de probabilité
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = result['churn_probability'] * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Probabilité de Churn (%)"},
                    gauge = {
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 30], 'color': "lightgreen"},
                            {'range': [30, 70], 'color': "yellow"},
                            {'range': [70, 100], 'color': "salmon"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 50
                        }
                    }
                ))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
                
                # Recommandations
                st.subheader("💡 Recommandations")
                if result['risk_level'] == "High":
                    st.error("""
                    **Action immédiate requise:**
                    - Contacter le client dans les 24h
                    - Proposer une offre de fidélisation personnalisée
                    - Analyser les points de friction récents
                    """)
                elif result['risk_level'] == "Medium":
                    st.warning("""
                    **Surveillance recommandée:**
                    - Planifier un appel de satisfaction
                    - Envoyer une communication personnalisée
                    - Proposer des avantages exclusifs
                    """)
                else:
                    st.success("""
                    **Client fidèle:**
                    - Maintenir la qualité de service
                    - Proposer des produits complémentaires
                    - Programme de parrainage
                    """)
                
                st.caption(f"Source: {source} | Modèle: Random Forest")
            else:
                st.error("Erreur lors de la prédiction. Vérifiez que le modèle est chargé.")

# ============================================================
# PAGE: EXPLORATION DES DONNÉES
# ============================================================

elif page == "📊 Exploration des Données":
    st.markdown('<h1 class="main-header">📊 Exploration des Données</h1>', unsafe_allow_html=True)
    
    df = load_data()
    
    if df is not None:
        # Métriques globales
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Clients", f"{len(df):,}")
        with col2:
            churn_rate = df['Exited'].mean() * 100
            st.metric("Taux de Churn", f"{churn_rate:.1f}%")
        with col3:
            st.metric("Âge Moyen", f"{df['Age'].mean():.0f} ans")
        with col4:
            st.metric("Solde Moyen", f"€{df['Balance'].mean():,.0f}")
        
        st.markdown("---")
        
        # Visualisations
        tab1, tab2, tab3 = st.tabs(["Distribution", "Corrélations", "Analyse par Feature"])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                # Distribution du churn
                fig = px.pie(df, names='Exited', title='Distribution du Churn',
                            color_discrete_sequence=['#00cc00', '#ff4b4b'])
                fig.update_traces(labels=['Reste', 'Part'])
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Distribution par âge
                fig = px.histogram(df, x='Age', color='Exited', 
                                  title='Distribution par Âge',
                                  barmode='overlay',
                                  color_discrete_sequence=['#1f77b4', '#ff4b4b'])
                st.plotly_chart(fig, use_container_width=True)
            
            # Distribution du solde
            fig = px.histogram(df, x='Balance', color='Exited',
                              title='Distribution du Solde',
                              barmode='overlay',
                              color_discrete_sequence=['#1f77b4', '#ff4b4b'])
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            # Matrice de corrélation
            corr_matrix = df.corr()
            fig = px.imshow(corr_matrix, 
                           title='Matrice de Corrélation',
                           color_continuous_scale='RdBu_r',
                           aspect='auto')
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            feature = st.selectbox("Sélectionner une feature", df.columns.drop('Exited'))
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.box(df, x='Exited', y=feature, 
                            title=f'{feature} par Statut Churn',
                            color='Exited',
                            color_discrete_sequence=['#1f77b4', '#ff4b4b'])
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.violin(df, x='Exited', y=feature,
                               title=f'Distribution de {feature}',
                               color='Exited',
                               color_discrete_sequence=['#1f77b4', '#ff4b4b'])
                st.plotly_chart(fig, use_container_width=True)
        
        # Tableau de données
        st.subheader("📋 Aperçu des Données")
        st.dataframe(df.head(100), use_container_width=True)
    else:
        st.warning("Données non disponibles. Vérifiez le fichier data/bank_churn.csv")

# ============================================================
# PAGE: DÉTECTION DE DRIFT
# ============================================================

elif page == "⚠️ Détection de Drift":
    st.markdown('<h1 class="main-header">⚠️ Détection de Data Drift</h1>', unsafe_allow_html=True)
    
    st.info("""
    Le **Data Drift** survient quand la distribution des données en production diffère 
    significativement des données d'entraînement, ce qui peut dégrader les performances du modèle.
    """)
    
    ref_data = load_data()
    
    # Charger ou générer les données de production
    if Path(PROD_DATA_PATH).exists():
        prod_data = pd.read_csv(PROD_DATA_PATH)
        st.success(f"✅ Données de production chargées: {len(prod_data)} lignes")
    else:
        st.warning("⚠️ Pas de données de production trouvées. Génération de données simulées...")
        if ref_data is not None:
            # Simuler un drift
            prod_data = ref_data.copy()
            prod_data['Age'] = prod_data['Age'] + np.random.normal(5, 2, len(prod_data))
            prod_data['Balance'] = prod_data['Balance'] * 1.2
            prod_data['CreditScore'] = prod_data['CreditScore'] - np.random.randint(0, 50, len(prod_data))
        else:
            prod_data = None
    
    if ref_data is not None and prod_data is not None:
        st.markdown("---")
        
        # Configuration
        col1, col2 = st.columns([1, 3])
        with col1:
            threshold = st.slider("Seuil de significativité", 0.01, 0.10, 0.05, 0.01)
        
        if st.button("🔍 Analyser le Drift", type="primary"):
            with st.spinner("Analyse en cours..."):
                drift_results = calculate_drift(ref_data, prod_data, threshold)
                
                # Résumé
                total_features = len(drift_results)
                drifted_features = sum(1 for r in drift_results.values() if r['drift_detected'])
                drift_percentage = (drifted_features / total_features) * 100
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Features Analysées", total_features)
                with col2:
                    st.metric("Features en Drift", drifted_features)
                with col3:
                    status = "🔴 CRITIQUE" if drift_percentage > 50 else "🟡 ATTENTION" if drift_percentage > 20 else "🟢 OK"
                    st.metric("Statut Global", status)
                
                st.markdown("---")
                
                # Détails par feature
                st.subheader("📊 Analyse Détaillée par Feature")
                
                drift_df = pd.DataFrame([
                    {
                        'Feature': feature,
                        'P-Value': f"{data['p_value']:.4f}",
                        'Statistique KS': f"{data['statistic']:.4f}",
                        'Drift Détecté': '🔴 Oui' if data['drift_detected'] else '🟢 Non'
                    }
                    for feature, data in drift_results.items()
                ])
                
                st.dataframe(drift_df, use_container_width=True)
                
                # Visualisation comparative
                st.subheader("📈 Comparaison des Distributions")
                
                drifted_cols = [f for f, r in drift_results.items() if r['drift_detected']]
                
                if drifted_cols:
                    selected_feature = st.selectbox("Feature à visualiser", drifted_cols)
                    
                    fig = make_subplots(rows=1, cols=2, subplot_titles=['Référence', 'Production'])
                    
                    fig.add_trace(
                        go.Histogram(x=ref_data[selected_feature], name='Référence', 
                                    marker_color='#1f77b4', opacity=0.7),
                        row=1, col=1
                    )
                    fig.add_trace(
                        go.Histogram(x=prod_data[selected_feature], name='Production',
                                    marker_color='#ff4b4b', opacity=0.7),
                        row=1, col=2
                    )
                    
                    fig.update_layout(title=f"Distribution de {selected_feature}", height=400)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Recommandations
                    st.subheader("💡 Actions Recommandées")
                    st.warning(f"""
                    **Drift détecté sur {len(drifted_cols)} feature(s): {', '.join(drifted_cols)}**
                    
                    Actions suggérées:
                    1. Vérifier la qualité des données entrantes
                    2. Analyser les changements business récents
                    3. Considérer un ré-entraînement du modèle
                    4. Mettre en place une surveillance continue
                    """)
                else:
                    st.success("✅ Aucun drift significatif détecté!")

# ============================================================
# PAGE: MÉTRIQUES DU MODÈLE
# ============================================================

elif page == "📈 Métriques du Modèle":
    st.markdown('<h1 class="main-header">📈 Métriques du Modèle</h1>', unsafe_allow_html=True)
    
    # Charger les métriques depuis MLflow
    mlflow_path = Path("mlruns/159076234787646138")
    
    runs = []
    if mlflow_path.exists():
        for run_dir in mlflow_path.iterdir():
            if run_dir.is_dir() and (run_dir / "metrics").exists():
                run_metrics = {}
                run_metrics['run_id'] = run_dir.name[:8]
                
                for metric_file in (run_dir / "metrics").iterdir():
                    try:
                        with open(metric_file, 'r') as f:
                            content = f.read().strip()
                            value = float(content.split()[1]) if ' ' in content else float(content)
                            run_metrics[metric_file.name] = value
                    except:
                        pass
                
                if run_metrics:
                    runs.append(run_metrics)
    
    if runs:
        # Dernière exécution
        latest_run = runs[-1]
        
        st.subheader("📊 Métriques du Dernier Entraînement")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        metrics_display = [
            ("Accuracy", "accuracy", col1),
            ("Precision", "precision", col2),
            ("Recall", "recall", col3),
            ("F1 Score", "f1_score", col4),
            ("ROC AUC", "roc_auc", col5)
        ]
        
        for name, key, col in metrics_display:
            with col:
                value = latest_run.get(key, 0)
                st.metric(name, f"{value:.4f}")
        
        st.markdown("---")
        
        # Graphique des métriques
        st.subheader("📈 Visualisation des Métriques")
        
        metrics_data = {
            'Métrique': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC'],
            'Valeur': [
                latest_run.get('accuracy', 0),
                latest_run.get('precision', 0),
                latest_run.get('recall', 0),
                latest_run.get('f1_score', 0),
                latest_run.get('roc_auc', 0)
            ]
        }
        
        fig = px.bar(metrics_data, x='Métrique', y='Valeur', 
                    title='Performance du Modèle',
                    color='Valeur',
                    color_continuous_scale='Viridis')
        fig.update_layout(yaxis_range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)
        
        # Historique des runs
        if len(runs) > 1:
            st.subheader("📜 Historique des Entraînements")
            runs_df = pd.DataFrame(runs)
            st.dataframe(runs_df, use_container_width=True)
        
        # Matrice de confusion (si image existe)
        if Path("confusion_matrix.png").exists():
            st.subheader("📊 Matrice de Confusion")
            st.image("confusion_matrix.png")
        
        # Feature importance (si image existe)
        if Path("feature_importance.png").exists():
            st.subheader("🎯 Importance des Features")
            st.image("feature_importance.png")
    else:
        st.warning("Aucune métrique MLflow trouvée. Exécutez d'abord `python train_model.py`")

# ============================================================
# PAGE: CONFIGURATION
# ============================================================

elif page == "🔧 Configuration":
    st.markdown('<h1 class="main-header">🔧 Configuration</h1>', unsafe_allow_html=True)
    
    st.subheader("📡 État des Services")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### API FastAPI")
        try:
            response = requests.get(f"{API_URL}/health", timeout=5)
            if response.status_code == 200:
                st.success(f"✅ API en ligne ({API_URL})")
                st.json(response.json())
            else:
                st.error(f"❌ API indisponible (status: {response.status_code})")
        except:
            st.warning(f"⚠️ API non accessible ({API_URL})")
            st.info("Démarrez l'API avec: `uvicorn app.main:app --reload`")
    
    with col2:
        st.markdown("### Modèle ML")
        if Path(MODEL_PATH).exists():
            model = load_model()
            if model:
                st.success(f"✅ Modèle chargé ({MODEL_PATH})")
                st.info(f"Type: {type(model).__name__}")
            else:
                st.error("❌ Erreur de chargement du modèle")
        else:
            st.warning(f"⚠️ Modèle non trouvé ({MODEL_PATH})")
            st.info("Entraînez le modèle avec: `python train_model.py`")
    
    st.markdown("---")
    
    st.subheader("📁 Fichiers de Données")
    
    data_files = [
        ("Données d'entraînement", DATA_PATH),
        ("Données de production", PROD_DATA_PATH),
        ("Modèle", MODEL_PATH)
    ]
    
    for name, path in data_files:
        if Path(path).exists():
            size = Path(path).stat().st_size / 1024
            st.success(f"✅ {name}: {path} ({size:.1f} KB)")
        else:
            st.warning(f"⚠️ {name}: {path} (non trouvé)")
    
    st.markdown("---")
    
    st.subheader("🔄 Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🗑️ Vider le Cache", use_container_width=True):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.success("Cache vidé!")
    
    with col2:
        if st.button("📊 Générer Données Prod", use_container_width=True):
            df = load_data()
            if df is not None:
                # Simuler des données de production avec drift
                prod_df = df.sample(frac=0.3).copy()
                prod_df['Age'] = prod_df['Age'] + np.random.normal(3, 2, len(prod_df))
                prod_df['Balance'] = prod_df['Balance'] * np.random.uniform(0.9, 1.3, len(prod_df))
                prod_df.to_csv(PROD_DATA_PATH, index=False)
                st.success(f"Données de production générées: {len(prod_df)} lignes")
    
    with col3:
        if st.button("📋 Voir les Logs", use_container_width=True):
            st.info("Consultez les logs dans le terminal ou Azure Application Insights")

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🏦 Bank Churn Prediction - MLOps Workshop</p>
    <p>Built with ❤️ using Streamlit, FastAPI & MLflow</p>
</div>
""", unsafe_allow_html=True)
