"""
Module de détection de Data Drift
Utilise le test de Kolmogorov-Smirnov pour détecter les changements de distribution
"""

import matplotlib
matplotlib.use("Agg")  # OBLIGATOIRE pour Docker / Azure

import pandas as pd
import numpy as np
from scipy.stats import ks_2samp, chi2_contingency
import json
import os
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logger = logging.getLogger(__name__)


class DriftDetector:
    """Classe pour la détection de data drift"""
    
    def __init__(self, threshold: float = 0.05, output_dir: str = "drift_reports"):
        self.threshold = threshold
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def ks_test(self, ref_data: np.ndarray, prod_data: np.ndarray) -> Tuple[float, float]:
        """Test de Kolmogorov-Smirnov pour variables continues"""
        stat, p_value = ks_2samp(ref_data, prod_data)
        return float(stat), float(p_value)
    
    def chi2_test(self, ref_data: pd.Series, prod_data: pd.Series) -> Tuple[float, float]:
        """Test du Chi-2 pour variables catégorielles"""
        try:
            # Créer la table de contingence
            ref_counts = ref_data.value_counts()
            prod_counts = prod_data.value_counts()
            
            all_categories = set(ref_counts.index) | set(prod_counts.index)
            contingency = pd.DataFrame({
                'ref': [ref_counts.get(cat, 0) for cat in all_categories],
                'prod': [prod_counts.get(cat, 0) for cat in all_categories]
            })
            
            chi2, p_value, _, _ = chi2_contingency(contingency.T)
            return float(chi2), float(p_value)
        except Exception as e:
            logger.warning(f"Chi2 test failed: {e}")
            return 0.0, 1.0
    
    def detect_single_feature(
        self, 
        ref_data: pd.Series, 
        prod_data: pd.Series,
        is_categorical: bool = False
    ) -> Dict[str, Any]:
        """Détecte le drift pour une seule feature"""
        
        ref_clean = ref_data.dropna()
        prod_clean = prod_data.dropna()
        
        if len(ref_clean) == 0 or len(prod_clean) == 0:
            return {
                "drift_detected": False,
                "error": "Données insuffisantes"
            }
        
        if is_categorical or ref_data.dtype == 'object':
            stat, p_value = self.chi2_test(ref_clean, prod_clean)
            test_type = "chi2"
        else:
            stat, p_value = self.ks_test(ref_clean.values, prod_clean.values)
            test_type = "ks"
        
        return {
            "p_value": p_value,
            "statistic": stat,
            "test_type": test_type,
            "drift_detected": p_value < self.threshold,
            "ref_mean": float(ref_clean.mean()) if not is_categorical else None,
            "prod_mean": float(prod_clean.mean()) if not is_categorical else None,
            "ref_std": float(ref_clean.std()) if not is_categorical else None,
            "prod_std": float(prod_clean.std()) if not is_categorical else None,
            "ref_count": len(ref_clean),
            "prod_count": len(prod_clean)
        }
    
    def detect_all(
        self, 
        ref_df: pd.DataFrame, 
        prod_df: pd.DataFrame,
        exclude_columns: list = None
    ) -> Dict[str, Dict[str, Any]]:
        """Détecte le drift pour toutes les features"""
        
        if exclude_columns is None:
            exclude_columns = ['Exited', 'target', 'label']
        
        results = {}
        
        for col in ref_df.columns:
            if col in exclude_columns or col not in prod_df.columns:
                continue
            
            is_categorical = ref_df[col].nunique() < 10
            results[col] = self.detect_single_feature(
                ref_df[col], 
                prod_df[col],
                is_categorical
            )
        
        return results
    
    def generate_report(
        self, 
        results: Dict[str, Dict[str, Any]],
        save: bool = True
    ) -> Dict[str, Any]:
        """Génère un rapport de drift complet"""
        
        total_features = len(results)
        drifted_features = [f for f, r in results.items() if r.get("drift_detected", False)]
        drift_percentage = (len(drifted_features) / total_features * 100) if total_features > 0 else 0
        
        # Déterminer le niveau de risque
        if drift_percentage > 50:
            risk_level = "CRITICAL"
            recommendation = "Ré-entraînement du modèle fortement recommandé"
        elif drift_percentage > 30:
            risk_level = "HIGH"
            recommendation = "Surveillance accrue et analyse approfondie requise"
        elif drift_percentage > 10:
            risk_level = "MEDIUM"
            recommendation = "Continuer la surveillance, planifier une analyse"
        else:
            risk_level = "LOW"
            recommendation = "Situation normale, maintenir la surveillance"
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_features": total_features,
                "drifted_features_count": len(drifted_features),
                "drifted_features": drifted_features,
                "drift_percentage": round(drift_percentage, 2),
                "risk_level": risk_level,
                "recommendation": recommendation,
                "threshold_used": self.threshold
            },
            "details": results
        }
        
        if save:
            report_path = f"{self.output_dir}/drift_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_path, "w") as f:
                json.dump(report, f, indent=2)
            logger.info(f"Drift report saved to {report_path}")
        
        return report
    
    def plot_drift_comparison(
        self,
        ref_df: pd.DataFrame,
        prod_df: pd.DataFrame,
        feature: str,
        save_path: Optional[str] = None
    ) -> None:
        """Génère un graphique comparatif pour une feature"""
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # Histogramme
        axes[0].hist(ref_df[feature].dropna(), alpha=0.5, label='Référence', bins=30, density=True)
        axes[0].hist(prod_df[feature].dropna(), alpha=0.5, label='Production', bins=30, density=True)
        axes[0].set_title(f'{feature} - Distribution')
        axes[0].legend()
        
        # Box plot
        data_to_plot = [ref_df[feature].dropna(), prod_df[feature].dropna()]
        bp = axes[1].boxplot(data_to_plot, labels=['Référence', 'Production'], patch_artist=True)
        bp['boxes'][0].set_facecolor('lightblue')
        bp['boxes'][1].set_facecolor('lightcoral')
        axes[1].set_title(f'{feature} - Box Plot')
        
        # KDE
        ref_df[feature].dropna().plot(kind='kde', ax=axes[2], label='Référence', color='blue')
        prod_df[feature].dropna().plot(kind='kde', ax=axes[2], label='Production', color='red')
        axes[2].set_title(f'{feature} - Density')
        axes[2].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_drift_summary(
        self,
        results: Dict[str, Dict[str, Any]],
        save_path: Optional[str] = None
    ) -> None:
        """Génère un graphique résumé du drift"""
        
        features = list(results.keys())
        p_values = [results[f].get('p_value', 1.0) for f in features]
        colors = ['red' if results[f].get('drift_detected', False) else 'green' for f in features]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        bars = ax.barh(features, p_values, color=colors, alpha=0.7)
        ax.axvline(x=self.threshold, color='black', linestyle='--', label=f'Seuil ({self.threshold})')
        
        ax.set_xlabel('P-Value')
        ax.set_title('Détection de Drift par Feature')
        ax.legend()
        
        # Ajouter les annotations
        for bar, p_val in zip(bars, p_values):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{p_val:.4f}', va='center', fontsize=8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


def detect_drift(
    reference_file: str, 
    production_file: str, 
    threshold: float = 0.05, 
    output_dir: str = "drift_reports"
) -> Dict[str, Dict[str, Any]]:
    """
    Fonction de compatibilité avec l'ancienne API
    Détecte le drift entre les données de référence et de production
    """
    os.makedirs(output_dir, exist_ok=True)
    
    ref = pd.read_csv(reference_file)
    prod = pd.read_csv(production_file)
    
    detector = DriftDetector(threshold=threshold, output_dir=output_dir)
    results = detector.detect_all(ref, prod)
    
    # Sauvegarder le rapport
    report_path = f"{output_dir}/drift_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
    
    return results


def generate_drift_data(
    reference_file: str,
    output_file: str,
    drift_features: Dict[str, Dict[str, float]] = None,
    sample_size: int = None
) -> pd.DataFrame:
    """
    Génère des données de production simulées avec drift
    
    Args:
        reference_file: Chemin vers les données de référence
        output_file: Chemin de sortie pour les données générées
        drift_features: Dict avec les modifications à appliquer
            Ex: {"Age": {"shift": 5, "scale": 1.1}, "Balance": {"scale": 1.2}}
        sample_size: Nombre de lignes à générer (None = même taille que ref)
    """
    ref = pd.read_csv(reference_file)
    
    if sample_size:
        prod = ref.sample(n=min(sample_size, len(ref)), replace=True).copy()
    else:
        prod = ref.copy()
    
    if drift_features is None:
        # Drift par défaut
        drift_features = {
            "Age": {"shift": 5, "scale": 1.0},
            "Balance": {"shift": 0, "scale": 1.2},
            "CreditScore": {"shift": -30, "scale": 1.0}
        }
    
    for feature, params in drift_features.items():
        if feature in prod.columns:
            shift = params.get("shift", 0)
            scale = params.get("scale", 1.0)
            noise = params.get("noise", 0)
            
            prod[feature] = prod[feature] * scale + shift
            if noise > 0:
                prod[feature] += np.random.normal(0, noise, len(prod))
    
    prod.to_csv(output_file, index=False)
    logger.info(f"Production data with drift saved to {output_file}")
    
    return prod