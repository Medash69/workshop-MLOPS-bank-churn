import pandas as pd
import numpy as np
import os

def generate_drifted_data(reference_file="data/bank_churn.csv", output_file="data/production_data.csv"):
    if not os.path.exists(reference_file):
        print(f"Erreur : {reference_file} introuvable.")
        return

    df = pd.read_csv(reference_file)
    # On simule un vieillissement de la population et une baisse des scores de crédit
    df['Age'] = df['Age'] + np.random.randint(5, 15, size=len(df))
    df['CreditScore'] = df['CreditScore'] - np.random.randint(50, 100, size=len(df))
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"✅ Données de drift générées dans {output_file}")

if __name__ == "__main__":
    # S'exécute depuis la racine du projet
    generate_drifted_data()