import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Désactive le conflit OpenMP
os.environ["OMP_NUM_THREADS"] = "1"  # Force une seule instance OpenMP

import tensorflow as tf  # Import après avoir défini les variables
from model_builder import build_model
from data_loader import load_data

def train_model():
    # Chargement des données
    train_data, test_data = load_data()
    
     # Définir le nombre de classes (par exemple, 27 classes)
    num_classes = 27  # Remplacez par la valeur correcte si nécessaire

    # Création du modèle
    model = build_model(num_classes)

    # Entraînement du modèle
    print("Entraînement du modèle...")
    history = model.fit(
        train_data,
        epochs=7,  # Nombre d'époques
        validation_data=test_data  # Données de validation
    )

    # Affichage des résultats de l'entraînement
    print("Entraînement terminé.")
    print("Historique:", history.history)  # Affichage des résultats d'entraînement (perte et précision)

    # Sauvegarde du modèle entraîné
    model.save('Plante_disease_model.h5')

    return model, history

if __name__ == "__main__":
    train_model()

