from train_model import train_model
from predict import predict_image

def main():
    # Étape 1 : Entraîner le modèle
    print("Début de l'entraînement du modèle...")
    model, history = train_model()
    print("Entraînement terminé.")

    # Étape 2 : Effectuer des prédictions (vous pouvez tester avec l'image de votre choix)
    img_path = 'dataset\train'  # Remplacez par le chemin de l'image à prédire
    print(f"Prédiction de l'image : {img_path}")
    predict_image(img_path)

if __name__ == "__main__":
    main()
