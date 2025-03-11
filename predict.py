import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

def predict_image(img_path):
    # Chargement du modèle
    model = tf.keras.models.load_model('Plante_disease_model.h5')

    # Chargement des classes
    class_labels = ["Apple_Apple_scab", "Apple_Black_rot", "Apple_Cedar_apple_rust",
    "Cherry_(including_sour)Powdery_mildew", "Corn_(maize)Cercospora_leaf_spot Gray_leaf_spot", "Corn(maize)Common_rust",
    "Corn_(maize)Northern_Leaf_Blight", "Grape__Black_rot",
    "Grape_Esca(Black_Measles)", "Grape_Leaf_blight(Isariopsis_Leaf_Spot)",
    "Orange_Haunglongbing(Citrus_greening)", "Peach__Bacterial_spot", "Pepper,_bell_Bacterial_spot",
    "Potato_Early_blight", "Potato_Late_blight",
    "Saine", "Squash_Powdery_mildew", "Strawberry_Leaf_scorch",
    "Tomato_Bacterial_spot", "Tomato_Early_blight", "Tomato_Late_blight",
    "Tomato_Leaf_Mold", "Tomato_Septoria_leaf_spot", "Tomato_Spider_mites Two-spotted_spider_mite",
    "Tomato_Target_Spot", "Tomato_Tomato_mosaic_virus", "Tomato_Tomato_Yellow_Leaf_Curl_Virus",]  # Remplacez par vos classes réelles

    # Prétraitement de l'image
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalisation

   # Prédiction
prediction = model.predict(img_array)
predicted_class = np.argmax(prediction)  # Trouver la classe avec la probabilité la plus élevée

# Vérification si la plante est malade
maladie_detectee = class_labels[predicted_class]
if "healthy" in maladie_detectee.lower() or "Saine" in maladie_detectee:
    print(f"La plante est en bonne santé ({maladie_detectee})")
else:
    print(f"⚠️ La plante est malade : {maladie_detectee} ⚠️")
