import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import tkinter as tk
from tkinter import filedialog, Label, Button
from PIL import Image, ImageTk

# Charger le modèle
model = tf.keras.models.load_model('Plante_disease_model.h5')

# Liste des classes
class_labels = ["Apple_Apple_scab", "Apple_Black_rot", "Apple_Cedar_apple_rust",
    "Cherry_(including_sour)Powdery_mildew", "Corn_(maize)Cercospora_leaf_spot Gray_leaf_spot", "Corn(maize)Common_rust",
    "Corn_(maize)Northern_Leaf_Blight", "Grape__Black_rot",
    "Grape_Esca(Black_Measles)", "Grape_Leaf_blight(Isariopsis_Leaf_Spot)",
    "Orange_Haunglongbing(Citrus_greening)", "Peach__Bacterial_spot", "Pepper,_bell_Bacterial_spot",
    "Potato_Early_blight", "Potato_Late_blight",
    "Saine", "Squash_Powdery_mildew", "Strawberry_Leaf_scorch",
    "Tomato_Bacterial_spot", "Tomato_Early_blight", "Tomato_Late_blight",
    "Tomato_Leaf_Mold", "Tomato_Septoria_leaf_spot", "Tomato_Spider_mites Two-spotted_spider_mite",
    "Tomato_Target_Spot", "Tomato_Tomato_mosaic_virus", "Tomato_Tomato_Yellow_Leaf_Curl_Virus",
    ]

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    prediction = model.predict(img_array)[0]
    predicted_class = np.argmax(prediction)
    confidence = prediction[predicted_class] * 100
    disease = class_labels[predicted_class]
    health_status = "en bonne santé" if "healthy" in disease.lower() or "Saine" in disease else "malade"
    return disease, confidence, health_status

def open_file():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", ".jpg;.png;*.jpeg")])
    if file_path:
        img = Image.open(file_path)
        img = img.resize((200, 200))
        img = ImageTk.PhotoImage(img)
        img_label.config(image=img)
        img_label.image = img
        disease, confidence, status = predict_image(file_path)
        result_label.config(text=f"La plante est {status} \n{disease} ({confidence:.2f}%)")

# Interface graphique
root = tk.Tk()
root.title("Détection des maladies des plantes")
root.geometry("400x500")

btn = Button(root, text="Choisir une image", command=open_file, font=("Arial", 14))
btn.pack(pady=20)

img_label = Label(root)
img_label.pack()

result_label = Label(root, text="", font=("Arial", 14), wraplength=300)
result_label.pack(pady=20)

root.mainloop()