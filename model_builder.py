import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

def build_model(num_classes):
    # Création du modèle séquentiel
    model = Sequential()
    
    # Première couche convolutionnelle
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    # Deuxième couche convolutionnelle
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    # Troisième couche convolutionnelle
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    # Aplatissement des données pour la couche dense
    model.add(Flatten())
    
    # Première couche dense
    model.add(Dense(512, activation='relu'))
    
    # Ajout d'une couche Dropout pour éviter le surapprentissage
    model.add(Dropout(0.5))
    
    # Dernière couche dense avec `27` neurones pour classification multi-classes
    model.add(Dense(num_classes, activation='softmax'))
    
    # Compilation du modèle avec l'optimiseur 'adam' et la perte 'categorical_crossentropy'
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model