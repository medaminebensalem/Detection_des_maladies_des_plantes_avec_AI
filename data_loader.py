import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_data():
    # Création d'un générateur d'images pour l'entraînement avec des augmentations d'images
    train_datagen = ImageDataGenerator(
        rescale=1./255,  # Mise à l'échelle des pixels de l'image entre 0 et 1
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )
    
    # Création d'un générateur pour les données de test sans augmentation
    test_datagen = ImageDataGenerator(rescale=1./255)

    # Chargement des données depuis le répertoire 'dataset'
    train_data = train_datagen.flow_from_directory(
    'dataset/train',
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'  # Classification multi-classes
)
    
    test_data = test_datagen.flow_from_directory(
    'dataset/test',
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'  # Classification multi-classes
)
    
    return train_data, test_data
