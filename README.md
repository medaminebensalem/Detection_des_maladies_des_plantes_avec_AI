# Detection_des_maladies_des_plantes_avec_AI
# Rapport de Projet : Détection des Maladies des Plantes avec CNN

## 1. Introduction
La détection précoce des maladies des plantes est essentielle pour assurer une agriculture durable et réduire les pertes de production. L'utilisation des techniques de Deep Learning, notamment les réseaux de neurones convolutionnels (CNN), permet d'automatiser ce processus avec une grande précision. Ce projet vise à développer un modèle de CNN capable de classifier les plantes en bonne santé et celles affectées par des maladies.

## 2. Chargement et Prétraitement des Images

### 2.1 Chargement des Images
Les données proviennent du jeu de données *PlantVillage*, qui contient des images de plantes avec différentes maladies ainsi que des plantes saines. Ces images seront utilisées pour entraîner et tester le modèle.

### 2.2 Prétraitement des Images
Les images subissent plusieurs étapes de prétraitement :
- **Redimensionnement** : Ajustement à une taille standard (ex: 128x128 pixels) pour assurer une uniformité des entrées.
- **Normalisation** : Mise à l'échelle des valeurs des pixels entre 0 et 1 pour améliorer l'apprentissage.
- **Augmentation des données** : Rotation, retournement et ajustement de luminosité pour augmenter la robustesse du modèle.

## 3. Construction du Modèle CNN
Le modèle de réseau de neurones convolutionnel (CNN) est conçu avec les couches suivantes :
- **Convolution 2D** : Extraction des caractéristiques visuelles.
- **MaxPooling** : Réduction de la dimension des caractéristiques pour diminuer la complexité.
- **Couches denses (Fully Connected)** : Classification finale des images.
- **Fonction d'activation ReLU** : Introduire la non-linéarité.
- **Softmax** : Prédiction des probabilités d'appartenance à chaque classe.

## 4. Entraînement du Modèle
Le modèle est entraîné sur un ensemble de données divisé en :
- **Ensemble d'entraînement (80%)**
- **Ensemble de validation (20%)**

L'optimiseur *Adam* et la fonction de perte *categorical crossentropy* sont utilisés pour améliorer l'apprentissage. Plusieurs époques d'entraînement sont réalisées jusqu'à atteindre une bonne précision.

## 5. Test et Évaluation
Le modèle est testé sur un ensemble d'images inédites pour mesurer sa précision. Les métriques suivantes sont utilisées :
- **Précision (Accuracy)** : Indique le taux de classification correcte.
- **Matrice de confusion** : Visualisation des erreurs de classification.
- **F1-score** : Équilibre entre précision et rappel.

## 6. Prédiction sur une Nouvelle Image
Une image externe est prétraitée et passée à travers le modèle pour obtenir une classification. L'utilisateur peut ainsi obtenir un diagnostic immédiat sur l'état de la plante.

## 7. Conclusion
Ce projet démontre l'efficacité du Deep Learning pour la détection des maladies des plantes. En améliorant l'architecture du modèle et en utilisant des ensembles de données plus variés, il est possible d'obtenir des performances encore meilleures.

