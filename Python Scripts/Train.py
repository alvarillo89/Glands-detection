"""
#########################################################################################################
Author: Álvaro Fernández García
Date: 03 - Mayo - 2019
Description: Entrenamiento final del clasficador encargado de detectar las Glándulas.
#########################################################################################################
"""


# Utilidades:
import os
import numpy as np
import math
from tensorflow import set_random_seed

# Construcción de los datos:
import matplotlib.image as mpimg
from libs.DataUtils import build_from_QuPath, add_scale, data_augmentation, zipImagesFiles

# Preprocesado:
from sklearn.utils import compute_class_weight
from keras.utils import to_categorical

# Módulos para la red:
import keras
from models.cnn import GlandsDetector
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler

# Métricas:
import keras_metrics as km

#########################################################################################################

# Establecer la semilla aleatoria:
SEED = 12
np.random.seed(SEED)
set_random_seed(SEED)

#########################################################################################################

# Definir la función que aplica el learning rate decay:
def step_decay(epoch):
	initial_lrate = 0.001
	drop = 0.1
	epochs_drop = 25.0
	lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))

	return lrate

#########################################################################################################

IMG_FOLDER = '../../Datos/Train/Images/'
QPH_FOLDER = '../../Datos/Train/Qpdat/'

# Unir cada imagen con su archivo de QuPath:
data = zipImagesFiles(IMG_FOLDER, QPH_FOLDER)

# Construir el conjunto de train:
X_train = []
y_train = []

for image in data:
    img = mpimg.imread(image[0])
    p, l, c = build_from_QuPath(img, image[1], {'Stroma':0, 'Gland':1, 'Contaminated':2}, 32)
    p = add_scale(img, p, c, 32, 128)
    p, l = data_augmentation(p, l)
    X_train.append(p)
    y_train.append(l)

X_train = np.concatenate(X_train, axis=0)
y_train = np.concatenate(y_train, axis=0)

# Convertir el problema en binario:
y_train[y_train==2] = 0

### PREPROCESADO ###

# Dado que los datos están desbalanceados, calcularemos el peso de cada clase:
class_weights = compute_class_weight('balanced', np.unique(y_train), y_train)
print("\nClass Weights:", class_weights, '\n')

# Normalizar los conjuntos de train y val:
mean = np.mean(X_train, axis=0)
std = np.std(X_train, axis=0)

X_train -= mean
X_train /= std

# Aplicar one hot encoder a las etiquetas:
y_train_ohe = to_categorical(y_train)

### ENTRENAMIENTO ###

BATCH = 64
EPOCHS = 100

# Definir el modelo:
model = GlandsDetector(input_shape=(32,32,6), output=2)

# Definir el optimizador:
adam = Adam()

# Compilar el modelo:
model.compile(
    loss='binary_crossentropy',
    optimizer=adam,
    metrics=['binary_accuracy', km.binary_f1_score(), km.binary_precision(), km.binary_recall()]
)

callback_list = [LearningRateScheduler(step_decay)]

# Entrenar:
history = model.fit(
    X_train, 
    y_train_ohe, 
    batch_size=BATCH, 
    epochs=EPOCHS, 
    class_weight=class_weights,
    callbacks=callback_list,
    verbose=2
)

# Guardar el modelo:
model.save('models/GlandsDetector.h5')
