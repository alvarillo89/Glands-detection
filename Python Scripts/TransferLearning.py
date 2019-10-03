"""
#########################################################################################################
Author: Álvaro Fernández García
Date: 4 - Junio - 2019
Description: Entrenamiento y validación de modelos entrenados en imagenet 
y aplicados a la detección de las Glándulas.
#########################################################################################################
"""

# Utilidades:
import math
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import set_random_seed

# Construcción de los datos:
import matplotlib.image as mpimg
from libs.DataUtils import build_from_QuPath, rescale, data_augmentation, zipImagesFiles

# Preprocesado:
from sklearn.utils import compute_class_weight
from keras.utils import to_categorical

# Módulos para la red:
import keras
from models.cnn import TransferLearning
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler

# Métricas:
import keras_metrics as km
from sklearn.metrics import confusion_matrix

#########################################################################################################

# Establecer la semilla aleatoria:
SEED = 12
np.random.seed(SEED)
set_random_seed(SEED)

#########################################################################################################

# Variables de control para elegir el modelo: solo una puede estar a true:
define_vgg = True
define_inception = False

# Si realizar finetune o no:
define_finetune = False
define_tuneLayers = 4

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

# 9 imágenes para entrenar, 2 para validar:
data_train = data[:len(data)-2]
data_val = data[len(data)-2:]

# Construir el conjunto de train:
X_train = []
y_train = []

for image in data_train:
    img = mpimg.imread(image[0])
    p, l, _ = build_from_QuPath(img, image[1], {'Stroma':0, 'Gland':1, 'Contaminated':2}, 32)
    p, l = data_augmentation(p, l)

    if define_inception:
        p = rescale(p, 75)
    
    X_train.append(p)
    y_train.append(l)

X_train = np.concatenate(X_train, axis=0)
y_train = np.concatenate(y_train, axis=0)

# Convertir el problema en binario:
y_train[y_train==2] = 0

# Construir el conjunto de validación:
X_val = []
y_val = []

for image in data_val:
    img = mpimg.imread(image[0])
    p, l, _ = build_from_QuPath(img, image[1], {'Stroma':0, 'Gland':1, 'Contaminated':2}, 32)
    
    if define_inception:
        p = rescale(p, 75)
    
    X_val.append(p)
    y_val.append(l)

X_val = np.concatenate(X_val, axis=0)
y_val = np.concatenate(y_val, axis=0)

# Convertir el problema en binario:
y_val[y_val==2] = 0

print(X_train.shape, y_train.shape)
print(X_val.shape, y_val.shape)

### PREPROCESADO ###

# Dado que los datos están desbalanceados, calcularemos el peso de cada clase:
class_weights = compute_class_weight('balanced', np.unique(y_train), y_train)
print("\nClass Weights:", class_weights, '\n')

# Normalizar los conjuntos de train y val:
mean = np.mean(X_train, axis=0)
std = np.std(X_train, axis=0)

X_train -= mean
X_train /= std

X_val -= mean
X_val /= std

# Aplicar one hot encoder a las etiquetas:
y_train_ohe = to_categorical(y_train)
y_val_ohe = to_categorical(y_val)

### ENTRENAMIENTO ###

BATCH = 64

if define_finetune:
    EPOCHS = 25
else:
    EPOCHS = 50

# Definir el modelo:
if define_vgg:
    model = TransferLearning(input_shape=(32,32,3), output=2, model='vgg', 
        fineTune=define_finetune, tuneLayers=define_tuneLayers)

if define_inception:
    model = TransferLearning(input_shape=(75,75,3), output=2, model='inception', 
        fineTune=define_finetune, tuneLayers=define_tuneLayers)
 
# Definir el optimizador:
adam = Adam()

# Compilar el modelo:
model.compile(
    loss='binary_crossentropy',
    optimizer=adam,
    metrics=['binary_accuracy', km.binary_f1_score(label=1), 
        km.binary_precision(label=1), km.binary_recall(label=1)]
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
    validation_data=(X_val, y_val_ohe),
    verbose=2
)

# Guardar el modelo:
model.save('models/TransferLearningModel.h5')

# Mostrar estadísticas:
x_axis = np.arange(1,EPOCHS+1)
plt.plot(x_axis, np.array(history.history['loss']), label='Train Loss')
plt.plot(x_axis, np.array(history.history['val_loss']), label='Val Loss')
plt.plot(x_axis, np.array(history.history['binary_accuracy']), label='Train Acc')
plt.plot(x_axis, np.array(history.history['val_binary_accuracy']), label='Val Acc')
plt.xlabel("Epochs")
plt.ylabel("Accuracy / Loss")
plt.title("Train and validation metrics over epochs")
plt.grid(True)
plt.legend()
plt.savefig("TransferLearningTrainCurve.png")

scores = model.evaluate(X_val, y_val_ohe)

print("Loss Validation: %.4f" % (scores[0]))
print("Accuracy Validation: %.4f" % (scores[1]))
print("F1 Score Validation: %.4f" % (scores[2]))
print("Precision Validation: %.4f" % (scores[3]))
print("Recall Validation: %.4f" % (scores[4]))

print(confusion_matrix(y_val, (model.predict(X_val)).argmax(axis=-1)))
