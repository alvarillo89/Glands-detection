"""
#########################################################################################################
Author: Álvaro Fernández García
Date: 21 - Marzo - 2019
Description: Contiene la definición para el modelo utilizado en la
detección de glándulas.
#########################################################################################################
"""


import os
import keras
import tensorflow as tf
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import BatchNormalization
from keras.applications import VGG16, InceptionV3
from keras.models import Model
from keras.layers.advanced_activations import LeakyReLU
from keras.regularizers import l2


def GlandsDetector(input_shape, output):
    """" input_shape -- Tupla con el tamaño de entrada para la red.
    output -- Número de clases para la salida.
    """

    # Ignorar warnings de tensorflow
    tf.logging.set_verbosity(tf.logging.ERROR)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

    # Crear el modelo:
    model = Sequential()
    model.add(Conv2D(filters=60, kernel_size=(3,3), padding='same', input_shape=input_shape))
    #model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2,2), padding='same'))
    model.add(Dropout(0.25))

    model.add(Conv2D(filters=100, kernel_size=(3,3), padding='same'))
    #model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2,2), padding='same'))
    model.add(Dropout(0.25))

    model.add(Conv2D(filters=150, kernel_size=(3,3), padding='same'))
    #model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2,2), padding='same'))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(150, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
    model.add(LeakyReLU(alpha=0.1))           
    model.add(Dropout(0.25))
    model.add(Dense(output, activation='softmax', kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))

    return model


def TransferLearning(input_shape, output, model, fineTune, tuneLayers=None):
    """ Esta función nos permite utilizar un modelo previamente entrenado
    en imagenet para nuestro problema:
    input_shape -- la entrada de la red. Debe tener obligatoriamente tres canales.
    output -- el número de clases a predecir por la red.
    model -- Que modelo cargar 'vgg' o 'inception'
    fineTune -- Booleano que indica si afinar todos los pesos de la red o sólo las
    últimas capas.
    tuneLayers -- Número de capas de la extracción de características a las que se 
    les aplicará ajuste fino (solo si fineTune = True). 
    """

    # Ignorar warnings de tensorflow
    tf.logging.set_verbosity(tf.logging.ERROR)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
    
    # Cargar la red:
    if model == 'vgg':
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    elif model == 'inception':
        base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=input_shape)
    else:
        raise ValueError("Unknow model name")

    # Añadir las últimas capas para nuestro problema:
    x = base_model.output
    x = Flatten()(x)

    # VGG output shape tras el flatten: 512
    if model == 'vgg':
        x = Dense(128, activation='relu', kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(x)
        x = Dropout(0.25)(x)
    else:
        # Inception output shape tras el flatten: 2048
        x = Dense(512, activation='relu', kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(x)
        x = Dropout(0.25)(x)

    predictions = Dense(output, activation='softmax', kernel_regularizer=l2(0.01), 
        bias_regularizer=l2(0.01))(x)

    # Crear el objeto de tipo modelo:
    model = Model(inputs=base_model.input, outputs=predictions)

    if not fineTune:
        # Congelar los pesos de las capas que no hemos añadido del modelo importado:
        for layer in base_model.layers:
            layer.trainable = False
    else:
        # Congelamos hasta N_layers - tuneLayers:
        for i in range(len(base_model.layers) - tuneLayers):
            base_model.layers[i].trainable = False

    return model