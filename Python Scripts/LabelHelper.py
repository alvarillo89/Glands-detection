"""
#########################################################################################################
Author: Álvaro Fernández García
Date: 21 - Marzo - 2019
Description: Este script permite utilizar el clasificador GlandsDetector para 
obtener una ayuda en el etiquetado de los datos. Permite clasificar una imagen y exportar los resultados
en un formato entendible por QuPath, en el que posteriormente se corregirán aquellas etiquetas
incorrectas. También permite alimentar el clasificador con nuevos datos  para refinar la predicción y 
hacerlo más útil.
#########################################################################################################
"""

import os
import numpy as np
from tensorflow import set_random_seed
from libs.DataUtils import data_augmentation, extract, build_mask, zipImagesFiles
from libs.DataUtils import draw_patches_of_class, build_from_QuPath, to_QuPath
from models.cnn import GlandsDetector
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.models import load_model
import matplotlib.image as mpimg
    
#########################################################################################################


def predict(imagePath, model, out='qupath'):
    """ Realiza la predicción para una imagen.
    imagePath -- Ruta a la imagen que etiquetar/dibujar.
    model -- Ruta al modelo a utilizar
    Si out = 'qupath' extrae el fichero para poder importar los resultados.
    Si out = 'draw' dibuja las etiquetas predichas por el modelo sobre la imagen.
    """

    # Obtener el nombre de la imagen:
    base = os.path.basename(imagePath)
    filename = os.path.splitext(base)[0]

    # Cargar el modelo:
    cnn = GlandsDetector(input_shape=(32,32,3), output=3)
    cnn.load_weights(model)

    # Cargar la imagen de test:
    img = mpimg.imread(imagePath)

    # Extraer los parches:
    if out == 'qupath':
        p, c = extract(img, mask=build_mask(img), size=32, stride=32)
    elif out == 'draw':
        p, c = extract(img, mask=build_mask(img), size=32, stride=16)
    else:
        raise ValueError("Unknow value for param out")

    # Preprocesar:
    mean = np.mean(p, axis=0)
    std = np.std(p, axis=0)
    p -= mean
    p /= std

    # Predict:
    l = cnn.predict_classes(p)

    if out == 'qupath':
        # Pasar los resultados a QuPath:
        classes = {0:'Stroma', 1:'Gland', 2:'Contaminated'}
        to_QuPath(c, l, filename + '.pred', classes, 32)
    else:
        # Dibujar:
        colors = {1:(255,0,0), 2:(0,0,255)}
        draw_patches_of_class(img, c, l, colors, filename + '_draw.png', 32)


#########################################################################################################


def feed_label_helper(imgPath, qpPath, model=None):
    """ Entrena el clasificador con los nuevos datos dados a partir de imagenes y quPathFiles.
    imgPath -- Ruta al directorio que contiene las imágenes.
    qpPath -- Ruta al directorio que contiene los archivos exportados de QuPath.
    Si model = None, crea un clasificador vacío.
    Guarda el nuevo modelo en el directorio actual.
    """

    # Establecer la semilla para numpy y tf:
    SEED = 12
    np.random.seed(SEED)
    set_random_seed(SEED)

    classes = {'Stroma':0, 'Gland':1, 'Contaminated':2}

    patches = []
    labels = []
    data = zipImagesFiles(imgPath, qpPath)

    for (image, qpf) in data:
        img = mpimg.imread(image)
        tmp_p,tmp_l,_ = build_from_QuPath(img, qpf, classes, 32)
        patches.append(tmp_p)
        labels.append(tmp_l)
    
    # Concatenar:
    p = np.concatenate(patches, axis=0)
    l = np.concatenate(labels, axis=0)

    # Data Augmentation:
    p, l = data_augmentation(p, l)
    l = to_categorical(l)

    # Preprocesar:
    mean = np.mean(p, axis=0)
    std = np.std(p, axis=0)
    p -= mean
    p /= std

    # Cargar el modelo:
    cnn = GlandsDetector(input_shape=(32,32,3), output=3)

    if model != None:
        cnn = load_model(model)
    else:
        cnn.compile(
            loss = 'categorical_crossentropy',
            optimizer = Adam(),
            metrics = ['categorical_accuracy']
        )

    # Entrenar:
    cnn.fit(p, l, batch_size=64, epochs=25)

    # Guardar el modelo:
    cnn.save('./new_LabelHelper.h5')


#########################################################################################################


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Help with data labeling")
    subparser = parser.add_subparsers(dest='feed or pred')
    subparser.required = True

    feed = subparser.add_parser('feed', help="Feeds the classifier with new examples for make" 
        + " it more accurate")
    feed.add_argument('--imgs', required=True, help="Path to images folder")
    feed.add_argument('--qpf', required=True, help="Path to QuPath exported regions files folder")
    feed.add_argument('--model', help="Model weights that will be used." 
        + " If not specified, an empty model will be used")

    pred = subparser.add_parser('pred', help="Predicts the labels of the input image")
    group = pred.add_mutually_exclusive_group(required=True)
    pred.add_argument('--img', required=True, help="Input image")
    pred.add_argument('--model', required=True, help="Model to be used")
    group.add_argument('--qupath', action='store_true', help="Export predicted labels to QuPath file")
    group.add_argument('--draw', action='store_true', help="Draw predicted labels on image")

    args = vars(parser.parse_args())
    
    if args['feed or pred'] == 'feed':
        feed_label_helper(args['imgs'], args['qpf'], args['model'])
    elif args['feed or pred'] == 'pred':
        if args['qupath']:
            predict(args['img'], args['model'])
        elif args['draw']:
            predict(args['img'], args['model'], 'draw')