"""
#########################################################################################################
Author: Álvaro Fernández García
Date: 03 - Mayo - 2019
Description: Test final del clasficador encargado de detectar las Glándulas. Dibuja
en las imágenes los contornos de las glándulas detectadas.
#########################################################################################################
"""


import os
from libs.DataUtils import extract, add_scale, build_mask, draw_regions
from models.cnn import GlandsDetector
import argparse
import matplotlib.image as mpimg
import numpy as np


if __name__ == "__main__":
    # Procesar los argumentos pasados:
    parser = argparse.ArgumentParser(description='Detects Glands')
    parser.add_argument('-i', '--img', help='Image where detect glands', required=True)

    args = vars(parser.parse_args())

    # Leer la imagen de entrada:
    img = mpimg.imread(args['img'])

    # Extraer los parches de la misma:
    print("Extracting patches...")
    p, c = extract(img, mask=build_mask(img), size=32, stride=16)

    print("Total patches extracted:", p.shape[0], ", adding new scale...")

    p = add_scale(img, p, c, 32, 128)
  
    # Preprocesar los parches:
    mean = np.mean(p, axis=0)
    std = np.std(p, axis=0)
    p -= mean
    p /= std

    # Cargar el modelo:
    detector = GlandsDetector(input_shape=(32,32,6), output=2)
    detector.load_weights('./models/GlandsDetector.h5')

    # Detectar las glándulas:
    print('Detecting...')
    preds = detector.predict(p)

    # Dibujar las regiones:
    draw_regions(img, preds, c, 32, os.path.splitext(args['img'])[0] + '_detection.png')

    print("Done!")