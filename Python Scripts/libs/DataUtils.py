"""
##############################################################################################################
Author: Álvaro Fernández García
Date: 19 - Marzo - 2019
Description: Script que contiene una serie de funciones útiles para el proceso de construcción del dataset.
Contiene las funciones necesarias para importar y exportar anotaciones a QuPath.
Además, estas funciones están optimizadas gracias a numba
##############################################################################################################
"""


# Optimización:
from numba import jit, float32, int64, int32

# Módulos necesarios:
import os
import random
import cv2
import numpy as np
from scipy.misc import imresize
from scipy.ndimage import median_filter
from PIL import Image, ImageDraw


##############################################################################################################


def build_from_QuPath(img, filePath, classes, size):
    """ A partir de las anotaciones etiquetadas en QuPath devuelve tres ndArrays, conteniendo
    los parches, los labels de dichos parches y las coordenadas.
    * img -- La imagen de la que extraer los parches.
    * filePath -- Ruta hasta el fichero con las coordenadas de QuPath.
    * classes -- diccionario que tiene como clave los nombres de las clases de QuPath y como
    valor el entero positivo asociado a esa clase.
    * size -- Tamaño del parche.
    """

    # Función para convertir un string de un flotante a entero:
    def pars(x):
        return int(float(x))

    # Abrir el fichero para lectura:
    file = open(filePath, 'r')

    # Contar el número de parches que tiene el fichero de QuPath:
    n_patches = sum(1 for line in open(filePath, 'r'))
    
    # Declaramos los arrays necesarios para la salida:
    patches = np.empty(shape=(n_patches, size, size, 3), dtype=np.float32)
    labels = np.empty(shape=n_patches, dtype=np.int64)
    coords = np.empty(shape=(n_patches, 2), dtype=np.int64)

    # Para cada anotación en el fichero:
    for i,annotation in enumerate(file):
        # Quitar el \n del final y dividir el string:
        data = annotation.rstrip('\n').split(' ')

        # Separar la clase, x, y, altura y anchura del parche:
        c,x,y,w,h = data[0], pars(data[1]), pars(data[2]), pars(data[3]), pars(data[4])

        # Extraer el parche:
        patches[i] = img[y:y+h,x:x+w,:]

        # Añadir los datos a los otros arrays:
        labels[i] = classes[c]
        coords[i][0] = y
        coords[i][1] = x

    # Cerrar el fichero:
    file.close()

    # Devolver los datos:
    return patches, labels, coords


##############################################################################################################


def to_QuPath(coords, labels, dstPath, dict, size):
    """ Convierte los ndarrays 'coords' y 'labels' al formato necesario
    para poder dibujar las anotaciones en QuPath.
    Además guarda la salida en el fichero dstPath.
    * dict -- diccionario que contiene como claves los enteros que representan
    las etiquetas y como valor un string con la correspondiente clase en QuPath.
    * size -- El tamaño de los parches

    Nota: colocar como nombre etiqueta 'null' si no se desea asignar ninguna
    clase a la región.
    """

    # Abrir el fichero para escritura:
    file = open(dstPath, 'w')

    # Comenzar a escribirlo:
    for c,l in zip(coords, labels):
        cad = '%s %.2f %.2f %.2f %.2f\n' % \
            (dict[l], float(c[1]), float(c[0]), float(size), float(size))

        file.write(cad)

    # Cerrar el el fichero:
    file.close()


##############################################################################################################


# Función para extraer la región del parche como una nueva imagen:
@jit(nopython=True)
def __patch(nda, c, size):
    return nda[c[0]:c[0]+size, c[1]:c[1]+size, :]


##############################################################################################################


@jit(float32[:,:](float32[:,:,:]), parallel=True)
def build_mask(img):
    """ Construye una máscara binaria que permitirá no tener en
    cuenta el fondo de la imagen a la hora de extraer parches.
    img -- La imagen de la que extraer la máscara.
    """

    # Convertir la imagen a escala de grises (para ello nos quedamos con el canal G):
    mask = img[:,:,1]
    # Umbralizar:
    mask = -np.log(mask)
    mask = np.where(mask < 0.25, False, True)
    # Eliminar el ruido con un filtro de mediana:
    mask = median_filter(mask, 64)
    mask = np.where(mask, 1., 0.).astype(np.float32)

    return mask


@jit(nopython=True)
def extract(img, mask, size, stride):
    """ Extrae todos los parches de la imagen
    Devuelve los parches y sus coordenadas.
    img -- La imagen de la que extraer los parches.
    size -- Tamaño de cada parche.
    stride -- Salto entre parches adyacentes
    """

    # Calcular el offset necesario para hallar el centro del parche:
    offset = size // 2

    # Extraer las coordenadas iniciales:

    initial_coords = [
        (i,j) for i in range(0, img.shape[0]-size, stride) \
            for j in range(0, img.shape[1]-size, stride)
    ]

    # Declarar los arrays necesarios
    coords = np.empty(shape=(len(initial_coords), 2), dtype=np.int64)
    patches = np.empty((len(initial_coords), size, size, img.shape[2]), dtype=np.float32)

    length = 0
    for c in initial_coords:
        # Si el centro del parche cae dentro de la máscara:
        if mask[c[0] + offset, c[1] + offset] > 0.0:
            # Extraer el parche:
            patches[length] = __patch(img, c, size)
            # Añadir las coordenadas:
            coords[length][0] = c[0]
            coords[length][1] = c[1]

            length += 1

    # Eliminar los parches sobrantes:
    patches = patches[:length,:,:,:]
    coords = coords[:length,:]

    return patches, coords


##############################################################################################################


def draw_patches_of_class(img, coords, labels, class_dict, title, size):
    """ Dibuja sobre la imagen los parches de las etiquetas especificadas.
    Almacena el resultado en una nueva imagen.
    img -- La imagen original.
    coords -- ndarray con las coordenadas de los parches
    labels -- ndarray con las etiquetas de los parches
    class_dict -- diccionario con las etiquetas a dibujar como clave y el color
    como valor
    title -- El nombre para la imagen a guardar
    size -- El tamaño del parche
    """

    # Creamos el objeto PIL Image
    local = (img.copy() * 255).astype(np.uint8)
    pil = Image.fromarray(local)
    draw = ImageDraw.Draw(pil)

    # Ahora le añadimos los parches:
    for i,c in enumerate(coords):
        if labels[i] in class_dict.keys():
            rect = [c[1], c[0], c[1]+size+1, c[0]+size+1]
            draw.rectangle(rect, fill=None, outline=class_dict[labels[i]])

    # Guardar la imagen:
    pil.save(title)


##############################################################################################################


def data_augmentation(X, y):
    """ Dado el conjunto de entrada, añade cada parche rotado
    90, 180 o 270 grados (seleccionado de forma aleatoria).
    Es interesante porque añade más variabilidad a la CNN
    e introduce un poco de regularización.
    """

    out_x = np.empty(shape=(X.shape[0]*2, X.shape[1], X.shape[2], X.shape[3]), dtype=np.float32)
    out_y = np.empty(shape=y.shape[0]*2, dtype=np.int64)

    # Copiar X e y:
    out_x[:X.shape[0],:,:] = X
    out_y[:y.shape[0]] = y
    
    for i,x in enumerate(X):
        out_x[i+X.shape[0]] = np.rot90(x, random.randint(1,3))
        out_y[i+X.shape[0]] = y[i]
    
    return out_x, out_y


##############################################################################################################


def rescale(X, target_scale):
    """ Reescala los parches al nuevo tamaño.
    X -- Array de numpy con los parches.
    target_scale -- La nueva escala.
    """
    
    out_x = np.empty((X.shape[0], target_scale, target_scale, X.shape[3]), dtype=np.float32)

    for i, x in enumerate(X):
        tmp = imresize(x, (target_scale, target_scale, 3), interp='bilinear')
        tmp = (tmp / 255).astype(np.float32)
        out_x[i] = tmp
        
    return out_x


##############################################################################################################


@jit(float32[:,:,:,:](float32[:,:,:], float32[:,:,:,:], int64[:,:], int32, int32))
def add_scale(img, patches, coords, original_scale, new_scale):
    """ Añade a un conjunto de parches el mismo parche pero
    visto desde una nueva escala. El tamaño de los parches de 
    salida será (w,h, c + 3)
    img -- La imagen de la que se extrajeron los parches.
    patches -- Los parches.
    coords -- Las coordenadas de los parches.
    original_scale -- La escala que se utilizó en la extracción
    de los parches.
    new_scale -- El tamaño de la nueva escala a añadir.
    """

    out = np.empty(shape=(patches.shape[0], original_scale, original_scale, patches.shape[3]+3), 
        dtype=np.float32)

    # Clacular el offset a aplicar a las coordenadas para extraer las nuevas:
    offset = (new_scale - original_scale) // 2

    # Aplicar padding a la imagen original para evitar parches que caigan fuera de los
    # límites de la imagen:
    img2 = np.pad(img, ((offset+1,), (offset+1,), (0,)), 'edge')
    coords2 = coords + offset + 1

    for i, (p,c) in enumerate(zip(patches, coords2)):
        new_c = (c[0]-offset, c[1]-offset)

        # Extraer el nuevo parche:
        tmp = __patch(img2, new_c, new_scale)

        # Hacer un reshape del parche al tamaño original:
        tmp = imresize(tmp, (original_scale, original_scale, 3), interp='bilinear')
        tmp = (tmp / 255).astype(np.float32)

        # Añadirle la nueva resolución por detrás:
        new_p = np.concatenate((p, tmp), axis=2)

        # Añadirlo al resultado final:
        out[i] = new_p
        
    return out


##############################################################################################################


def zipImagesFiles(images, files):
    """ Genera una lista de tuplas emparejando cada imagen con su correspondiente
    archivo de QuPath.
    """

    data = []
    for img in os.listdir(images):
        qp = ""
        
        for file in os.listdir(files):
            if os.path.splitext(img)[0] == os.path.splitext(file)[0]:
                qp = file
                break

        if qp == "":
            raise ValueError(img + " has no QuPath file mate")

        data.append((os.path.join(images, img), os.path.join(files, qp)))
    
    return data


##############################################################################################################


def draw_regions(img, pred, coords, size, title):
    """ A partir de las predicciones hechas por el clasificador,
    dibuja en la imagen el contorno de las glándulas.
    img -- La imagen en la que dibujar el resultado.
    pred -- Las predicciones hechas por el clasificador.
    coords -- Las coordenadas de los parches.
    size -- El tamaño del parche.
    title -- Título de la imagen a guardar.
    """

    # Copiar la imagen para no modificar la original:
    out = (img.copy() * 255).astype(np.uint8)
    overlay = out.copy()

    # Nos quedamos con la probabilidad de que cada parche sea glándula:
    probs = pred[:,1].flatten()

    # A todos los píxeles de un parche le asignamos la misma probabilidad y para
    # combinarlas calculamos la media:

    # Sumar:
    mask = np.zeros(shape=(img.shape[0], img.shape[1]), dtype=img.dtype)
    for i,c in enumerate(coords):
        mask[c[0]:c[0]+size,c[1]:c[1]+size] = mask[c[0]:c[0]+size,c[1]:c[1]+size] + probs[i]

    # Dividir entre el número de contribuciones:
    mask /= 4.0

    # Umbralizamos:
    # Hallado en el proceso de validación cruzada con la curva ROC:
    OPTIMAL_THRESHOLD = 0.081414
    mask = np.where(mask < OPTIMAL_THRESHOLD, 0, 255).astype(np.uint8)

    # Hallar los contornos:
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Dibujarlos:
    opacity = 0.3
    cv2.fillPoly(overlay, pts=contours, color=(0,255,0))
    cv2.addWeighted(overlay, opacity, out, 1 - opacity, 0, out)
    cv2.drawContours(out, contours, -1, (0,255,0), 2)

    # Guardar la imagen:
    pil = Image.fromarray(out)
    pil.save(title)