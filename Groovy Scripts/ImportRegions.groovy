/*
///////////////////////////////////////////////////////////////////////////////////
@author: Álvaro Fernández García
@date 19 de marzo de 2019
@brief: Script de QuPath para añadir a una imagen las regiones anotadas
en un fichero.
///////////////////////////////////////////////////////////////////////////////////
*/


import qupath.lib.objects.PathAnnotationObject
import qupath.lib.objects.classes.PathClassFactory
import qupath.lib.roi.RectangleROI
import qupath.lib.scripting.QPEx


// La siguiente variable deberá tener la ruta al fichero con extensión .pred
def filePath = ''

// Obtener una referencia a la imagen:
def imageData = QPEx.getCurrentImageData()

// Iterar sobre las líneas del fichero:
new File(filePath).eachLine { line ->

    // Extraer los datos de la línea
    String[] data = line.split()
    def label = data[0]
    def x = Float.parseFloat(data[1])
    def y = Float.parseFloat(data[2])
    def w = Float.parseFloat(data[3])
    def h = Float.parseFloat(data[4])

    // Crear la anotación:
    def roi = new RectangleROI(x, y, w, h)
    def annotation = new PathAnnotationObject(roi, PathClassFactory.getPathClass(label))

    // Añadirla a la imagen:
    imageData.getHierarchy().addPathObject(annotation, false)
}