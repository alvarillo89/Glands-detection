/*
///////////////////////////////////////////////////////////////////////////////////
@author: Álvaro Fernández García
@date 19 de marzo de 2019
@brief: Script de QuPath para extraer las coordenadas de las regiones anotadas
junto con la clase asociada. Guarda el resultado en un fichero de texto con
extensión .qpdat
///////////////////////////////////////////////////////////////////////////////////
*/


import qupath.lib.scripting.QPEx


// Obtenemos el nombre de la imagen:
def imgPath = QPEx.getCurrentImageData().getServerPath()
def lastStash = imgPath.lastIndexOf("/")
def fileName = imgPath.substring(lastStash + 1)
def lastDot = fileName.lastIndexOf('.')
fileName = fileName.substring(0, lastDot)

// Añadir la extensión:
fileName += '.qpdat'

// Declaramos el string que contendrá las coordenadas:
def coords = ''

// Iterar sobre las anotaciones de la imagen:
for (annotation in getAnnotationObjects()) 
{
    // Extraer la clase de la anotación:
    def pathClass = annotation.getPathClass()

    // Solo la añadimos si la clase es distinta de null:
    if(pathClass != null)
    {
        // Extraer la región:
        def roi = annotation.getROI()

        // Añadirle la información del parche a la salida:
        coords += String.format('%s %.2f %.2f %.2f %.2f',
            pathClass, roi.getBoundsX(), roi.getBoundsY(), roi.getBoundsWidth(), roi.getBoundsHeight())
        
        // Añadir un salto de línea:
        coords += System.lineSeparator()
    }
}

// Guardar las coordenadas en un fichero:
def file = new File('../../' + fileName)
file.text = coords