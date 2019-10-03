#!/bin/bash
# Automatiza la realización del test sobre las distintas regiones:

mkdir ../../Datos/Test/Detections/

for file in ../../Datos/Test/*.png; do
    python Test.py -i "$file"
done

mv ../../Datos/Test/*_detection.png ../../Datos/Test/Detections/
